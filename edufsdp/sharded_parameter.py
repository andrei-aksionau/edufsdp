"""
Parameter sharding abstraction for Educational FSDP.

Provides `ShardedParameter` to wrap standard PyTorch `nn.Parameter` instances,
managing:
  1. Sharding original weight tensors across a distributed `DeviceMesh`.
  2. Symmetric state transitions (`set_unsharded` <-> `set_sharded`).
  3. Memory cleanup and local vs. full tensor attribute lifecycle.

Parameter lifecycle:

                 ┌──────────────────────────────────────────────┐
                 │                SHARDED STATE                 │
                 │  module.weight/bias = DTensor (1/N memory)   │
                 └──────────────────┬───────────────────────────┘
                                    │       ▲
                    set_unsharded() │       │ set_sharded()
                                    ▼       │
                 ┌──────────────────────────┴───────────────────┐
                 │               GATHERED STATE                 │
                 │  module.weight/bias = Tensor (Full memory)   │
                 └──────────────────────────────────────────────┘
"""

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class ShardedParameter:
    """
    Manages parameter shard metadata and tensor state transitions on a target `nn.Module`.

    This class acts as a bridge between PyTorch's `DTensor` sharding primitives and the
    underlying `nn.Module` execution graph.

    Key Responsibilities:
      1. Shards full parameter data across a `DeviceMesh` along dimension 0 (`Shard(dim=0)`).
      2. Dynamically swaps module attributes between local `DTensor` shards and full
         unsharded `Tensor` instances during forward/backward steps.
      3. Manages persistent metadata shape shells with lazy storage allocation to avoid
         frequent memory allocation overhead and memory leakage.
    """

    def __init__(
        self,
        fqn: str,
        module: nn.Module,
        parameter: nn.Parameter,
        device_mesh: DeviceMesh,
    ) -> None:
        """
        Initializes parameter sharding state and installs the sharded DTensor onto the module.

        Args:
            fqn: Fully qualified parameter name (e.g., 'layer1.weight').
            module: Parent PyTorch module owning this parameter.
            parameter: Unsharded source parameter to split across ranks.
            device_mesh: PyTorch DeviceMesh defining rank layout.
        """
        self.fqn = fqn
        self.module = module
        self.device_mesh = device_mesh

        # ---------------------------------------------------------------------
        # 1. Distribute raw tensor memory into a DTensor parameter
        #
        # Why torch.no_grad() is required here:
        # - Prevents PyTorch from recording `distribute_tensor` ops in the global
        #   autograd graph during model initialization.
        # - Ensures `sharded_dtensor` is created as a clean leaf node (grad_fn=None).
        # ---------------------------------------------------------------------
        with torch.no_grad():
            sharded_dtensor = distribute_tensor(
                parameter.data,
                device_mesh=self.device_mesh,
                placements=[Shard(dim=0)],
            )
            self.dtensor: DTensor = cast(
                "DTensor",
                nn.Parameter(
                    sharded_dtensor,
                    requires_grad=parameter.requires_grad,
                ),
            )

        # 2. Mount the DTensor Parameter directly on the target module attribute
        setattr(self.module, self.parameter_name, self.dtensor)

        # ---------------------------------------------------------------------
        # 3. Initialize persistent shape shell for the global unsharded tensor.
        # During the forward pass, PyTorch's autograd engine captures references
        # to forward tensors in saved-tensor metadata for backward.
        # If we created new tensor instances for full size all-gathered data
        # dynamically, autograd would hold strong references to their allocations,
        # preventing memory release.
        # By using a persistent buffer whose untyped storage is resized (`resize_(0)`)
        # upon sharding, we ensure physical memory is instantly freed even while
        # autograd retains the tensor shell.
        # ---------------------------------------------------------------------
        self._unsharded_buffer = torch.empty(
            self.dtensor.shape,
            dtype=self.dtensor.dtype,
            device=self.local_shard.device,
        )
        self.release_unsharded_buffer()

    @property
    def fqn(self) -> str:
        """The Fully Qualified Name of the parameter."""
        return self._fqn

    @fqn.setter
    def fqn(self, value: str) -> None:
        """Sets the FQN and automatically updates the parameter basename."""
        self._fqn = value
        self.parameter_name = value.split(".")[-1]

    @property
    def local_shard(self) -> Tensor:
        """Raw rank-local storage slice (`Tensor`) backing the DTensor."""
        return self.dtensor.to_local()

    @property
    def shape(self) -> torch.Size:
        """
        Active tensor shape depending on lifecycle state.

        - GATHERED: Global shape (e.g., [24, 18])
        - SHARDED: Local shard shape (e.g., [12, 18] on 2 ranks)
        """
        if self.is_gathered:
            return self._unsharded_buffer.shape
        return self.local_shard.shape

    @property
    def is_gathered(self) -> bool:
        """`True` while the full buffer owns memory (storage size is the source of truth)."""
        return self._unsharded_buffer.untyped_storage().size() > 0

    @property
    def unsharded_buffer(self) -> Tensor:
        """
        The global unsharded tensor instance.

        Acts as a lazy accessor: if the underlying untyped storage is currently released
        (size 0), accessing this property automatically triggers storage allocation to
        fit the full global shape.
        """
        storage = self._unsharded_buffer.untyped_storage()
        if storage.size() == 0:
            storage.resize_(
                self._unsharded_buffer.numel() * self._unsharded_buffer.element_size()
            )
        return self._unsharded_buffer

    def release_unsharded_buffer(self) -> None:
        """Truncates untyped storage to 0 bytes to release memory back to the allocator."""
        self._unsharded_buffer.untyped_storage().resize_(0)

    def set_unsharded(self, full_tensor: Tensor) -> None:
        """
        Attaches the full gathered tensor to the parent module attribute.

        Bypasses nn.Module.__setattr__ type checks (which reject plain Tensors
        on nn.Parameter attributes) by writing directly to module.__dict__,
        allowing C++ operations like F.linear to consume the contiguous tensor.

        Args:
            full_tensor: Unsharded full-size tensor gathered across ranks.
        """
        self.module.__dict__[self.parameter_name] = full_tensor

    def set_sharded(self) -> None:
        """
        Restores the parent module attribute back to the rank-local DTensor `nn.Parameter`.

        This release the reference to the full unsharded tensor memory,
        and resize down the underlying storage to zero bytes to release memory.
        """
        self.module.__dict__.pop(self.parameter_name, None)
        self.release_unsharded_buffer()
