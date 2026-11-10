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
      3. Provides memory-tracking properties (local shard sizes, global ranges).
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

        # TODO: add some description
        # TODO: does it need to be created with full shape and then freed, or we can create of
        # size of 0?
        self._full = torch.empty(
            self.dtensor.shape,
            dtype=self.dtensor.dtype,
            device=self.local_shard.device,
        )
        self.free_full()

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

    # TODO: do we even need this?
    @property
    def full_buffer(self) -> Tensor:
        """Persistent full-size buffer. Shape is permanent; memory exists only while gathered."""
        storage = self._full.untyped_storage()
        if storage.size() == 0:
            storage.resize_(self._full.numel() * self._full.element_size())
        return self._full

    @property
    def is_gathered(self) -> bool:
        """`True` while the full buffer owns memory (storage size is the source of truth)."""
        return self._full.untyped_storage().size() > 0

    # TODO: that needs to be updated
    @property
    def shape(self) -> torch.Size:
        """
        Active tensor shape depending on lifecycle state.

        - GATHERED: Global shape (e.g., [24, 18])
        - SHARDED: Local shard shape (e.g., [12, 18] on 2 ranks)
        """
        if self.is_gathered:
            return self._full.shape
        return self.local_shard.shape

    # TODO: that doesn't look pretty
    # def allocate_full(self) -> Tensor:
    #     """Re-allocates the full buffer's memory if it was freed and returns the buffer."""
    #     # TODO: feels odd to have a separate var 'storage', but to return 'full'
    #     storage = self._full.untyped_storage()
    #     if storage.size() == 0:
    #         storage.resize_(self._full.numel() * self._full.element_size())
    #     return self._full

    def free_full(self) -> None:
        """Releases the buffer's memory. Every alias, including saved tensors, is emptied."""
        self._full.untyped_storage().resize_(0)

    def set_unsharded(self, full_tensor: Tensor) -> None:
        """
        Attaches the full gathered tensor to the parent module attribute.

        Standard `setattr(module, parameter_name, full_tensor)` invokes PyTorch's
        `nn.Module.__setattr__`, which rejects assigning a plain `Tensor`
        to an existing `nn.Parameter` attribute.

        By assigning directly into `module.__dict__`, we bypass parameter type checks
        so native C++ PyTorch functions (e.g., `F.linear`) can directly consume standard
        contiguous tensors during forward and backward computations.

        Args:
            full_tensor: Unsharded full-size tensor gathered across ranks.

        """
        self.module.__dict__[self.parameter_name] = full_tensor

    def set_sharded(self) -> None:
        """
        Restores the parent module attribute back to the rank-local DTensor `nn.Parameter`.

        This releases references to the full unsharded tensor memory, allowing
        PyTorch's CUDA memory manager / Python garbage collector to reclaim the memory.
        """
        self.module.__dict__.pop(self.parameter_name, None)
        self.free_full()
