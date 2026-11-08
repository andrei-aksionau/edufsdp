"""
Autograd and module execution hooks for Educational FSDP.

Provides:
1. `FSDPAllGatherFunction`: Autograd function handling All-Gather and Reduce-Scatter.
2. `ForwardPreHook`: Managing parameter reconstruction before calculation.
3. `ForwardPostHook`: Clearing full weights back to local shards after forward pass.
4. `BackwardPreHook`: Re-gathering parameters for backward pass.
5. `BackwardPostHook`: Freeing parameters after backward pass.


=============================================================================================
                                   HOOK EXECUTION SEQUENCE
=============================================================================================

 FORWARD PASS:
 ┌───────────────────────────────┐         ┌───────────────────────────┐
 │ ForwardPreHook                │  ---->  │ FSDPAllGatherFunction     │ ──────┐
 │  • Intercepts forward         │         │  • Executes All-Gather    │       │
 │  • Triggers gathering         │         │    across all ranks       │       │
 │    via FSDPAllGatherFunction  │         │                           │       │
 └───────────────────────────────┘         └───────────────────────────┘       │
                                                                               │
                                                                               │
 ┌───────────────────────────────┐         ┌───────────────────────────┐       │
 │ ForwardPostHook               │  <----  │ Module Forward Compute    │ <─────┘
 │  • Frees gathered             │         │  • Executes layer ops     │
 │    weights                    │         │    using full parameters  │
 └───────────────────────────────┘         └───────────────────────────┘

 -------------------------------------------------------------------------------

 BACKWARD PASS:
 ┌───────────────────────────────┐         ┌───────────────────────────┐
 │ BackwardPreHook               │  ---->  │ FSDPAllGatherFunction     │ ──────┐
 │  • Intercepts backward        │         │  • Re-executes All-Gather │       │
 │  • Triggers gathering         │         │    for gradient calc      │       │
 │    via FSDPAllGatherFunction  │         │                           │       │
 └───────────────────────────────┘         └───────────────────────────┘       │
                                                                               │
                                                                               │
 ┌───────────────────────────────┐         ┌───────────────────────────┐       │
 │ FSDPAllGatherFunction         │  <----  │ Module Backward Compute   │ <─────┘
 │  • Autograd triggers          │         │  • Computes gradients     │
 │    .backward()                │         │    w.r.t. parameters      │
 │  • Executes Reduce-Scatter    │         │                           │
 └─────────────┬─────────────────┘         └───────────────────────────┘
               |
               v
 ┌───────────────────────────────┐
 │ BackwardPostHook              │
 │  • Frees gathered             │
 │    weights                    │
 └───────────────────────────────┘

"""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from edufsdp.logging import logger

if TYPE_CHECKING:
    from edufsdp.sharded_parameter import ShardedParameter


class FSDPAllGatherFunction(torch.autograd.Function):
    """
    Symmetric autograd function.

    Pairs forward parameter All-Gather with backward gradient Reduce-Scatter.
    """

    @staticmethod
    def forward(
        ctx: Any, local_parameter: Tensor, sharded_parameter: ShardedParameter
    ) -> Tensor:
        """
        Gathers rank-local parameter shards into the full global parameter.

        By receiving ``local_parameter`` explicitly as a tensor argument alongside the
        ``sharded_parameter`` metadata container, PyTorch autograd can directly track
        the leaf tensor in the execution graph, ensuring backward gradients are
        correctly routed back to ``local_parameter.grad``.

        Args:
            ctx: Autograd context used to store metadata across forward and backward passes.
            local_parameter: Rank-local parameter slice with autograd graph history.
            sharded_parameter: Parameter with distributed topology and device mesh.

        Returns:
            The reconstructed global parameter tensor assembled via All-Gather.

        """
        ctx.parameter = sharded_parameter

        # 1. Wrap rank-local parameter shard as a DTensor sharded along dimension 0
        sharded_dtensor = DTensor.from_local(
            local_parameter,
            device_mesh=sharded_parameter.device_mesh,
            placements=[Shard(dim=0)],
        )

        # 2. Execute All-Gather collective across process group (Shard(0) -> Replicate())
        replicated_dtensor = sharded_dtensor.redistribute(
            device_mesh=sharded_parameter.device_mesh,
            placements=[Replicate()],
        ).to_local()

        logger.log_transition(
            event_name="all_gather",
            parameter_name=sharded_parameter.fqn,
            shape_before=sharded_parameter.local_shard.shape,
            shape_after=replicated_dtensor.shape,
        )

        return replicated_dtensor

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Tensor, None]:
        """
        Reduces incoming full global gradients into rank-local gradient shards.

        Applies a Reduce-Scatter collective across the process group to aggregate and
        shard accumulated gradients.

        Args:
            ctx: Autograd context containing stored distributed metadata from forward pass.
            *grad_outputs: Tuple containing the incoming full-shape un-reduced gradient tensor
                calculated during backward pass.

        Returns:
            Rank-local sharded gradient tensor scaled by world size,
            and None matching non-tensor metadata argument from forward pass.

        """
        (grad_output,) = grad_outputs
        parameter: ShardedParameter = ctx.parameter

        # 1. Wrap un-reduced full gradient as Partial() across the process group. Because
        # every rank calculated gradients over the full parameter shape, the global gradient
        # across ranks is mathematically an un-reduced additive sum across the process group.
        # Wrapping grad_output as DTensor with `placements=[Partial()]` explicitly informs
        # PyTorch DTensor that this local tensor represents an un-reduced partial sum across
        # ranks.
        grad_dtensor = DTensor.from_local(
            grad_output,
            device_mesh=parameter.device_mesh,
            placements=[Partial()],
        )

        # 2. Execute Reduce-Scatter collective across mesh (Partial() -> Shard(0)) Calling
        # `.redistribute(placements=[Shard(dim=0)])` on a Partial() DTensor cleanly triggers
        # a Reduce-Scatter collective, summing the partial gradients across ranks while
        # simultaneously sharding the reduced result along dimension 0.
        sharded_grad_dtensor = grad_dtensor.redistribute(
            device_mesh=parameter.device_mesh,
            placements=[Shard(dim=0)],
        )

        sharded_grad = sharded_grad_dtensor.to_local()

        # 3. Average reduced additive gradient sum by process group world size
        world_size = parameter.device_mesh.size()
        sharded_grad = sharded_grad / world_size

        logger.log_transition(
            event_name="reduce_scatter",
            parameter_name=f"{parameter.fqn}.grad",
            shape_before=grad_output.shape,
            shape_after=sharded_grad.shape,
        )

        # Matches forward inputs signature (local_parameter, sharded_parameter)
        return sharded_grad, None


class ForwardPreHook:
    """
    Module pre-forward hook that gathers parameter shards before computation starts.

    Triggers immediately prior to `nn.Module.forward()`. Reconstructs full-sized
    weight and bias tensors from local 1/N shards via `FSDPAllGatherFunction`, updating
    the wrapped `ShardedParameter` state to `is_gathered=True`.
    """

    def __init__(self, sharded_parameters: dict[str, ShardedParameter]) -> None:
        """
        Initializes the hook with its target sharded parameters.

        Args:
            sharded_parameters: Dictionary of target sharded parameters managed by this hook.

        """
        self.sharded_parameters = sharded_parameters

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        """Executes parameter all-gather before forward pass execution."""
        for sharded_parameter in self.sharded_parameters.values():
            if sharded_parameter.is_gathered:
                continue

            # Run All-Gather with or without autograd graph tracking
            if sharded_parameter.dtensor.requires_grad and torch.is_grad_enabled():
                full_tensor = FSDPAllGatherFunction.apply(
                    sharded_parameter.local_shard, sharded_parameter
                )
            else:
                with torch.no_grad():
                    full_tensor = FSDPAllGatherFunction.apply(
                        sharded_parameter.local_shard, sharded_parameter
                    )

            sharded_parameter.set_unsharded(full_tensor)


class ForwardPostHook:
    """
    Module post-forward hook that releases full parameters immediately after computation.

    Triggers immediately after `nn.Module.forward()` returns. Releases allocated full
    parameter tensors, reverting `ShardedParameter` back to 1/N local storage to minimize
    peak memory consumption during the remainder of the forward pass.
    """

    def __init__(self, sharded_parameters: dict[str, ShardedParameter]) -> None:
        """
        Initializes the hook with its target sharded parameters.

        Args:
            sharded_parameters: Dictionary of target sharded parameters managed by this hook.

        """
        self.sharded_parameters = sharded_parameters

    def __call__(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        """Frees gathered full parameters after forward pass execution."""
        for sharded_parameter in self.sharded_parameters.values():
            if not sharded_parameter.is_gathered:
                continue

            before_shape = sharded_parameter.shape
            sharded_parameter.set_sharded()
            after_shape = sharded_parameter.shape

            logger.log_transition(
                event_name="free",
                parameter_name=sharded_parameter.fqn,
                shape_before=before_shape,
                shape_after=after_shape,
            )


class BackwardPreHook:
    """
    Module backward pre-hook that re-gathers full weights for gradient calculations.

    Triggers during `torch.autograd.backward()` right before backpropagation traverses
    through this specific layer. Re-executes All-Gather to supply full parameters
    required for gradient matrix multiplications.
    """

    def __init__(self, sharded_parameters: dict[str, ShardedParameter]) -> None:
        """
        Initializes the hook with its target sharded parameters.

        Args:
            sharded_parameters: Dictionary of target sharded parameters managed by this hook.

        """
        self.sharded_parameters = sharded_parameters

    def __call__(self, module: nn.Module, grad_output: Any) -> None:
        """Executes parameter all-gather before layer backward pass execution."""
        for sharded_parameter in self.sharded_parameters.values():
            if sharded_parameter.is_gathered:
                continue

            # Prevent autograd from recording a duplicate AllGather graph node
            # during backward execution, which would trigger graph mutation errors.
            with torch.no_grad():
                full_tensor = FSDPAllGatherFunction.apply(
                    sharded_parameter.local_shard, sharded_parameter
                )

            sharded_parameter.set_unsharded(full_tensor)


class BackwardPostHook:
    """
    Module backward post-hook that frees gathered parameters after gradient computations.

    Triggers after layer gradient calculations complete. Releases full parameter memory,
    leaving only sharded 1/N local parameters and 1/N local gradient shards in device memory.
    """

    def __init__(self, sharded_parameters: dict[str, ShardedParameter]) -> None:
        """
        Initializes the hook with its target sharded parameters.

        Args:
            sharded_parameters: Dictionary of target sharded parameters managed by this hook.

        """
        self.sharded_parameters = sharded_parameters

    def __call__(
        self,
        module: nn.Module,
        grad_input: Tensor | tuple[Tensor, ...],
        grad_output: Tensor | tuple[Tensor, ...],
    ) -> None:
        """Frees gathered full parameters after layer backward pass execution."""
        for sharded_parameter in self.sharded_parameters.values():
            if not sharded_parameter.is_gathered:
                continue

            before_shape = sharded_parameter.shape
            sharded_parameter.set_sharded()
            after_shape = sharded_parameter.shape

            logger.log_transition(
                event_name="free",
                parameter_name=sharded_parameter.fqn,
                shape_before=before_shape,
                shape_after=after_shape,
            )
