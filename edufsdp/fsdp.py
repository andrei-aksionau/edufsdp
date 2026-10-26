"""
Educational FSDP Wrapper module.

Transforms a target `nn.Module` by recursively sharding parameters across ranks
and attaching pre/post execution hooks for automatic parameter gathering and freeing.

Execution lifecycle for each submodule:

              Forward Pass                                   Backward Pass
 ┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
 │ ForwardPreHook                       │        │ BackwardPreHook                      │
 │   --> all_gather()                   │        │   --> all_gather()                   │
 │   Gathers weight & bias to full size │        │   Gathers weight & bias to full size │
 │   [Full memory]                      │        │   [Full memory]                      │
 └──────────────────┬───────────────────┘        └──────────────────┬───────────────────┘
                    │                                               │
                    ▼                                               ▼
 ┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
 │ Forward Compute                      │        │ Backward Compute                     │
 │   Executes layer ops (e.g. F.linear) │        │   Calculates weight, bias & input    │
 │   using unsharded full parameters    │        │   gradients using full parameters    │
 └──────────────────┬───────────────────┘        └──────────────────┬───────────────────┘
                    │                                               │
                    ▼                                               ▼
 ┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
 │ ForwardPostHook                      │        │ reduce_scatter()                     │
 │   --> free()                         │        │   Averages & shards weight & bias    │
 │   Drops full weight & bias tensors,  │        │   gradients across ranks into local  │
 │   restoring DTensors [1/N memory]    │        │   1/N DTensor shard buffers          │
 └──────────────────────────────────────┘        └──────────────────┬───────────────────┘
                                                                    │
                                                                    ▼
                                                 ┌──────────────────────────────────────┐
                                                 │ BackwardPostHook                     │
                                                 │   --> free()                         │
                                                 │   Drops full weight & bias tensors,  │
                                                 │   restoring DTensors [1/N memory]    │
                                                 └──────────────────────────────────────┘
"""

from typing import TYPE_CHECKING

import torch.nn as nn

from edufsdp.hooks import BackwardPostHook, BackwardPreHook, ForwardPostHook, ForwardPreHook
from edufsdp.sharded_parameter import ShardedParameter

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


def fully_shard[ModuleT: nn.Module](module: ModuleT, device_mesh: DeviceMesh) -> ModuleT:
    """
    Shards a PyTorch `nn.Module` in-place using Fully Sharded Data Parallelism (FSDP v2 style).

    Recursively traverses the module tree, wraps all child parameters into `ShardedParameter`
    instances, and registers module-level execution hooks to orchestrate collective operations
    (`all_gather`, `reduce_scatter`) across a process group `DeviceMesh` in batches.

    Args:
        module: Target PyTorch module to shard and wrap in-place.
        device_mesh: PyTorch DeviceMesh defining process rank layout.

    Returns:
        The sharded module instance (passed in).

    """
    sharded_parameters: dict[str, ShardedParameter] = {}

    def _recursive_wrap(submodule: nn.Module, prefix: str = "") -> None:
        """
        Traverses the module subtree recursively to wrap parameters.

        For each submodule:
            1. Recursively traverses child modules (bottom-up traversal).
            2. Shards direct parameters into `ShardedParameter` wrappers.

        Args:
            submodule: Current module instance in the traversal.
            prefix: Accumulated dot-separated FQN path (e.g., 'layer1.').

        """
        # 1. Post-order recursion: process child submodules first
        for child_name, child_module in submodule.named_children():
            child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
            _recursive_wrap(child_module, prefix=child_prefix)

        # 2. Process parameters owned directly by this submodule instance
        for parameter_name, parameter in list(submodule.named_parameters(recurse=False)):
            fqn = f"{prefix}{parameter_name}" if prefix else parameter_name

            sharded_parameter = ShardedParameter(
                fqn=fqn,
                module=submodule,
                parameter=parameter,
                device_mesh=device_mesh,
            )
            sharded_parameters[fqn] = sharded_parameter

    # Execute recursive tree wrapping
    _recursive_wrap(module)

    # 3. Attach module-level lifecycle hooks to trigger batch all_gather and free
    if sharded_parameters:
        module.register_forward_pre_hook(ForwardPreHook(sharded_parameters))
        module.register_forward_hook(ForwardPostHook(sharded_parameters))
        module.register_full_backward_pre_hook(BackwardPreHook(sharded_parameters))
        module.register_full_backward_hook(BackwardPostHook(sharded_parameters))

        # Store reference to sharded parameters on the module for inspection and checkpointing
        object.__setattr__(module, "_sharded_parameters", sharded_parameters)

    return module
