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

from typing import TYPE_CHECKING, cast

from edufsdp.hooks import BackwardPostHook, BackwardPreHook, ForwardPostHook, ForwardPreHook
from edufsdp.sharded_parameter import ShardedParameter

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.distributed.device_mesh import DeviceMesh


def fully_shard(module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
    """
    Shards a PyTorch module and its entire un-sharded subtree as a single FSDP unit.

    Recursively collects all parameters within the module's subtree into a single dictionary,
    registers execution hooks exclusively on the target module, and respects pre-sharded
    child boundaries.

    Args:
        module: Target PyTorch module (e.g., a TransformerBlock or the whole model) to shard.
        device_mesh: PyTorch DeviceMesh defining process rank layout for sharding.

    Returns:
        The sharded module instance (passed in).

    """

    if hasattr(module, "_sharded_parameters"):
        return module

    sharded_parameters: dict[str, ShardedParameter] = {}

    def _collect_subtree(submodule: nn.Module, prefix: str = "") -> None:
        """Recursively collects parameters across the subtree into the root dictionary."""
        # If this child is already an independent sharded unit, # do not absorb its parameters.
        # Instead, patch its FQNs with the new prefix.
        if submodule is not module and hasattr(submodule, "_sharded_parameters"):
            existing_params = cast(
                "dict[str, ShardedParameter]",
                submodule._sharded_parameters,
            )
            updated_params: dict[str, ShardedParameter] = {}
            for old_key, sharded_param in existing_params.items():
                param_basename = old_key.split(".")[-1]
                absolute_fqn = f"{prefix}.{param_basename}" if prefix else param_basename
                sharded_param.fqn = absolute_fqn  # Property setter updates name & fqn
                updated_params[absolute_fqn] = sharded_param
            object.__setattr__(submodule, "_sharded_parameters", updated_params)
            return

        # Collect parameters owned directly by this submodule
        for param_name, parameter in submodule.named_parameters(recurse=False):
            fqn = f"{prefix}.{param_name}" if prefix else param_name
            sharded_parameter = ShardedParameter(
                fqn=fqn,
                module=submodule,
                parameter=parameter,
                device_mesh=device_mesh,
            )
            sharded_parameters[fqn] = sharded_parameter

        # Recurse into child submodules
        for child_name, child_module in submodule.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            _collect_subtree(child_module, prefix=child_prefix)

    # Execute subtree parameter collection starting from the target module
    _collect_subtree(module)

    # Register hooks ONCE at the target module level for the entire collected block
    if sharded_parameters:
        module.register_forward_pre_hook(ForwardPreHook(sharded_parameters))
        module.register_forward_hook(ForwardPostHook(sharded_parameters))
        module.register_full_backward_pre_hook(BackwardPreHook(sharded_parameters))
        module.register_full_backward_hook(BackwardPostHook(sharded_parameters))

        # Store reference to sharded parameters on the module for inspection and checkpointing
        object.__setattr__(module, "_sharded_parameters", sharded_parameters)

    return module
