"""
Core FSDP wrapping and unit orchestration for Educational FSDP.

Provides the primary user-facing entrypoint (`fully_shard`) to wrap PyTorch modules,
collecting subtree parameters, registering collective module-level pre-hooks, and
attaching tensor-level hooks for immediate backward gradient-time reclamation.

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

from edufsdp.hooks import BackwardPostHook, BackwardPreHook, ForwardPostHook, ForwardPreHook
from edufsdp.sharded_parameter import ShardedParameter

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.distributed.device_mesh import DeviceMesh


def fully_shard(module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
    """
    Shards a PyTorch module and its entire un-sharded subtree as a single FSDP unit.

    Recursively collects all parameters within the module's subtree into a single dictionary,
    registers forward pre/post-hooks and backward pre-hooks on the target module, and binds
    tensor-level backward post-hooks directly to parameter DTensors to ensure precise
    gradient-time reclamation and prevent premature memory release.

    Args:
        module: Target PyTorch module (e.g., a TransformerBlock or the whole model) to shard.
        device_mesh: PyTorch DeviceMesh defining process rank layout for sharding.

    Returns:
        The sharded module instance.
    """
    # Prevent repeated wrapping and FQN modification.
    if hasattr(module, "_sharded_parameters"):
        return module

    sharded_parameters: dict[str, ShardedParameter] = {}

    def _collect_subtree(submodule: nn.Module, prefix: str = "") -> None:
        """Collects parameters not already managed by nested FSDP units."""
        # Independently wrapped children manage their own parameters.
        if submodule is not module and hasattr(submodule, "_sharded_parameters"):
            return

        # Collect parameters owned directly by this submodule.
        for param_name, parameter in submodule.named_parameters(recurse=False):
            fqn = f"{prefix}.{param_name}" if prefix else param_name

            sharded_parameter = ShardedParameter(
                fqn=fqn,
                module=submodule,
                parameter=parameter,
                device_mesh=device_mesh,
            )
            sharded_parameters[fqn] = sharded_parameter

            # Register tensor-level backward hook for gradient-time reclamation.
            if sharded_parameter.dtensor.requires_grad:
                sharded_parameter.dtensor.register_hook(BackwardPostHook(sharded_parameter))

        # Recurse into child modules.
        for child_name, child_module in submodule.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            _collect_subtree(child_module, prefix=child_prefix)

    _collect_subtree(module)

    # Mark the module as wrapped, even when it manages no parameters itself.
    object.__setattr__(module, "_sharded_parameters", sharded_parameters)

    # Register hooks only when this FSDP unit manages parameters.
    if sharded_parameters:
        module.register_forward_pre_hook(ForwardPreHook(sharded_parameters))
        module.register_forward_hook(ForwardPostHook(sharded_parameters))
        module.register_full_backward_pre_hook(BackwardPreHook(sharded_parameters))

    # Build a mapping of module objects to their names relative to this FSDP root.
    module_names = {id(submodule): name for name, submodule in module.named_modules()}

    # Normalize FQNs in every FSDP unit within this subtree.
    for _, submodule in module.named_modules():
        if not hasattr(submodule, "_sharded_parameters"):
            continue

        existing_params = submodule._sharded_parameters
        updated_params: dict[str, ShardedParameter] = {}

        for sharded_parameter in existing_params.values():
            owner_name = module_names[id(sharded_parameter.module)]
            parameter_name = sharded_parameter.parameter_name

            absolute_fqn = f"{owner_name}.{parameter_name}" if owner_name else parameter_name

            sharded_parameter.fqn = absolute_fqn
            updated_params[absolute_fqn] = sharded_parameter

        object.__setattr__(submodule, "_sharded_parameters", updated_params)

    return module
