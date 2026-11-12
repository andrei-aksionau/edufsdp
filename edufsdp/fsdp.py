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
    Shards a PyTorch module and its unsharded subtree as a single FSDP unit.

    Recursively collects all parameters within the module's subtree, registers
    lifecycle hooks, and normalizes fully qualified names across the hierarchy.

    Args:
        module: Target PyTorch module to shard.
        device_mesh: PyTorch DeviceMesh defining process rank layout for sharding.

    Returns:
        The sharded module instance.
    """
    # Prevent repeated wrapping and FQN modification if the module is already sharded.
    if hasattr(module, "_sharded_parameters"):
        return module

    # Step 1: Collect all parameters within this module's un-wrapped subtree.
    sharded_parameters = _collect_subtree_parameters(module, device_mesh)

    # Attach the collected sharded parameter map to the root module object.
    object.__setattr__(module, "_sharded_parameters", sharded_parameters)

    # Step 2: Register FSDP communication and memory management hooks if parameters exist.
    if sharded_parameters:
        _register_fsdp_hooks(module, sharded_parameters)

    # Step 3: Normalize absolute fully qualified names (FQNs) across all nested FSDP units.
    _normalize_subtree_fqns(module)

    return module


def _collect_subtree_parameters(
    root_module: nn.Module, device_mesh: DeviceMesh
) -> dict[str, ShardedParameter]:
    """
    Recursively traverses the module subtree to discover and wrap parameters.

    Stops traversal when encountering independently wrapped nested FSDP units.
    Wraps each raw PyTorch parameter into a managed ShardedParameter instance.

    Args:
        root_module: The root module of the current FSDP wrapping boundary.
        device_mesh: PyTorch DeviceMesh specifying process group layout.

    Returns:
        A dictionary mapping absolute fully qualified names to ShardedParameter instances.
    """
    sharded_parameters: dict[str, ShardedParameter] = {}

    def _recurse(submodule: nn.Module, prefix: str = "") -> None:
        if submodule is not root_module and hasattr(submodule, "_sharded_parameters"):
            return

        for param_name, parameter in submodule.named_parameters(recurse=False):
            fqn = f"{prefix}.{param_name}" if prefix else param_name

            sharded_parameter = ShardedParameter(
                fqn=fqn,
                module=submodule,
                parameter=parameter,
                device_mesh=device_mesh,
            )
            sharded_parameters[fqn] = sharded_parameter

        for child_name, child_module in submodule.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            _recurse(child_module, prefix=child_prefix)

    _recurse(root_module)
    return sharded_parameters


def _register_fsdp_hooks(
    module: nn.Module, sharded_parameters: dict[str, ShardedParameter]
) -> None:
    """
    Attaches module-level hooks and parameter-level tensor hooks for FSDP coordination.

    Registers module forward pre/post-hooks and backward pre-hooks to manage global
    all-gather and clean-up phases. Additionally, iterates through the provided
    sharded parameters to bind tensor-level `BackwardPostHook` directly to each
    individual parameter DTensor. This ensures that memory reclamation happens
    immediately upon gradient completion for each specific parameter, avoiding race
    conditions during block-level wrapping.

    Args:
        module: The target PyTorch module receiving the hooks.
        sharded_parameters: Dictionary of sharded parameters managed by this FSDP unit.
    """
    module.register_forward_pre_hook(ForwardPreHook(sharded_parameters))
    module.register_forward_hook(ForwardPostHook(sharded_parameters))
    module.register_full_backward_pre_hook(BackwardPreHook(sharded_parameters))

    for sharded_parameter in sharded_parameters.values():
        if sharded_parameter.dtensor.requires_grad:
            sharded_parameter.dtensor.register_hook(BackwardPostHook(sharded_parameter))


def _normalize_subtree_fqns(root_module: nn.Module) -> None:
    """
    Normalizes parameter fully qualified names (FQNs) relative to the root module hierarchy.

    Inspected module names are mapped and assigned to each sharded parameter to ensure
    consistent naming across distributed ranks and checkpointing boundaries.

    Args:
        root_module: The root module whose subtree FQNs require normalization.
    """
    module_names = {id(submodule): name for name, submodule in root_module.named_modules()}

    for _, submodule in root_module.named_modules():
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
