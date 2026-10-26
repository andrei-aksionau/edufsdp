"""
State dictionary management for Educational FSDP.

Handles extracting and loading model state dictionaries in two formats:
1. `Local`: Saves/loads rank-local 1/N parameter tensor shards.
2. `Full`: Saves/loads consolidated global parameters (unsharded).
"""

from enum import Enum
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from edufsdp.hooks import FSDPAllGatherFunction

if TYPE_CHECKING:
    from collections.abc import Mapping

    from edufsdp.sharded_parameter import ShardedParameter


class StateDictType(Enum):
    """Supported state dict format types."""

    Full = "full"  # Consolidated global parameters across ranks
    Local = "local"  # Rank-local sharded parameter tensors


class StateDictManager:
    """Manages state dict extraction and restoration for sharded models."""

    @staticmethod
    def _collect_sharded_parameters(module: nn.Module) -> dict[str, ShardedParameter]:
        """
        Recursively collect sharded parameters across the module tree.

        Args:
            module: Target root `nn.Module` to inspect.

        Returns:
            Dictionary mapping parameter FQNs to their `ShardedParameter` wrappers.

        """
        sharded_parameters_dict: dict[str, ShardedParameter] = {}

        def _walk(submodule: nn.Module, prefix: str = "") -> None:
            if hasattr(submodule, "_sharded_parameters"):
                submodule_parameters = cast(
                    "dict[str, ShardedParameter]",
                    submodule._sharded_parameters,
                )
                for submodule_name, sharded_parameter in submodule_parameters.items():
                    fqn = f"{prefix}{submodule_name}" if prefix else submodule_name
                    sharded_parameters_dict[fqn] = sharded_parameter

            for child_name, child_module in submodule.named_children():
                child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
                _walk(child_module, prefix=child_prefix)

        _walk(module)
        return sharded_parameters_dict

    @staticmethod
    @torch.no_grad()
    def get_state_dict(
        module: nn.Module,
        state_dict_type: StateDictType = StateDictType.Local,
    ) -> dict[str, Tensor]:
        """
        Extracts module state dictionary in either `Local` or `Full` format.

        Args:
            module: Target root `nn.Module` containing sharded submodules.
            state_dict_type: Target format (`StateDictType.Local` or `StateDictType.Full`).

        Returns:
            Map of parameter FQNs to local shards or full tensors.

        """
        state_dict: dict[str, Tensor] = {}
        sharded_parameters_dict = StateDictManager._collect_sharded_parameters(module)

        for parameter_name, parameter in module.named_parameters():
            if parameter_name not in sharded_parameters_dict:
                tensor = parameter.data
            else:
                sharded_parameter = sharded_parameters_dict[parameter_name]

                if state_dict_type == StateDictType.Local:
                    tensor = sharded_parameter.dtensor
                else:
                    gathered = sharded_parameter._gathered_tensor
                    tensor = (
                        gathered
                        if gathered is not None
                        else FSDPAllGatherFunction.apply(
                            sharded_parameter.local_shard, sharded_parameter
                        )
                    )

            # Store detached copy to prevent state contamination
            state_dict[parameter_name] = tensor.detach().clone()

        return state_dict

    @staticmethod
    @torch.no_grad()
    def load_state_dict(
        module: nn.Module,
        state_dict: Mapping[str, Tensor | DTensor],
        state_dict_type: StateDictType = StateDictType.Local,
    ) -> None:
        """
        Loads a state dictionary checkpoint into sharded model parameters.

        Args:
            module: Target root `nn.Module` containing sharded submodules.
            state_dict: Incoming map of parameter FQNs to tensors.
            state_dict_type: Format of incoming state_dict (`Local` or `Full`).

        Raises:
            KeyError: If a parameter in `state_dict` does not exist in `module`.

        """
        module_named_parameters = dict(module.named_parameters())
        sharded_parameters_dict = StateDictManager._collect_sharded_parameters(module)

        for sd_name, sd_tensor in state_dict.items():
            if sd_name not in module_named_parameters:
                raise KeyError(
                    f"Parameter '{sd_name}' from state_dict cannot be found in the module."
                )

            # Handle non-sharded parameters
            if sd_name not in sharded_parameters_dict:
                target_parameter = module_named_parameters[sd_name]
                target_parameter.copy_(sd_tensor.to(target_parameter.device))
                continue

            parameter = sharded_parameters_dict[sd_name]

            if state_dict_type == StateDictType.Local:
                # Handle incoming state dict tensor whether it is a DTensor or raw local Tensor
                local_sd_tensor = (
                    sd_tensor.to_local() if isinstance(sd_tensor, DTensor) else sd_tensor
                )
                parameter.local_shard.copy_(local_sd_tensor.to(parameter.local_shard.device))

            elif state_dict_type == StateDictType.Full:
                # Distribute global tensor across mesh to extract current rank's shard
                sharded_dtensor = distribute_tensor(
                    sd_tensor.to(parameter.local_shard.device),
                    device_mesh=parameter.device_mesh,
                    placements=[Shard(dim=0)],
                )

                # Copy shard data into the existing rank-local parameter buffer
                parameter.local_shard.copy_(sharded_dtensor.to_local())
