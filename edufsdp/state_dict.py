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
from torch.distributed.tensor import DTensor, distribute_tensor

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
            Dictionary mapping absolute parameter FQNs to their `ShardedParameter` wrappers.
        """
        sharded_parameters_dict: dict[str, ShardedParameter] = {}

        for _, submodule in module.named_modules():
            if hasattr(submodule, "_sharded_parameters"):
                sub_params = cast(
                    "dict[str, ShardedParameter]",
                    submodule._sharded_parameters,
                )
                sharded_parameters_dict.update(sub_params)

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
                    if sharded_parameter.is_gathered:
                        # Mid-forward/backward: the full weights are already in the buffer.
                        tensor = sharded_parameter.unsharded_buffer.detach().clone()
                    else:
                        # Borrow the buffer: gather, copy out, give the memory back.
                        gathered = FSDPAllGatherFunction.apply(
                            sharded_parameter.local_shard, sharded_parameter
                        )
                        tensor = gathered.detach().clone()
                        sharded_parameter.release_unsharded_buffer()

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
                local_sd_tensor = (
                    sd_tensor.to_local() if isinstance(sd_tensor, DTensor) else sd_tensor
                )
                parameter.local_shard.copy_(local_sd_tensor.to(parameter.local_shard.device))

            elif state_dict_type == StateDictType.Full:
                full_tensor = sd_tensor.to(parameter.dtensor.device)
                assert full_tensor.shape == parameter.dtensor.shape

                sharded_incoming = distribute_tensor(
                    full_tensor,
                    device_mesh=parameter.device_mesh,
                    placements=parameter.dtensor.placements,
                )
                parameter.dtensor.copy_(sharded_incoming)
