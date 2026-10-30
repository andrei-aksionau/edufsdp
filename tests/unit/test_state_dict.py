"""
Unit tests for edufsdp.state_dict covering Local and Full state dict extraction,
loading, and parameter state preservation.
"""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.tensor import DTensor

from edufsdp.fsdp import fully_shard
from edufsdp.state_dict import StateDictManager, StateDictType
from tests.base.test_class import DistributedTestBase

# TODO: update tests to include obtaining collection of sharded parameters


def _fully_shard_model(model: nn.Module, device_mesh) -> nn.Module:
    """Recursively discovers and fully_shards all submodules owning parameters dynamically."""
    for _, submodule in model.named_modules():
        if any(p.requires_grad for p in submodule.parameters(recurse=False)):
            fully_shard(submodule, device_mesh=device_mesh)
    fully_shard(model, device_mesh=device_mesh)
    return model


@pytest.mark.multi_rank
class TestStateDictManager(DistributedTestBase):
    """
    Test suite for StateDictManager handling local shards, full checkpoints,
    and parameter isolation.
    """

    def test_get_state_dict_local_format(self) -> None:
        """
        Verifies extraction of Local state dict returns rank-local shard shapes
        and values.
        """
        fsdp_model = _fully_shard_model(self.base_model, self.device_mesh)
        state_dict = StateDictManager.get_state_dict(
            fsdp_model,
            StateDictType.Local,
        )
        sharded_parameters = StateDictManager._collect_sharded_parameters(fsdp_model)

        assert set(state_dict.keys()) == {"fc1.weight", "fc2.weight"}

        for fqn, tensor in state_dict.items():
            _, expected_local_shape = self.expected_shapes[fqn]
            assert isinstance(tensor, DTensor)
            assert tensor.to_local().shape == expected_local_shape

            # Verify values match local shard storage directly
            expected_local_shard = sharded_parameters[fqn].local_shard
            torch.testing.assert_close(tensor.to_local(), expected_local_shard)

    def test_get_state_dict_full_format_matches_baseline(self) -> None:
        """
        Verifies extraction of Full state dict consolidates global tensors
        matching the unsharded baseline.
        """
        fsdp_model = _fully_shard_model(copy.deepcopy(self.base_model), self.device_mesh)
        full_state_dict = StateDictManager.get_state_dict(
            fsdp_model,
            StateDictType.Full,
        )

        assert set(full_state_dict.keys()) == {"fc1.weight", "fc2.weight"}

        # Compare against original reference base_model parameters
        for fqn, tensor in full_state_dict.items():
            global_shape, _ = self.expected_shapes[fqn]
            assert tensor.shape == global_shape

            baseline_parameter = getattr(self.base_model, fqn.split(".")[0]).weight
            torch.testing.assert_close(tensor, baseline_parameter)

    def test_get_state_dict_full_preserves_sharded_parameter_state(self) -> None:
        """
        Verifies that extracting a Full state dict gathers tensors during extraction
        but leaves parameters sharded.
        """
        fsdp_model = _fully_shard_model(self.base_model, self.device_mesh)
        sharded_parameters = StateDictManager._collect_sharded_parameters(fsdp_model)

        # Confirm all parameters start sharded
        for sharded_parameter in sharded_parameters.values():
            assert not sharded_parameter.is_gathered

        # Extract Full state dict
        full_state_dict = StateDictManager.get_state_dict(
            fsdp_model,
            StateDictType.Full,
        )

        # Assert returned state dict contains full tensors
        for fqn, tensor in full_state_dict.items():
            global_shape, _ = self.expected_shapes[fqn]
            assert tensor.shape == global_shape

        # Verify that underlying ShardedParameter instances remain
        # in sharded state (is_gathered=False)
        for sharded_parameter in sharded_parameters.values():
            assert not sharded_parameter.is_gathered
            _, expected_local_shape = self.expected_shapes[sharded_parameter.fqn]
            assert sharded_parameter.shape == expected_local_shape

    def test_load_state_dict_local_format(self) -> None:
        """Verifies reloading Local format state dict updates rank-local shards correctly."""
        fsdp_model_src = _fully_shard_model(copy.deepcopy(self.base_model), self.device_mesh)
        local_state_dict = StateDictManager.get_state_dict(
            fsdp_model_src,
            StateDictType.Local,
        )

        # Modify source weights slightly
        with torch.no_grad():
            for tensor in local_state_dict.values():
                tensor.add_(5.0)

        fsdp_model_target = _fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )
        StateDictManager.load_state_dict(
            fsdp_model_target,
            local_state_dict,
            StateDictType.Local,
        )

        # Verify target shards reflect updated values
        target_local_state_dict = StateDictManager.get_state_dict(
            fsdp_model_target,
            StateDictType.Local,
        )
        for fqn in local_state_dict:
            torch.testing.assert_close(target_local_state_dict[fqn], local_state_dict[fqn])

    def test_load_state_dict_full_format(self) -> None:
        """
        Verifies reloading Full format state dict
        correctly redistributes global tensors across ranks.
        """
        fsdp_model_src = _fully_shard_model(copy.deepcopy(self.base_model), self.device_mesh)
        full_state_dict = StateDictManager.get_state_dict(
            fsdp_model_src,
            StateDictType.Full,
        )

        fsdp_model_target = _fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )
        StateDictManager.load_state_dict(
            fsdp_model_target,
            full_state_dict,
            StateDictType.Full,
        )

        # Verify target extracted full state dict matches source full state dict
        target_full_state_dict = StateDictManager.get_state_dict(
            fsdp_model_target,
            StateDictType.Full,
        )
        for fqn in full_state_dict:
            torch.testing.assert_close(target_full_state_dict[fqn], full_state_dict[fqn])

    def test_load_state_dict_raises_key_error_on_unknown_parameter(self) -> None:
        """
        Verifies KeyError is raised when incoming state dict
        contains a parameter not present in module.
        """
        fsdp_model = _fully_shard_model(self.base_model, self.device_mesh)
        invalid_state_dict = {"nonexistent.weight": torch.zeros(2, 2)}

        with pytest.raises(KeyError):
            StateDictManager.load_state_dict(
                fsdp_model,
                invalid_state_dict,
                StateDictType.Local,
            )
