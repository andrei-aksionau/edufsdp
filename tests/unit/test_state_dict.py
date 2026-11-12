"""
Unit tests for edufsdp.state_dict covering Local and Full state dict extraction,
loading, parameter state preservation, and parameter collection across strategies.
"""

from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from edufsdp.fsdp import fully_shard
from edufsdp.state_dict import StateDictManager, StateDictType
from tests.base.test_class import DistributedTestBase


class SubBlock(nn.Module):
    """A multi-layer sub-block containing linear projections."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class ComplexTestModel(nn.Module):
    """
    Multi-block model with nested submodules for testing different
    FSDP wrapping strategies.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_dim = 16
        self.hidden_dim = 4
        self.output_dim = 8

        self.block1 = SubBlock(self.input_dim, self.hidden_dim, self.hidden_dim)
        self.block2 = SubBlock(self.hidden_dim, self.hidden_dim, self.output_dim)
        self.lm_head = nn.Linear(self.output_dim, self.output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.lm_head(x)


@pytest.mark.multi_rank
class TestStateDictManager1(DistributedTestBase):
    """
    Test suite for StateDictManager handling local shards, full checkpoints,
    parameter isolation, and collection across wrapping strategies.
    """

    def setup_method(self, method: Any) -> None:
        """
        Sets up device mesh and computes expected global/local shapes
        for ComplexTestModel.
        """
        super().setup_method(method)
        if not dist.is_initialized():
            return

        temp_model = ComplexTestModel()
        self.expected_shapes = {}
        for name, param in temp_model.named_parameters():
            global_shape = param.shape
            local_shape = torch.Size([global_shape[0] // self.world_size, *global_shape[1:]])
            self.expected_shapes[name] = (global_shape, local_shape)

    def _apply_sharding(self, model: ComplexTestModel, strategy: str) -> None:
        """Helper to apply either layer-level or block-level sharding before root wrap."""
        if strategy == "layer":
            fully_shard(model.block1.fc1, self.device_mesh)
            fully_shard(model.block1.fc2, self.device_mesh)
            fully_shard(model.block2.fc1, self.device_mesh)
            fully_shard(model.block2.fc2, self.device_mesh)
            fully_shard(model.lm_head, self.device_mesh)
        elif strategy == "block":
            fully_shard(model.block1, self.device_mesh)
            fully_shard(model.block2, self.device_mesh)
            fully_shard(model.lm_head, self.device_mesh)
        else:
            raise ValueError(f"Unknown sharding strategy: {strategy}")

        fully_shard(model, self.device_mesh)

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_collect_sharded_parameters(self, strategy: str) -> None:
        """Verifies that _collect_sharded_parameters extracts all expected parameter FQNs."""
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        sharded_map = StateDictManager._collect_sharded_parameters(model)
        assert set(sharded_map.keys()) == set(self.expected_shapes.keys())

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_get_state_dict_local_format(self, strategy: str) -> None:
        """
        Verifies extraction of Local state dict returns
        rank-local shard shapes and values.
        """
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        state_dict = StateDictManager.get_state_dict(model, StateDictType.Local)
        sharded_parameters = StateDictManager._collect_sharded_parameters(model)

        assert set(state_dict.keys()) == set(self.expected_shapes.keys())

        for fqn, tensor in state_dict.items():
            _, expected_local_shape = self.expected_shapes[fqn]
            assert isinstance(tensor, DTensor)
            assert tensor.to_local().shape == expected_local_shape

            expected_local_shard = sharded_parameters[fqn].local_shard
            torch.testing.assert_close(tensor.to_local(), expected_local_shard)

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_get_state_dict_full_format_matches_baseline(self, strategy: str) -> None:
        """
        Verifies extraction of Full state dict consolidates global tensors matching
        the unsharded baseline.
        """
        torch.manual_seed(42)
        base_model = ComplexTestModel().to(self.device)

        model = ComplexTestModel().to(self.device)
        model.load_state_dict(base_model.state_dict())
        self._apply_sharding(model, strategy)

        full_state_dict = StateDictManager.get_state_dict(model, StateDictType.Full)

        assert set(full_state_dict.keys()) == set(self.expected_shapes.keys())

        for fqn, tensor in full_state_dict.items():
            global_shape, _ = self.expected_shapes[fqn]
            assert tensor.shape == global_shape

            baseline_parameter = base_model.get_parameter(fqn)
            torch.testing.assert_close(tensor, baseline_parameter)

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_get_state_dict_full_preserves_sharded_parameter_state(
        self, strategy: str
    ) -> None:
        """
        Verifies that extracting a Full state dict gathers tensors during extraction
        but leaves parameters sharded.
        """
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)
        sharded_parameters = StateDictManager._collect_sharded_parameters(model)

        for sharded_parameter in sharded_parameters.values():
            assert not sharded_parameter.is_gathered

        _ = StateDictManager.get_state_dict(model, StateDictType.Full)

        for sharded_parameter in sharded_parameters.values():
            assert not sharded_parameter.is_gathered
            _, expected_local_shape = self.expected_shapes[sharded_parameter.fqn]
            assert sharded_parameter.shape == expected_local_shape

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_load_state_dict_local_format(self, strategy: str) -> None:
        """Verifies reloading Local format state dict updates rank-local shards correctly."""
        model_src = ComplexTestModel().to(self.device)
        self._apply_sharding(model_src, strategy)
        local_state_dict = StateDictManager.get_state_dict(model_src, StateDictType.Local)

        with torch.no_grad():
            for tensor in local_state_dict.values():
                tensor.add_(5.0)

        model_target = ComplexTestModel().to(self.device)
        self._apply_sharding(model_target, strategy)
        StateDictManager.load_state_dict(model_target, local_state_dict, StateDictType.Local)

        target_local_state_dict = StateDictManager.get_state_dict(
            model_target, StateDictType.Local
        )
        for fqn in local_state_dict:
            torch.testing.assert_close(target_local_state_dict[fqn], local_state_dict[fqn])

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_load_state_dict_full_format(self, strategy: str) -> None:
        """
        Verifies reloading Full format state dict correctly redistributes
        global tensors across ranks.
        """
        model_src = ComplexTestModel().to(self.device)
        self._apply_sharding(model_src, strategy)
        full_state_dict = StateDictManager.get_state_dict(model_src, StateDictType.Full)

        model_target = ComplexTestModel().to(self.device)
        self._apply_sharding(model_target, strategy)
        StateDictManager.load_state_dict(model_target, full_state_dict, StateDictType.Full)

        target_full_state_dict = StateDictManager.get_state_dict(
            model_target, StateDictType.Full
        )
        for fqn in full_state_dict:
            torch.testing.assert_close(target_full_state_dict[fqn], full_state_dict[fqn])

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_load_state_dict_raises_key_error_on_unknown_parameter(
        self, strategy: str
    ) -> None:
        """
        Verifies KeyError is raised when incoming state dict contains
        a parameter not present in module.
        """
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)
        invalid_state_dict = {"nonexistent.weight": torch.zeros(2, 2)}

        with pytest.raises(KeyError):
            StateDictManager.load_state_dict(model, invalid_state_dict, StateDictType.Local)
