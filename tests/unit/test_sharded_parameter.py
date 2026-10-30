"""
Unit tests for edufsdp.sharded_parameter covering
DTensor sharding and state transitions.
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from edufsdp.sharded_parameter import ShardedParameter
from tests.base.test_class import DistributedTestBase


# TODO: test changes, like properties
@pytest.mark.multi_rank
class TestShardedParameter(DistributedTestBase):
    """
    Test suite verifying ShardedParameter lifecycle, tensor attributes,
    and state transitions.
    """

    def test_sharded_parameter_initialization(self) -> None:
        """Verifies initial DTensor conversion, local shard shapes, and module assignment."""
        layer = self.base_model.fc1
        fqn = "fc1.weight"

        sharded_parameter = ShardedParameter(
            fqn=fqn,
            module=layer,
            parameter=nn.Parameter(layer.weight),
            device_mesh=self.device_mesh,
        )

        assert sharded_parameter.fqn == fqn
        assert sharded_parameter.parameter_name == "weight"
        assert sharded_parameter.module is layer
        assert sharded_parameter.device_mesh is self.device_mesh

        # Verify attribute was replaced with DTensor nn.Parameter
        assert isinstance(layer.weight, nn.Parameter)
        assert isinstance(layer.weight.data, DTensor)

        # Verify shape metadata
        _, expected_local_shape = self.expected_shapes[fqn]
        assert sharded_parameter.local_shard.shape == expected_local_shape
        assert not sharded_parameter.is_gathered
        assert sharded_parameter.shape == expected_local_shape

    def test_set_unsharded_and_set_sharded_lifecycle(self) -> None:
        """
        Verifies symmetric transition between sharded DTensor
        and full unsharded tensor state.
        """
        layer = self.base_model.fc1
        fqn = "fc1.weight"
        global_shape, local_shape = self.expected_shapes[fqn]

        sharded_parameter = ShardedParameter(
            fqn=fqn,
            module=layer,
            parameter=nn.Parameter(layer.weight),
            device_mesh=self.device_mesh,
        )

        # 1. State before gather (sharded)
        assert not sharded_parameter.is_gathered
        assert sharded_parameter.shape == local_shape
        assert sharded_parameter._gathered_tensor is None
        assert isinstance(layer.weight, nn.Parameter)
        assert isinstance(layer.weight.data, DTensor)

        # 2. Attach gathered full tensor (unsharded)
        fake_gathered_tensor = torch.randn(global_shape)
        sharded_parameter.set_unsharded(fake_gathered_tensor)

        assert sharded_parameter.is_gathered
        assert sharded_parameter.shape == global_shape
        assert layer.weight is fake_gathered_tensor
        assert isinstance(layer.weight, Tensor)
        assert not isinstance(layer.weight, nn.Parameter)

        # 3. Restore local shard (sharded)
        sharded_parameter.set_sharded()

        assert not sharded_parameter.is_gathered
        assert sharded_parameter.shape == local_shape
        assert sharded_parameter._gathered_tensor is None
        assert isinstance(layer.weight, nn.Parameter)
        assert isinstance(layer.weight.data, DTensor)

    def test_local_shard_tensor_values_match_global_chunk(self) -> None:
        """
        Verifies that local DTensor shard data matches
        the expected slice of the original weight.
        """
        layer = self.base_model.fc1
        fqn = "fc1.weight"
        original_data = layer.weight.data.clone()

        sharded_parameter = ShardedParameter(
            fqn=fqn,
            module=layer,
            parameter=nn.Parameter(layer.weight),
            device_mesh=self.device_mesh,
        )

        # Dimension 0 sharding: calculate rank slice index
        shard_size = original_data.shape[0] // self.world_size
        start_idx = self.rank * shard_size
        end_idx = start_idx + shard_size
        expected_rank_chunk = original_data[start_idx:end_idx]

        torch.testing.assert_close(sharded_parameter.local_shard, expected_rank_chunk)
