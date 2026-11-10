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

    def test_sharded_parameter_fqn_getter_setter(self) -> None:
        """Verifies that updating the fqn property automatically updates parameter_name."""
        layer = self.base_model.fc1
        initial_fqn = "fc1.weight"

        sharded_parameter = ShardedParameter(
            fqn=initial_fqn,
            module=layer,
            parameter=nn.Parameter(layer.weight),
            device_mesh=self.device_mesh,
        )

        assert sharded_parameter.fqn == initial_fqn
        assert sharded_parameter.parameter_name == "weight"

        # Update fqn to a hierarchical path (simulating parent/block sharding)
        new_fqn = "transformer_block_1.fc1.weight"
        sharded_parameter.fqn = new_fqn

        assert sharded_parameter.fqn == new_fqn
        assert sharded_parameter.parameter_name == "weight"

    def test_unsharded_buffer_lazy_allocation(self) -> None:
        """
        Verifies that accessing unsharded_buffer lazily allocates storage
        only when empty, preserving tensor metadata shell.
        """
        layer = self.base_model.fc1
        fqn = "fc1.weight"
        global_shape, _ = self.expected_shapes[fqn]

        sharded_parameter = ShardedParameter(
            fqn=fqn,
            module=layer,
            parameter=nn.Parameter(layer.weight),
            device_mesh=self.device_mesh,
        )

        # Initially zeroed storage upon setup
        assert not sharded_parameter.is_gathered

        # First access triggers lazy allocation
        buffer_first = sharded_parameter.unsharded_buffer
        assert sharded_parameter.is_gathered
        assert buffer_first.shape == global_shape

        # Subsequent access returns the same buffer without re-allocating
        buffer_second = sharded_parameter.unsharded_buffer
        assert buffer_second is buffer_first

    def test_sharded_parameter_shape_and_gathered_properties(self) -> None:
        """
        Verifies that shape and is_gathered properties correctly reflect
        the transition between sharded and unsharded states.
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

        # Initial state: sharded
        assert not sharded_parameter.is_gathered
        assert sharded_parameter.shape == local_shape

        # Accessing unsharded_buffer allocates storage but doesn't set module dict yet
        buffer = sharded_parameter.unsharded_buffer
        assert sharded_parameter.is_gathered
        assert sharded_parameter.shape == global_shape
        assert buffer.shape == global_shape

        # Releasing buffer resets gathered status and shape back to local shard
        sharded_parameter.release_unsharded_buffer()
        assert not sharded_parameter.is_gathered
        assert sharded_parameter.shape == local_shape

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
        assert isinstance(layer.weight, nn.Parameter)
        assert isinstance(layer.weight.data, DTensor)

        # 2. Attach gathered full tensor (unsharded)
        fake_gathered_tensor = torch.randn(global_shape)
        _ = (
            sharded_parameter.unsharded_buffer
        )  # Accessing full_buffer will resize it if it's zero
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
        assert isinstance(layer.weight, nn.Parameter)
        assert isinstance(layer.weight.data, DTensor)
