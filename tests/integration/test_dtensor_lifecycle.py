"""
Integration tests verifying parameter DTensor types, placements,
sharded shapes, and lifecycle transitions.
"""

from typing import Any

import pytest
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from tests.base.test_class import DistributedTestBase, FSDPType


@pytest.mark.multi_rank
class TestCustomDTensorLifecycle(DistributedTestBase):
    """
    Test suite covering static sharding, runtime weight DTensor placements,
    and lifecycle invariants.
    """

    def setup_method(self, method: Any) -> None:
        """Initializes fully sharded model via custom FSDP implementation."""
        if not dist.is_initialized():
            return
        super().setup_method(method)

        self.wrapped_model = self.fully_shard_model(FSDPType.CUSTOM)

    def test_dtensor_resting_and_forward_lifecycle(self) -> None:
        """
        Verifies DTensor types, placements, and local shard shapes
        at rest and post-forward.
        """
        # 1. Check resting state before forward
        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(self.wrapped_model, layer_name), attr)

            assert isinstance(module_parameter, DTensor)
            assert module_parameter.device_mesh == self.device_mesh
            assert module_parameter.placements == (Shard(0),)
            assert module_parameter.to_local().shape == expected_local_shape

        # 2. Run Forward Pass
        _ = self.wrapped_model(self.dummy_input)

        # 3. Check resting state post-forward (should be resharded back to resting DTensor)
        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(self.wrapped_model, layer_name), attr)

            assert isinstance(module_parameter, DTensor)
            assert module_parameter.device_mesh == self.device_mesh
            assert module_parameter.placements == (Shard(0),)
            assert module_parameter.to_local().shape == expected_local_shape

    def test_dtensor_backward_lifecycle_and_gradients(self) -> None:
        """
        Verifies weight DTensors and gradient DTensor placements, shapes,
        and types post-backward.
        """
        output = self.wrapped_model(self.dummy_input)
        loss = output.sum()
        loss.backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(self.wrapped_model, layer_name), attr)

            # 1. Weight parameter checks (Type, Mesh, Placement, Sharded Shape)
            assert isinstance(module_parameter, DTensor)
            assert module_parameter.device_mesh == self.device_mesh
            assert module_parameter.placements == (Shard(0),)
            assert module_parameter.to_local().shape == expected_local_shape

            # 2. Gradient parameter checks (Type, Mesh, Placement, Local Shard Shape)
            assert module_parameter.grad is not None, (
                f"Rank {self.rank}: {parameter_name}.grad is None"
            )
            assert isinstance(module_parameter.grad, DTensor)
            assert module_parameter.grad.device_mesh == self.device_mesh
            assert module_parameter.grad.placements == (Shard(0),)
            assert module_parameter.grad.to_local().shape == expected_local_shape
