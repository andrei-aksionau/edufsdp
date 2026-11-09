"""
Unit tests for edufsdp.hooks covering autograd all-gather/reduce-scatter
and execution hooks.
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from edufsdp.hooks import (
    BackwardPostHook,
    BackwardPreHook,
    ForwardPostHook,
    ForwardPreHook,
    FSDPAllGatherFunction,
)
from edufsdp.sharded_parameter import ShardedParameter
from tests.base.test_class import DistributedTestBase


@pytest.mark.multi_rank
class TestHooks(DistributedTestBase):
    """Test suite for autograd functions and execution hooks managing parameter lifecycles."""

    def test_fsdp_all_gather_function_forward_and_backward(self) -> None:
        """Verifies forward parameter all-gather and backward gradient reduce-scatter."""
        layer = self.base_model.fc1
        fqn = "fc1.weight"
        global_shape, local_shape = self.expected_shapes[fqn]

        sharded_parameter = ShardedParameter(
            fqn=fqn,
            module=layer,
            parameter=nn.Parameter(layer.weight),
            device_mesh=self.device_mesh,
        )

        local_parameter = sharded_parameter.local_shard.detach().requires_grad_(True)

        # 1. Test Forward All-Gather
        sharded_parameter.allocate_full()
        gathered_tensor = FSDPAllGatherFunction.apply(local_parameter, sharded_parameter)

        assert gathered_tensor.shape == global_shape
        assert gathered_tensor.requires_grad

        # 2. Test Backward Reduce-Scatter gradient computation
        fake_grad_output = torch.ones_like(gathered_tensor)
        gathered_tensor.backward(fake_grad_output)

        assert local_parameter.grad is not None
        assert local_parameter.grad.shape == local_shape

        # Verify mean reduction across world_size
        # (sum of ones across 2 ranks is 2; divided by # world_size=2 gives 1.0)
        torch.testing.assert_close(local_parameter.grad, torch.ones_like(local_parameter))

    def test_forward_pre_hook_and_forward_post_hook_lifecycle(self) -> None:
        """
        Verifies parameter gathering before forward execution
        and freeing immediately after.
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

        pre_hook = ForwardPreHook({sharded_parameter.fqn: sharded_parameter})
        post_hook = ForwardPostHook({sharded_parameter.fqn: sharded_parameter})

        # Initial state: sharded
        assert not sharded_parameter.is_gathered
        assert sharded_parameter.shape == local_shape

        # 1. Execute ForwardPreHook
        pre_hook(layer, (torch.randn(self.batch_size, self.base_model.input_dim),))

        assert sharded_parameter.is_gathered
        assert sharded_parameter.shape == global_shape
        assert isinstance(layer.weight, Tensor)

        # Re-triggering pre-hook when already gathered should be a no-op
        pre_hook(layer, (torch.randn(self.batch_size, self.base_model.input_dim),))
        assert sharded_parameter.is_gathered

        # 2. Execute ForwardPostHook
        post_hook(
            self.base_model,
            (torch.randn(self.batch_size, self.base_model.input_dim),),
            torch.randn(self.batch_size, self.base_model.hidden_dim),
        )

        assert not sharded_parameter.is_gathered
        assert sharded_parameter.shape == local_shape
        assert isinstance(layer.weight, nn.Parameter)

    def test_forward_hooks_with_multiple_parameters(self) -> None:
        """
        Verifies that forward hooks correctly manage multiple sharded parameters
        together.
        """

        # Setup multiple parameters (fc1 and fc2) managed by a single hook dictionary
        param1 = self.base_model.fc1.weight
        param2 = self.base_model.fc2.weight

        sharded_param1 = ShardedParameter(
            fqn="fc1.weight",
            module=self.base_model.fc1,
            parameter=nn.Parameter(param1),
            device_mesh=self.device_mesh,
        )
        sharded_param2 = ShardedParameter(
            fqn="fc2.weight",
            module=self.base_model.fc2,
            parameter=nn.Parameter(param2),
            device_mesh=self.device_mesh,
        )

        sharded_params_dict = {
            sharded_param1.fqn: sharded_param1,
            sharded_param2.fqn: sharded_param2,
        }

        pre_hook = ForwardPreHook(sharded_params_dict)
        post_hook = ForwardPostHook(sharded_params_dict)

        # Initially both should be sharded (not gathered)
        assert not sharded_param1.is_gathered
        assert not sharded_param2.is_gathered

        # Trigger pre-hook (simulating block entry)
        pre_hook(self.base_model, (torch.randn(self.batch_size, self.base_model.input_dim),))

        # Both parameters should now be all-gathered
        assert sharded_param1.is_gathered
        assert sharded_param2.is_gathered

        # Trigger post-hook (simulating block exit)
        post_hook(
            self.base_model,
            (torch.randn(self.batch_size, self.base_model.input_dim),),
            torch.randn(self.batch_size, self.base_model.output_dim),
        )

        # Both parameters should now be re-sharded / freed
        assert not sharded_param1.is_gathered
        assert not sharded_param2.is_gathered

    def test_backward_pre_hook_and_backward_post_hook_lifecycle(self) -> None:
        """Verifies parameter re-gathering during backward pass and memory release after."""
        layer = self.base_model.fc1
        fqn = "fc1.weight"
        global_shape, local_shape = self.expected_shapes[fqn]

        sharded_parameter = ShardedParameter(
            fqn=fqn,
            module=layer,
            parameter=nn.Parameter(layer.weight),
            device_mesh=self.device_mesh,
        )

        backward_pre_hook = BackwardPreHook({sharded_parameter.fqn: sharded_parameter})
        backward_post_hook = BackwardPostHook(sharded_parameter)

        # Initial state: sharded
        assert not sharded_parameter.is_gathered
        assert sharded_parameter.shape == local_shape

        # 1. Execute BackwardPreHook
        backward_pre_hook(layer, (torch.randn(self.batch_size, self.base_model.hidden_dim),))

        assert sharded_parameter.is_gathered
        assert sharded_parameter.shape == global_shape

        # Re-triggering backward pre-hook when already gathered should be a no-op
        backward_pre_hook(layer, (torch.randn(self.batch_size, self.base_model.hidden_dim),))
        assert sharded_parameter.is_gathered

        # 2. Execute BackwardPostHook
        backward_post_hook(
            layer,
            (torch.randn(self.batch_size, self.base_model.input_dim),),
            (torch.randn(self.batch_size, self.base_model.hidden_dim),),
        )

        assert not sharded_parameter.is_gathered
        assert sharded_parameter.shape == local_shape

    def test_backward_hooks_with_multiple_parameters(self) -> None:
        """
        Verifies that backward hooks correctly manage multiple sharded parameters
        together.
        """

        # Setup multiple parameters (fc1 and fc2) managed by a single hook dictionary
        param_1 = self.base_model.fc1.weight
        param_2 = self.base_model.fc2.weight

        sharded_param_1 = ShardedParameter(
            fqn="fc1.weight",
            module=self.base_model.fc1,
            parameter=nn.Parameter(param_1),
            device_mesh=self.device_mesh,
        )
        sharded_param_2 = ShardedParameter(
            fqn="fc2.weight",
            module=self.base_model.fc2,
            parameter=nn.Parameter(param_2),
            device_mesh=self.device_mesh,
        )

        backward_pre_hook = BackwardPreHook({
            sharded_param_1.fqn: sharded_param_1,
            sharded_param_2.fqn: sharded_param_2,
        })
        backward_post_hook_1 = BackwardPostHook(sharded_param_1)
        backward_post_hook_2 = BackwardPostHook(sharded_param_2)

        # Initially both should be sharded (not gathered)
        assert not sharded_param_1.is_gathered
        assert not sharded_param_2.is_gathered

        # Trigger backward pre-hook (simulating block entry)
        backward_pre_hook(self.base_model, (torch.randn(2, self.base_model.output_dim),))

        # Both parameters should now be all-gathered
        assert sharded_param_1.is_gathered
        assert sharded_param_2.is_gathered

        # Trigger backward post-hook (simulating block exit)
        backward_post_hook_1(
            self.base_model,
            (torch.randn(2, self.base_model.input_dim),),
            (torch.randn(2, self.base_model.output_dim),),
        )
        backward_post_hook_2(
            self.base_model,
            (torch.randn(2, self.base_model.input_dim),),
            (torch.randn(2, self.base_model.output_dim),),
        )

        # Both parameters should now be re-sharded / freed
        assert not sharded_param_1.is_gathered
        assert not sharded_param_2.is_gathered
