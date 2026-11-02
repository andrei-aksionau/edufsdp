"""
Unit tests for edufsdp.fsdp covering module wrapping, parameter traversal, hooks, and state
dict delegation.
"""

from typing import Any

import pytest
import torch
from torch import distributed as dist

from edufsdp.hooks import BackwardPostHook, BackwardPreHook, ForwardPostHook, ForwardPreHook
from edufsdp.sharded_parameter import ShardedParameter
from edufsdp.state_dict import StateDictManager
from tests.base.test_class import DistributedTestBase, FSDPType


# TODO: tests how wrapping works with different models and different wrapping strategies
@pytest.mark.multi_rank
class TestFSDP(DistributedTestBase):
    """
    Test suite for FSDP wrapper tree wrapping, parameter collection, hook registration, and
    delegation.
    """

    def setup_method(self, method: Any) -> None:
        """Initializes fully sharded model via custom FSDP implementation."""
        if not dist.is_initialized():
            return
        super().setup_method(method)

        self.wrapped_model = self.fully_shard_model(FSDPType.CUSTOM)
        self.sharded_parameters = StateDictManager._collect_sharded_parameters(
            self.wrapped_model
        )

    def test_sharded_parameters_property_completeness(self) -> None:
        """
        Verifies that the sharded_parameters property correctly traverses the module tree
        and maps all parameters.
        """

        assert set(self.sharded_parameters.keys()) == {"fc1.weight", "fc2.weight"}

        for fqn, sharded_parameter in self.sharded_parameters.items():
            assert isinstance(sharded_parameter, ShardedParameter)
            assert sharded_parameter.fqn == fqn
            _, expected_local_shape = self.expected_shapes[fqn]
            assert sharded_parameter.local_shard.shape == expected_local_shape

    def test_hook_registration_on_submodules(self) -> None:
        """
        Verifies that exact FSDP hook instances are registered on submodules owning
        parameters.
        """
        sharded_parameter_names = set(self.sharded_parameters)

        for submodule_name, submodule in self.wrapped_model.named_modules():
            submodule_parameter_names = {
                f"{submodule_name}.{name}"
                for name, _ in submodule.named_parameters(recurse=False)
            }

            # Only check submodules that contain sharded parameters
            sharded_in_submodule = sharded_parameter_names & submodule_parameter_names
            if not sharded_in_submodule:
                continue

            # Verify specific FSDP hook instance types exist among registered hooks
            pre_hooks = list(submodule._forward_pre_hooks.values())
            post_hooks = list(submodule._forward_hooks.values())
            backward_pre_hooks = list(submodule._backward_pre_hooks.values())
            backward_post_hooks = list(submodule._backward_hooks.values())

            assert any(isinstance(hook, ForwardPreHook) for hook in pre_hooks)
            assert any(isinstance(hook, ForwardPostHook) for hook in post_hooks)
            assert any(isinstance(hook, BackwardPreHook) for hook in backward_pre_hooks)
            assert any(isinstance(hook, BackwardPostHook) for hook in backward_post_hooks)

            # Mark these sharded parameters as verified
            sharded_parameter_names -= sharded_in_submodule

        # Ensure every sharded parameter belonged to a module with valid hooks
        assert not sharded_parameter_names, (
            "FSDP sharded parameters were not verified on any submodule: "
            f"{sharded_parameter_names}"
        )

    def test_fsdp_forward_and_backward_execution(self) -> None:
        """
        Verifies end-to-end forward and backward passes execute correctly through the FSDP
        wrapper.
        """

        # 1. Forward Pass
        output = self.wrapped_model(self.dummy_input)
        assert output.shape == torch.Size([self.batch_size, self.base_model.output_dim])

        # 2. Backward Pass
        loss = output.sum()
        loss.backward()

        # 3. Verify gradients are correctly reduced and sharded across ranks
        for fqn in self.sharded_parameters:
            parameter = getattr(self.wrapped_model, fqn.split(".")[0]).weight
            assert parameter.grad is not None
            _, expected_local_shape = self.expected_shapes[fqn]
            assert parameter.grad.to_local().shape == expected_local_shape
