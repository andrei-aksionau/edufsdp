"""
Unit tests for edufsdp.fsdp covering module wrapping, parameter traversal, hooks, and state
dict delegation.
"""

import pytest
import torch

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

    def test_sharded_parameters_property_completeness(self) -> None:
        """
        Verifies that the sharded_parameters property correctly traverses the module tree
        and maps all parameters.
        """
        fsdp_model = self.fully_shard_model(FSDPType.CUSTOM)
        sharded_map = StateDictManager._collect_sharded_parameters(fsdp_model)

        assert set(sharded_map.keys()) == {"fc1.weight", "fc2.weight"}, (
            f"{sharded_map.keys()=}"
        )

        for fqn, sharded_parameter in sharded_map.items():
            assert isinstance(sharded_parameter, ShardedParameter)
            assert sharded_parameter.fqn == fqn, f"{sharded_parameter.fqn=}"
            _, expected_local_shape = self.expected_shapes[fqn]
            assert sharded_parameter.local_shard.shape == expected_local_shape

    def test_hook_registration_on_submodules(self) -> None:
        """
        Verifies that exact FSDP hook instances are registered on submodules owning
        parameters.
        """
        fsdp_model = self.fully_shard_model(FSDPType.CUSTOM)
        sharded_parameter_names = set(StateDictManager._collect_sharded_parameters(fsdp_model))

        for submodule_name, submodule in fsdp_model.named_modules():
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
        fsdp_model = self.fully_shard_model(FSDPType.CUSTOM)
        sharded_parameters = StateDictManager._collect_sharded_parameters(fsdp_model)
        batch_size = 4

        input_tensor = torch.randn(batch_size, 16)

        # 1. Forward Pass
        output = fsdp_model(input_tensor)
        assert output.shape == torch.Size([batch_size, 8])

        # 2. Backward Pass
        loss = output.sum()
        loss.backward()

        # 3. Verify gradients are correctly reduced and sharded across ranks
        for fqn in sharded_parameters:
            parameter = getattr(fsdp_model, fqn.split(".")[0]).weight
            assert parameter.grad is not None
            _, expected_local_shape = self.expected_shapes[fqn]
            assert parameter.grad.to_local().shape == expected_local_shape
