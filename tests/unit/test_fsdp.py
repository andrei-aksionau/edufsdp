import pytest
import torch
import torch.nn as nn

from edufsdp.fsdp import fully_shard
from edufsdp.hooks import BackwardPostHook, BackwardPreHook, ForwardPostHook, ForwardPreHook
from edufsdp.sharded_parameter import ShardedParameter
from edufsdp.state_dict import StateDictManager
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
    Multi-block model with nested submodules for testing
    different FSDP wrapping strategies.
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
class TestFSDP(DistributedTestBase):
    """Test suite for FSDP wrapper tree wrapping, parameter collection, and execution."""

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

        # Root wrap to trigger hierarchical traversal and FQN patching
        fully_shard(model, self.device_mesh)

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_sharded_parameters_property_completeness(self, strategy: str) -> None:
        """Verifies sharded_parameters collection and FQN patching across strategies."""
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        sharded_map = StateDictManager._collect_sharded_parameters(model)
        expected_fqns = {
            "block1.fc1.weight",
            "block1.fc2.weight",
            "block2.fc1.weight",
            "block2.fc2.weight",
            "lm_head.weight",
        }
        assert set(sharded_map.keys()) == expected_fqns

        for fqn, sharded_parameter in sharded_map.items():
            assert isinstance(sharded_parameter, ShardedParameter)
            assert sharded_parameter.fqn == fqn

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_hook_registration_on_submodules(self, strategy: str) -> None:
        """
        Verifies that exact FSDP hook instances are registered
        on owning submodules/containers.
        """
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        sharded_map = StateDictManager._collect_sharded_parameters(model)
        sharded_parameter_names = set(sharded_map.keys())

        for _, submodule in model.named_modules():
            if not hasattr(submodule, "_sharded_parameters"):
                continue

            sub_params = submodule._sharded_parameters
            submodule_parameter_names = set(sub_params.keys())

            sharded_in_submodule = sharded_parameter_names & submodule_parameter_names
            if not sharded_in_submodule:
                continue

            pre_hooks = list(submodule._forward_pre_hooks.values())
            post_hooks = list(submodule._forward_hooks.values())
            backward_pre_hooks = list(submodule._backward_pre_hooks.values())

            assert any(isinstance(hook, ForwardPreHook) for hook in pre_hooks)
            assert any(isinstance(hook, ForwardPostHook) for hook in post_hooks)
            assert any(isinstance(hook, BackwardPreHook) for hook in backward_pre_hooks)

            # Verify tensor-level backward post-hooks for parameters owned by this submodule
            for param_name in sharded_in_submodule:
                sharded_param = sub_params[param_name]
                if sharded_param.dtensor.requires_grad:
                    tensor_hooks = list(sharded_param.dtensor._backward_hooks.values())
                    assert any(isinstance(hook, BackwardPostHook) for hook in tensor_hooks)

            sharded_parameter_names -= sharded_in_submodule

        assert not sharded_parameter_names, (
            "FSDP sharded parameters were not verified on any module: "
            f"{sharded_parameter_names}"
        )

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_forward_and_backward_execution(self, strategy: str) -> None:
        """
        Verifies end-to-end forward and backward passes
        execute correctly across strategies.
        """
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        batch_size = 4
        input_tensor = torch.randn(batch_size, model.input_dim)

        output = model(input_tensor)
        assert output.shape == torch.Size([batch_size, model.output_dim])

        loss = output.sum()
        loss.backward()

        sharded_map = StateDictManager._collect_sharded_parameters(model)
        for sharded_param in sharded_map.values():
            parameter = getattr(sharded_param.module, sharded_param.parameter_name)
            assert parameter.grad is not None
            global_shape = parameter.grad.shape
            expected_local_shape = torch.Size([
                global_shape[0] // self.world_size,
                *global_shape[1:],
            ])
            assert parameter.grad.to_local().shape == expected_local_shape

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_missing_backward_pre_hook_raises_error(self, strategy: str) -> None:
        """
        Verifies that removing the backward pre-hook causes the backward pass
        to fail because parameters remain sharded when gradients are computed.
        """
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        # Manually remove/deregister the backward pre-hook from the root/owning modules
        for _, submodule in model.named_modules():
            if hasattr(submodule, "_backward_pre_hooks"):
                submodule._backward_pre_hooks.clear()

        batch_size = 4
        input_tensor = torch.randn(batch_size, model.input_dim, device=self.device)

        # Forward pass works normally
        output = model(input_tensor)
        loss = output.sum()

        # Attempting backward without parameters gathered should trigger an error
        expected_msg = r"setStorage: sizes .* are out of bounds for storage of size 0"
        with pytest.raises(RuntimeError, match=expected_msg):
            loss.backward()

    def test_sharded_parameters_fqn_propagation_through_nested_wrappers(self) -> None:
        """
        Verifies FQN consistency across independently wrapped modules
        when their parent modules are progressively wrapped.
        """
        model = ComplexTestModel().to(self.device)

        # First, independently shard leaf modules.
        fully_shard(model.block1.fc1, self.device_mesh)
        fully_shard(model.block2.fc1, self.device_mesh)

        # Next, wrap their parent blocks.
        fully_shard(model.block1, self.device_mesh)
        fully_shard(model.block2, self.device_mesh)

        # Finally, wrap the root model.
        fully_shard(model, self.device_mesh)

        expected_fqns = set(dict(model.named_parameters()))

        # Map each module object to its absolute name in the model hierarchy.
        module_names = {id(submodule): name for name, submodule in model.named_modules()}

        observed_fqns = set()

        # Inspect every FSDP unit, including nested independently wrapped units.
        for wrapper_name, wrapper_module in model.named_modules():
            if not hasattr(wrapper_module, "_sharded_parameters"):
                continue

            for fqn, sharded_parameter in wrapper_module._sharded_parameters.items():
                owner_name = module_names[id(sharded_parameter.module)]

                expected_fqn = (
                    f"{owner_name}.{sharded_parameter.parameter_name}"
                    if owner_name
                    else sharded_parameter.parameter_name
                )

                assert fqn == expected_fqn, (
                    f"Incorrect FQN in wrapper '{wrapper_name}': "
                    f"got '{fqn}', expected '{expected_fqn}'"
                )

                assert sharded_parameter.fqn == expected_fqn

                observed_fqns.add(fqn)

        # Every model parameter must belong to exactly one FSDP unit.
        assert observed_fqns == expected_fqns

        # Recursive collection must also preserve all absolute FQNs.
        collected_parameters = StateDictManager._collect_sharded_parameters(model)

        assert set(collected_parameters) == expected_fqns

        for fqn, sharded_parameter in collected_parameters.items():
            assert sharded_parameter.fqn == fqn

    def test_fully_shard_is_idempotent_for_parameterless_root(self) -> None:
        """
        Verifies that repeatedly wrapping a root with only independently
        wrapped descendants doesn't modify their parameter FQNs.
        """
        model = ComplexTestModel().to(self.device)

        # Independently wrap every parameter-bearing leaf module.
        leaf_modules = [
            model.block1.fc1,
            model.block1.fc2,
            model.block2.fc1,
            model.block2.fc2,
            model.lm_head,
        ]

        for leaf_module in leaf_modules:
            fully_shard(leaf_module, self.device_mesh)

        # All parameter-bearing descendants are already independent FSDP units.
        fully_shard(model, self.device_mesh)

        def snapshot_fqns() -> dict[tuple[str, str], tuple[int, str]]:
            return {
                (module_name, fqn): (
                    id(sharded_parameter),
                    sharded_parameter.fqn,
                )
                for module_name, submodule in model.named_modules()
                if hasattr(submodule, "_sharded_parameters")
                for fqn, sharded_parameter in submodule._sharded_parameters.items()
            }

        original_snapshot = snapshot_fqns()

        # Repeated wrapping must not modify existing metadata.
        fully_shard(model, self.device_mesh)
        fully_shard(model, self.device_mesh)

        assert snapshot_fqns() == original_snapshot
