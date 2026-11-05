"""
Integration tests verifying parameter sharding, lifecycle shape transitions, and runtime
execution hook behaviors across strategies using ComplexTestModel.
"""

from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from edufsdp.fsdp import fully_shard
from tests.base.test_class import DistributedTestBase

if TYPE_CHECKING:
    from edufsdp.sharded_parameter import ShardedParameter


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
    Multi-block model with nested submodules for testing different FSDP wrapping
    strategies.
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
class TestShardingPlacement(DistributedTestBase):
    """
    Test suite covering static sharding, runtime weight lifecycle hooks, and sharded shape
    invariants across layer-level and block-level wrapping strategies.
    """

    def setup_method(self, method: Any) -> None:
        """Computes expected global and local shapes for ComplexTestModel."""
        super().setup_method(method)
        if not dist.is_initialized():
            return

        temp_model = ComplexTestModel()
        self.expected_shapes = {}
        for name, param in temp_model.named_parameters():
            global_shape = param.shape
            local_shape = torch.Size([global_shape[0] // self.world_size, *global_shape[1:]])
            self.expected_shapes[name] = (global_shape, local_shape)

        self.dummy_input = torch.randn(4, temp_model.input_dim, device=self.device)

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

    def _get_nested_attr(self, obj: Any, attr_path: str) -> Any:
        """Helper to retrieve nested module or parameter attributes by dot-separated path."""
        parts = attr_path.split(".")
        for part in parts:
            obj = getattr(obj, part)
        return obj

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_static_resting_sharding(self, strategy: str) -> None:
        """
        Verifies that all wrapped layer parameters rest in a sharded state matching rank
        placement.
        """

        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            *parent_path, attr = parameter_name.split(".")
            parent_module = self._get_nested_attr(model, ".".join(parent_path))
            module_parameter = getattr(parent_module, attr).to_local()
            assert module_parameter.shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} module resting shape "
                f"{module_parameter.shape} != {expected_local_shape}"
            )

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_forward_pre_hook_all_gather_shapes(self, strategy: str) -> None:
        """
        Verifies that ForwardPreHook gathers all parameters to global shape prior to forward
        execution.
        """

        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        gathered_shapes = {}

        # Register pre-hooks on every module holding sharded parameters
        for _, submodule in model.named_modules():
            if not hasattr(submodule, "_sharded_parameters"):
                continue

            def make_pre_hook(sub_mod: nn.Module):
                def hook(module: nn.Module, inputs: Any) -> None:
                    sharded_params = cast(
                        "dict[str, ShardedParameter]",
                        sub_mod._sharded_parameters,
                    )
                    for sharded_param in sharded_params.values():
                        param = getattr(sharded_param.module, sharded_param.parameter_name)
                        gathered_shapes[sharded_param.fqn] = param.shape

                return hook

            submodule.register_forward_pre_hook(make_pre_hook(submodule))

        _ = model(self.dummy_input)

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert parameter_name in gathered_shapes, (
                f"Rank {self.rank}: {parameter_name} not captured in pre-hooks"
            )
            assert gathered_shapes[parameter_name] == global_shape, (
                f"Rank {self.rank}: {parameter_name} gathered shape was "
                f"{gathered_shapes[parameter_name]}, expected {global_shape}"
            )

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_forward_post_hook_memory_cleanup_shapes(self, strategy: str) -> None:
        """
        Verifies that ForwardPostHook frees gathered parameters back to local sharded shapes
        post execution.
        """

        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        _ = model(self.dummy_input)

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            *parent_path, attr = parameter_name.split(".")
            parent_module = self._get_nested_attr(model, ".".join(parent_path))
            module_parameter = getattr(parent_module, attr).to_local()
            assert module_parameter.shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} post-forward shape was not freed "
                f"back to {expected_local_shape}, got {module_parameter.shape}"
            )

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_backward_pre_hook_all_gather_shapes(self, strategy: str) -> None:
        """Verifies that module weights are all-gathered during backward pass execution."""
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        pre_backward_parameter_shapes = {}

        for parameter_name, *_ in self.expected_shapes.items():
            *parent_path, attr = parameter_name.split(".")
            parent_module = self._get_nested_attr(model, ".".join(parent_path))

            def make_hook(full_key: str, mod: torch.nn.Module, attribute_name: str):
                def hook(module: torch.nn.Module, grad_output: tuple) -> None:
                    tensor_val = getattr(mod, attribute_name)
                    pre_backward_parameter_shapes[full_key] = tensor_val.shape

                return hook

            parent_module.register_full_backward_pre_hook(
                make_hook(parameter_name, parent_module, attr)
            )

        output = model(self.dummy_input)
        loss = output.sum()
        loss.backward()

        assert len(pre_backward_parameter_shapes) == len(self.expected_shapes), (
            f"Rank {self.rank}: Captured {len(pre_backward_parameter_shapes)} parameters, "
            f"expected {len(self.expected_shapes)}"
        )

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert pre_backward_parameter_shapes[parameter_name] == global_shape, (
                f"Rank {self.rank}: {parameter_name} shape during backward was "
                f"{pre_backward_parameter_shapes[parameter_name]}, "
                f"expected global shape {global_shape}"
            )

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_backward_gradients_reduce_scattered(self, strategy: str) -> None:
        """
        Verifies that gradients are computed, sharded, and match local parameter shapes post
        loss.backward().
        """

        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        torch.manual_seed(42)
        output = model(self.dummy_input)
        loss = output.sum()
        loss.backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            *parent_path, attr = parameter_name.split(".")
            parent_module = self._get_nested_attr(model, ".".join(parent_path))
            parameter = getattr(parent_module, attr)
            local_grad = parameter.grad.to_local()

            assert local_grad is not None, (
                f"Rank {self.rank}: {parameter_name} grad was None after loss.backward()"
            )

            grad_shape = local_grad.shape
            assert grad_shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} local grad shape {grad_shape} "
                f"does not match expected local parameter shape {expected_local_shape}"
            )

    @pytest.mark.parametrize("strategy", ["layer", "block"])
    def test_backward_post_hook_memory_cleanup_and_sharded_grads(self, strategy: str) -> None:
        """
        Verifies that module weights return to sharded shapes post-backward and grad holds
        sharded gradients.
        """

        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        output = model(self.dummy_input)
        loss = output.sum()
        loss.backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            *parent_path, attr = parameter_name.split(".")
            parent_module = self._get_nested_attr(model, ".".join(parent_path))
            module_parameter = getattr(parent_module, attr)

            assert module_parameter.to_local().shape == expected_local_shape, (
                f"Rank {self.rank}: Post-backward {parameter_name} weight shape was not freed "
                f"back to {expected_local_shape}, got {module_parameter.to_local().shape}"
            )

            assert module_parameter.grad.to_local() is not None, (
                f"Rank {self.rank}: {parameter_name}.grad was None on module parameter "
                "post-backward"
            )
            assert module_parameter.grad.to_local().shape == expected_local_shape, (
                f"Rank {self.rank}: Post-backward {parameter_name}.grad shape was "
                f"{module_parameter.grad.to_local().shape}, expected sharded shape "
                f"{expected_local_shape}"
            )

    @pytest.mark.parametrize("strategy", ["block"])
    def test_block_execution_isolation(self, strategy: str) -> None:
        """Verifies that executing one block leaves the other block's parameters sharded."""
        model = ComplexTestModel().to(self.device)
        self._apply_sharding(model, strategy)

        def verify_isolation(
            executing_block: nn.Module, resting_block: nn.Module, block_name: str
        ) -> None:
            gathered_states = []

            def pre_hook(module: nn.Module, inputs: Any) -> None:
                sharded_params = cast(
                    "dict[str, ShardedParameter]",
                    resting_block._sharded_parameters,
                )
                gathered_states.extend(sp.is_gathered for sp in sharded_params.values())

            handle = executing_block.register_forward_pre_hook(pre_hook)
            try:
                _ = model(self.dummy_input)
            finally:
                handle.remove()

            assert not any(gathered_states), (
                f"Rank {self.rank}: {block_name} parameters were gathered during execution "
                "of the other block"
            )

        # Check that block2 remains sharded while block1 executes
        verify_isolation(model.block1, model.block2, "block2")

        # Check that block1 remains sharded while block2 executes
        verify_isolation(model.block2, model.block1, "block1")
