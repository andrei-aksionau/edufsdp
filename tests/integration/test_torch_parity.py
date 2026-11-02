"""Integration parity tests comparing edufsdp against PyTorch native FSDP v2 (fully_shard)."""

from typing import Any

import pytest
import torch
import torch.nn as nn
from torch import Size, Tensor
from torch import distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.tensor import DTensor

from edufsdp.state_dict import StateDictManager, StateDictType
from tests.base.test_class import DistributedTestBase, FSDPType


class TestParityBase(DistributedTestBase):
    def setup_method(self, method: Any) -> None:
        """Initializes fully sharded models via custom and reference FSDP implementations."""
        if not dist.is_initialized():
            return
        super().setup_method(method)

        self.wrapped_custom = self.fully_shard_model(FSDPType.CUSTOM)
        self.wrapped_reference = self.fully_shard_model(FSDPType.REFERENCE)

        self.sharded_parameters_custom = StateDictManager._collect_sharded_parameters(
            self.wrapped_custom
        )


@pytest.mark.multi_rank
class TestStateDictParity(TestParityBase):
    """
    Verifies state dictionary extraction parity (Local and Full) between EducationalFSDP and
    Native FSDP2.
    """

    def test_full_state_dict(self) -> None:
        """Verifies full (unsharded) state dict parity matches PyTorch native FSDP2."""
        custom_sd = StateDictManager.get_state_dict(
            self.wrapped_custom, state_dict_type=StateDictType.Full
        )
        reference_sd, _ = get_state_dict(
            self.wrapped_reference,
            optimizers=(),
            options=StateDictOptions(full_state_dict=True),
        )

        if self.rank == 0:
            assert set(custom_sd.keys()) == set(reference_sd.keys()), (
                "State dict keys mismatch between EducationalFSDP and Native FSDP!"
            )
            for key in custom_sd:
                torch.testing.assert_close(
                    custom_sd[key], reference_sd[key], rtol=1e-6, atol=1e-6
                )

    def test_local_state_dict(self) -> None:
        """Verifies local (sharded) state dict parity matches PyTorch native FSDP2."""
        custom_sd = StateDictManager.get_state_dict(
            self.wrapped_custom, state_dict_type=StateDictType.Local
        )
        reference_sd, _ = get_state_dict(
            self.wrapped_reference,
            optimizers=(),
            options=StateDictOptions(full_state_dict=False),
        )

        assert set(custom_sd.keys()) == set(reference_sd.keys()), (
            "Local state dict keys mismatch between EducationalFSDP and Native FSDP!"
        )
        for key in custom_sd:
            custom_value = custom_sd[key]
            reference_value = reference_sd[key]
            assert isinstance(custom_value, DTensor)
            assert isinstance(reference_value, DTensor)

            torch.testing.assert_close(
                custom_value.to_local(), reference_value.to_local(), rtol=1e-6, atol=1e-6
            )


@pytest.mark.multi_rank
class TestDTensorStructureParity(TestParityBase):
    """
    Verifies DTensor property, placement, mesh, and gradient structure parity against
    FSDP2.
    """

    def test_dtensor_properties_and_gradients(self) -> None:
        """
        Verifies parameter types, placements, device mesh, local shapes, and gradient
        DTensor types.
        """

        out_custom = self.wrapped_custom(self.dummy_input)
        out_custom.sum().backward()

        out_reference = self.wrapped_reference(self.dummy_input)
        out_reference.sum().backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")

            custom_parameter = getattr(getattr(self.wrapped_custom, layer_name), attr)
            reference_parameter = getattr(getattr(self.wrapped_reference, layer_name), attr)

            assert isinstance(custom_parameter, DTensor)
            assert isinstance(reference_parameter, DTensor)
            assert custom_parameter.placements == reference_parameter.placements
            assert custom_parameter.device_mesh == reference_parameter.device_mesh
            assert (
                custom_parameter.to_local().shape
                == reference_parameter.to_local().shape
                == expected_local_shape
            )

            assert isinstance(custom_parameter.grad, DTensor)
            assert isinstance(reference_parameter.grad, DTensor)
            assert (
                custom_parameter.grad.to_local().shape
                == reference_parameter.grad.to_local().shape
                == expected_local_shape
            )


@pytest.mark.multi_rank
class TestOptimizerStateParity(TestParityBase):
    """Verifies optimizer state sharding and structure parity against FSDP2."""

    def test_optimizer_state_sharding(self) -> None:
        """
        Verifies optimizer state buffer names, sharding shapes, and parity post optimizer
        step().
        """

        optimizer_custom = torch.optim.Adam(self.wrapped_custom.parameters(), lr=0.001)
        optimizer_reference = torch.optim.Adam(self.wrapped_reference.parameters(), lr=0.001)

        out_custom = self.wrapped_custom(self.dummy_input)
        out_custom.sum().backward()
        optimizer_custom.step()

        out_reference = self.wrapped_reference(self.dummy_input)
        out_reference.sum().backward()
        optimizer_reference.step()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")

            custom_parameter = getattr(getattr(self.wrapped_custom, layer_name), attr)
            reference_parameter = getattr(getattr(self.wrapped_reference, layer_name), attr)

            assert custom_parameter in optimizer_custom.state
            assert reference_parameter in optimizer_reference.state

            custom_state = optimizer_custom.state[custom_parameter]
            reference_state = optimizer_reference.state[reference_parameter]

            for state_key in ("exp_avg", "exp_avg_sq"):
                assert state_key in custom_state
                custom_tensor = custom_state[state_key].to_local()
                reference_tensor = reference_state[state_key].to_local()

                assert custom_tensor.shape == reference_tensor.shape == expected_local_shape


@pytest.mark.multi_rank
class TestExecutionLifecycleParity(TestParityBase):
    """
    Verifies strict parity of execution lifecycle state transitions between EducationalFSDP
    and Native FSDP2.
    """

    def test_sharding_after_wrapping(self) -> None:
        """
        1. Verifies that parameters are correctly sharded immediately upon wrapping for both
        models.
        """

        for parameter_name, (
            _global_shape,
            expected_local_shape,
        ) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            custom_param = getattr(getattr(self.wrapped_custom, layer_name), attr)
            reference_param = getattr(getattr(self.wrapped_reference, layer_name), attr)

            # Both must be sharded to local shape initially
            assert custom_param.to_local().shape == expected_local_shape
            assert reference_param.to_local().shape == expected_local_shape
            assert not self.sharded_parameters_custom[parameter_name].is_gathered

    def test_forward_parameters_are_gathered(self) -> None:
        """
        2. Verifies that parameters are all-gathered during the forward pass in both
        models.
        """

        custom_shapes_during_forward: dict[str, Size] = {}
        reference_shapes_during_forward: dict[str, Size] = {}

        def make_hook(model: nn.Module, shapes_dict: dict[str, Size]):
            def hook(module: nn.Module, _args: Any):
                for parameter_name in self.expected_shapes:
                    layer_name, attr = parameter_name.split(".")
                    if module == getattr(model, layer_name):
                        param = getattr(module, attr)
                        shapes_dict[parameter_name] = param.shape

            return hook

        registered_hooks = []
        for parameter_name in self.expected_shapes:
            for model, shapes_dict in zip(
                (self.wrapped_custom, self.wrapped_reference),
                (custom_shapes_during_forward, reference_shapes_during_forward),
                strict=False,
            ):
                parameter = getattr(model, parameter_name.split(".")[0])
                hook = parameter.register_forward_pre_hook(
                    make_hook(
                        model=model,
                        shapes_dict=shapes_dict,
                    )
                )
                registered_hooks.append(hook)

        self.wrapped_custom(self.dummy_input)
        self.wrapped_reference(self.dummy_input)

        for hook in registered_hooks:
            hook.remove()

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert custom_shapes_during_forward.get(parameter_name) == global_shape
            assert reference_shapes_during_forward.get(parameter_name) == global_shape

    def test_forward_parameters_resharded_after_forward(self) -> None:
        """
        3. Verifies that parameters are resharded back to local shapes after forward in both
        models.
        """

        self.wrapped_custom(self.dummy_input)
        self.wrapped_reference(self.dummy_input)

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            custom_param = getattr(getattr(self.wrapped_custom, layer_name), attr)
            reference_param = getattr(getattr(self.wrapped_reference, layer_name), attr)

            assert custom_param.to_local().shape == expected_local_shape
            assert reference_param.to_local().shape == expected_local_shape
            assert not self.sharded_parameters_custom[parameter_name].is_gathered

    def test_backward_parameters_are_gathered(self) -> None:
        """
        4. Verifies that parameters are all-gathered during the backward pass in both
        models.
        """

        custom_shapes_during_backward: dict[str, torch.Size] = {}
        reference_shapes_during_backward: dict[str, torch.Size] = {}

        def make_hook(model: nn.Module, shapes_dict: dict[str, Size]):
            def hook(module: nn.Module, _grad_output: Tensor | tuple[Tensor, ...]):
                for parameter_name in self.expected_shapes:
                    layer_name, attr = parameter_name.split(".")
                    if module == getattr(model, layer_name):
                        param = getattr(module, attr)
                        shapes_dict[parameter_name] = param.shape

            return hook

        registered_hooks = []
        for parameter_name in self.expected_shapes:
            for model, shapes_dict in zip(
                (self.wrapped_custom, self.wrapped_reference),
                (custom_shapes_during_backward, reference_shapes_during_backward),
                strict=False,
            ):
                parameter = getattr(model, parameter_name.split(".")[0])
                hook = parameter.register_forward_pre_hook(
                    make_hook(
                        model=model,
                        shapes_dict=shapes_dict,
                    )
                )
                registered_hooks.append(hook)

        self.wrapped_custom(self.dummy_input).sum().backward()
        self.wrapped_reference(self.dummy_input).sum().backward()

        for hook in registered_hooks:
            hook.remove()

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert custom_shapes_during_backward.get(parameter_name) == global_shape
            assert reference_shapes_during_backward.get(parameter_name) == global_shape

    def test_backward_gradients_are_reduce_scattered(self) -> None:
        """
        5. Verifies that gradients are reduce-scattered to local sharded shapes matching
        Native FSDP2.
        """

        self.wrapped_custom(self.dummy_input).sum().backward()
        self.wrapped_reference(self.dummy_input).sum().backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            custom_param = getattr(getattr(self.wrapped_custom, layer_name), attr)
            reference_param = getattr(getattr(self.wrapped_reference, layer_name), attr)

            assert custom_param.grad is not None
            assert reference_param.grad is not None

            custom_grad_local = custom_param.grad.to_local()
            reference_grad_local = reference_param.grad.to_local()

            assert (
                custom_grad_local.shape == reference_grad_local.shape == expected_local_shape
            )
            torch.testing.assert_close(
                custom_grad_local, reference_grad_local, rtol=1e-5, atol=1e-8
            )

    def test_parameters_resharded_after_backward(self) -> None:
        """
        6. Verifies that parameters are resharded back to local shapes after backward in
        both models.
        """

        self.wrapped_custom(self.dummy_input).sum().backward()
        self.wrapped_reference(self.dummy_input).sum().backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            custom_param = getattr(getattr(self.wrapped_custom, layer_name), attr)
            reference_param = getattr(getattr(self.wrapped_reference, layer_name), attr)

            assert custom_param.to_local().shape == expected_local_shape
            assert reference_param.to_local().shape == expected_local_shape
            assert not self.sharded_parameters_custom[parameter_name].is_gathered


@pytest.mark.multi_rank
class TestConvergenceParity(TestParityBase):
    """Verifies end-to-end training loop convergence parity against FSDP2."""

    def test_multi_step_training_convergence(self) -> None:
        """
        Verifies forward outputs and loss values match Native FSDP2 over a 5-step SGD
        training loop.
        """

        custom_optimizer = torch.optim.SGD(self.wrapped_custom.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(self.wrapped_reference.parameters(), lr=0.01)

        torch.manual_seed(100 + self.rank)
        criterion = nn.MSELoss()

        for step in range(1, 6):
            custom_optimizer.zero_grad()
            edu_out = self.wrapped_custom(self.dummy_input)
            edu_loss = criterion(edu_out, self.dummy_target)
            edu_loss.backward()
            custom_optimizer.step()

            reference_optimizer.zero_grad()
            reference_out = self.wrapped_reference(self.dummy_input)
            reference_loss = criterion(reference_out, self.dummy_target)
            reference_loss.backward()
            reference_optimizer.step()

            output_diff = torch.abs(edu_out - reference_out).max().item()
            loss_diff = abs(edu_loss.item() - reference_loss.item())

            assert output_diff < 1e-5, f"Forward output diverged at step {step}!"
            assert loss_diff < 1e-5, f"Loss value diverged at step {step}!"
