"""Unit tests for edufsdp.model containing toy MLP network definitions."""

from typing import Any

import pytest
import torch
from torch import distributed as dist

from edufsdp.model import MLP


@pytest.mark.multi_rank
class TestMLP:
    """
    Test suite verifying dynamic dimensions, weights,
    and forward/backward execution of MLP.
    """

    def setup_method(self, method: Any) -> None:
        """Setup test class args for later use."""
        if dist.is_initialized():
            self.world_size = dist.get_world_size()

    def test_mlp_initialization_and_layer_dimensions(self) -> None:
        """Verifies that layer dimensions scale dynamically with process world size."""
        model = MLP()

        expected_in = 3 * self.world_size
        expected_hidden = 4 * self.world_size
        expected_out = 2 * self.world_size

        assert model.input_dim == expected_in
        assert model.hidden_dim == expected_hidden
        assert model.output_dim == expected_out

        # PyTorch linear weights have shape (out_features, in_features)
        assert model.layer_1.weight.shape == torch.Size([expected_hidden, expected_in])
        assert model.layer_1.bias.shape == torch.Size([expected_hidden])

        assert model.layer_2.weight.shape == torch.Size([expected_out, expected_hidden])
        assert model.layer_2.bias.shape == torch.Size([expected_out])

    def test_mlp_forward_pass_output_shape_and_relu(self) -> None:
        """Verifies output shape correctness and non-linear activation during forward pass."""
        model = MLP()
        batch_size = 4

        # Deterministic input batch
        x = torch.randn(batch_size, model.input_dim)
        output = model(x)

        # Output shape verification
        assert output.shape == torch.Size([batch_size, model.output_dim])

        # Verify ReLU activation explicitly clamps negative values in intermediate output
        hidden_before_relu = model.layer_1(x)
        hidden_after_relu = model.relu(hidden_before_relu)
        assert torch.all(hidden_after_relu >= 0.0)

    def test_mlp_backward_pass_gradient_generation(self) -> None:
        """
        Verifies gradient computation across all trainable parameters during
        backward pass.
        """
        model = MLP()
        batch_size = 4

        x = torch.randn(batch_size, model.input_dim)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are non-zero for all parameters
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Gradient missing for parameter {name}"
            assert parameter.grad.shape == parameter.shape, (
                f"Gradient shape mismatch for parameter {name}"
            )
            assert not torch.all(torch.isclose(parameter.grad, torch.tensor(0.0))), (
                f"Gradient is all zeros for parameter {name}"
            )
