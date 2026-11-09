"""
Integration tests verifying model training convergence,
numerical stability, and parameter updates.
"""

from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from tests.base.test_class import DistributedTestBase, FSDPType


@pytest.mark.multi_rank
class TestCustomTrainingConvergence(DistributedTestBase):
    """
    Test suite covering end-to-end training execution, loss reduction,
    and weight updates with custom FSDP.
    """

    def setup_method(self, method: Any) -> None:
        """Initializes distributed base environment and sets up training components."""
        if not dist.is_initialized():
            return
        super().setup_method(method)

        self.wrapped_model = self.fully_shard_model(FSDPType.CUSTOM)
        self.optimizer = torch.optim.Adam(self.wrapped_model.parameters(), lr=0.1)
        self.criterion = nn.MSELoss()

    def test_loss_decreases_across_steps(self) -> None:
        """
        Verifies that training loss successfully decreases over
        a multi-step optimization loop.
        """
        losses = []

        for _ in range(5):
            self.optimizer.zero_grad()
            output = self.wrapped_model(self.dummy_input)
            loss = self.criterion(output, self.dummy_target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        assert len(losses) > 1, f"[Rank {self.rank}] Recorded insufficient loss values."
        start_loss, end_loss = losses[0], losses[-1]

        # TODO: ideally to test that the slop is negative, because out of 5 iterations
        # the 2-5 iters might have the same value, but since it's smaller than the first -
        # test passes. Although, in this case the slop will be also negative. So what, check if
        # all vals aren't identical? Or to have a different test for it?
        assert end_loss < start_loss, (
            f"[Rank {self.rank}] Loss failed to decrease: "
            f"initial={start_loss:.6f}, final={end_loss:.6f}"
        )

    def test_loss_numerical_stability(self) -> None:
        """Verifies that intermediate loss values do not become NaN or Inf during training."""
        for step in range(5):
            self.optimizer.zero_grad()
            output = self.wrapped_model(self.dummy_input)
            loss = self.criterion(output, self.dummy_target)

            assert not torch.isnan(loss).item(), (
                f"[Rank {self.rank}] Loss became NaN at step {step}!"
            )
            assert not torch.isinf(loss).item(), (
                f"[Rank {self.rank}] Loss became Inf at step {step}!"
            )

            loss.backward()
            self.optimizer.step()

    def test_parameters_update_after_optimization(self) -> None:
        """Verifies that model parameters actually change after an optimizer step."""
        initial_params = {
            name: param.clone() for name, param in self.wrapped_model.named_parameters()
        }

        self.optimizer.zero_grad()
        output = self.wrapped_model(self.dummy_input)
        loss = self.criterion(output, self.dummy_target)
        loss.backward()
        self.optimizer.step()

        for name, current_param in self.wrapped_model.named_parameters():
            initial_param = initial_params[name]
            curr_tensor = (
                current_param.to_local()
                if isinstance(current_param, DTensor)
                else current_param
            )
            init_tensor = (
                initial_param.to_local()
                if isinstance(initial_param, DTensor)
                else initial_param
            )

            assert not torch.equal(curr_tensor, init_tensor), (
                f"[Rank {self.rank}] Parameter {name} failed to update after optimizer step!"
            )

    def test_gradient_accumulation(self) -> None:
        """
        Verifies that gradients accumulate correctly across
        multiple backward passes without resetting.
        """
        # 1. Run first backward pass and capture baseline gradients
        self.optimizer.zero_grad()
        output_1 = self.wrapped_model(self.dummy_input)
        loss_1 = self.criterion(output_1, self.dummy_target)
        loss_1.backward()

        single_pass_grads = {
            name: param.grad.to_local().clone()
            for name, param in self.wrapped_model.named_parameters()
            if param.grad is not None and isinstance(param.grad, DTensor)
        }

        # 2. Run second backward pass without zeroing (accumulating gradients)
        output_2 = self.wrapped_model(self.dummy_input)
        loss_2 = self.criterion(output_2, self.dummy_target)
        loss_2.backward()

        # 3. Verify gradients accumulated properly and retained sharded shapes
        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            parameter = getattr(getattr(self.wrapped_model, layer_name), attr)

            assert parameter.grad is not None, (
                f"Rank {self.rank}: {parameter_name}.grad is None after accumulation."
            )

            local_grad = parameter.grad.to_local()

            # Verify shape invariance
            assert local_grad.shape == expected_local_shape, (
                f"Rank {self.rank}: Accumulated gradient shape mismatch "
                f"- Got {local_grad.shape}, Expected {expected_local_shape}"
            )

            # Verify value accumulation (doubled since inputs/targets are identical)
            expected_accumulated = single_pass_grads[parameter_name] * 2
            torch.testing.assert_close(
                local_grad,
                expected_accumulated,
                msg=(
                    f"Rank {self.rank}: Gradient accumulation values diverged "
                    f"for {parameter_name}"
                ),
            )

        # 4. Verify zero_grad clears or resets gradients successfully
        self.optimizer.zero_grad()
        for parameter_name in self.expected_shapes:
            layer_name, attr = parameter_name.split(".")
            parameter = getattr(getattr(self.wrapped_model, layer_name), attr)

            assert parameter.grad is None or torch.all(parameter.grad == 0), (
                f"Rank {self.rank}: Gradients were not cleared after zero_grad() "
                f"for {parameter_name}"
            )
