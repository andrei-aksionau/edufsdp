"""
Execution script for `edufsdp`.

Demonstrates:
1. Rank-local input batch sharding for Data Parallelism.
2. FSDP forward & backward execution with ASCII memory lifecycle logging.
3. Global loss reduction across ranks.
4. Exporting and reloading Local vs. Full state dict checkpoints.
5. CLI options for controlling iteration counts and memory visualizations.
"""

import argparse
import os
import platform
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from edufsdp.fsdp import fully_shard
from edufsdp.logging import logger
from edufsdp.model import MLP
from edufsdp.state_dict import StateDictManager, StateDictType

if TYPE_CHECKING:
    from collections.abc import Generator


def init_distributed() -> None:
    """
    Initializes default Gloo distributed environment and process group.

    Sets required rendezvous environment variables (`MASTER_ADDR`, `MASTER_PORT`)
    and configures OS-specific socket interfaces (`GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`)
    to force binding to local loopback interfaces on macOS (`lo0`) and Linux (`lo`),
    preventing IPv6 DNS resolution warnings and connection timeouts. Sets a rank-indexed
    seed for reproducible local data generation.
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    current_os = platform.system()
    if current_os == "Darwin":  # macOS
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
        os.environ["TP_SOCKET_IFNAME"] = "lo0"
    elif current_os == "Linux":  # Linux / Lightning AI Studio
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    else:
        # On Windows, let Gloo auto-detect the interface
        os.environ.pop("GLOO_SOCKET_IFNAME", None)

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    # Distinct seed per rank for reproducible rank-local data generation
    torch.manual_seed(42 + dist.get_rank())


@contextmanager
def log_hooks_enabled(enabled: bool = True) -> Generator[None]:
    """
    Context manager to safely toggle hook logging during a code block.

    Args:
        enabled (bool): If True, enables `logger.log_hooks` for the duration of the context
            and restores previous logger state upon exit. If False, yields immediately without
            altering logger state.

    Yields:
        None

    """
    if not enabled:
        yield
        return

    previous_state = logger.log_hooks
    logger.log_hooks = True
    try:
        yield
    finally:
        logger.log_hooks = previous_state
        logger.reset_layer_prefix()


def reduce_loss(local_loss: Tensor) -> float:
    """
    Computes mean loss across all distributed process ranks for logging.

    Performs an `all_reduce` sum across all process group ranks using `dist.ReduceOp.SUM`,
    scales the resulting scalar tensor by the world size, and returns the reduced Python float.

    Args:
        local_loss (Tensor): Local scalar loss tensor calculated on the current rank.

    Returns:
        float: Mean global loss calculated across all ranks.

    """
    loss_tensor = local_loss.detach().clone()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    loss_tensor /= dist.get_world_size()
    return loss_tensor.item()


def run_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: Tensor,
    targets: Tensor,
) -> None:
    """
    Executes a single forward and backward training step with optional hook visualization.

    Resets the logger step counter, executes model forward pass, reduces local and global
    loss values, backpropagates gradients, and updates model parameters using the optimizer.

    Args:
        model (nn.Module): FSDP-wrapped PyTorch module.
        optimizer (torch.optim.Optimizer): Optimizer instance attached to model parameters.
        criterion (nn.Module): Loss criterion used to calculate local loss.
        inputs (Tensor): Rank-local micro-batch input tensor.
        targets (Tensor): Rank-local target tensor corresponding to local inputs.

    """
    logger.reset_layer_prefix()

    # 1. Forward Pass
    if logger.log_hooks:
        logger.log_section_header("Forward Pass")

    outputs = model(inputs)
    local_loss = criterion(outputs, targets)

    # 2. Report Local & Reduced Global Loss
    global_loss = reduce_loss(local_loss)
    logger.log_header("Loss Values")
    logger.log_ordered(f"Local Loss: {local_loss.item():.4f}")
    logger.log_rank0(f"- Global Average Loss: {global_loss:.4f}")

    # 3. Backward Pass
    if logger.log_hooks:
        logger.reset_layer_prefix()
        logger.log_section_header("Backward Pass")

    optimizer.zero_grad()
    local_loss.backward()

    # 4. Optimizer Step
    optimizer.step()


def check_sd_loading(fsdp_model: nn.Module, device_mesh: DeviceMesh) -> None:
    """
    Validates state dict extraction and restoration across Local and Full formats.

    Args:
        fsdp_model: Trained FSDP model instance.
        device_mesh: Distributed device mesh layout.

    """
    logger.log_section_header("State Dict Checkpoint Verification")

    for sd_type in StateDictType:
        # Retrieve state dict from the provide model
        sd = StateDictManager.get_state_dict(fsdp_model, state_dict_type=sd_type)
        logger.log_header(f"{sd_type.name.capitalize()} State Dict")
        for name, tensor in sd.items():
            shape = tuple(
                tensor.to_local().shape if isinstance(tensor, DTensor) else tensor.shape
            )
            logger.log_rank0(f"  {name:<25} │ {shape}")
        logger.log_rank0("\n")

        # Create a new model with random weight, load state_dict into it
        fresh_fsdp_model = MLP()
        fully_shard(fresh_fsdp_model.layer1, device_mesh)
        fully_shard(fresh_fsdp_model.layer2, device_mesh)
        fully_shard(fresh_fsdp_model, device_mesh)
        StateDictManager.load_state_dict(fresh_fsdp_model, sd, state_dict_type=sd_type)

        # Verify that state dictionaries do match
        fresh_sd = StateDictManager.get_state_dict(fresh_fsdp_model, state_dict_type=sd_type)

        for key in sd:
            val_a = sd[key]
            val_b = fresh_sd[key]

            a = val_a.to_local() if isinstance(val_a, DTensor) else val_a
            b = val_b.to_local() if isinstance(val_b, DTensor) else val_b
            torch.testing.assert_close(a, b)

        logger.log_ordered(
            f"✓ Successfully loaded {sd_type.name.capitalize()} State Dict "
            "into fresh model instance"
        )
        logger.log_rank0("\n")


def parse_args() -> argparse.Namespace:
    """
    Configures command-line options for training steps and visual memory logs.

    Returns:
        Structured namespace containing parsed CLI options and runtime flags.

    """
    parser = argparse.ArgumentParser(description="Educational FSDP Training Script")
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of training iterations to run.",
    )
    parser.add_argument(
        "--show-memory-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ASCII memory state logs during training passes.",
    )
    parser.add_argument(
        "--memory-first-iter-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True and num_iterations > 1, show memory logs only during the first step.",
    )
    parser.add_argument(
        "--show-memory-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable ASCII memory state logs during state dict gathering.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution entrypoint for the FSDP demonstration script.

    Parses command-line arguments, initializes the Gloo process group and device mesh,
    instantiates the sharded MLP model, prepares rank-local input shards for Data
    Parallelism, executes the training step loop with selective ASCII memory logging,
    and validates state dict saving/loading.
    """
    args = parse_args()

    init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", mesh_shape=(world_size,))
    logger.log_header(f"NUMBER OF RANKS: {world_size}")

    # 1. Initialize sharded model, optimizer, and loss criterion
    model = MLP()
    fully_shard(model.layer1, mesh)
    fully_shard(model.layer2, mesh)
    fully_shard(model, mesh)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    # 2. Prepare a distinct shard of the total batch for each rank (Data Parallelism)
    batch_per_rank = 2
    global_inputs = torch.randn(
        world_size * batch_per_rank, model.input_dim, requires_grad=True
    )
    # Why 'requires_grad=True': a full_backward_hook fires only after the input gradient
    # is computed, and without a gradient on the input it fires early, resulting in premature
    # weights freeing. (This path was chosen for simplicity of this educational repo.)
    global_targets = torch.randn(world_size * batch_per_rank, model.output_dim)

    # 3. Slice rank-local input batch
    start_idx = rank * batch_per_rank
    end_idx = start_idx + batch_per_rank
    local_inputs = global_inputs[start_idx:end_idx]
    local_targets = global_targets[start_idx:end_idx]

    # 4. Execute training loop with selective memory logging
    for iter_idx in range(args.num_iterations):
        # Determine whether memory hooks should log during this iteration
        should_log = args.show_memory_training and (
            not args.memory_first_iter_only or iter_idx == 0
        )

        with log_hooks_enabled(enabled=should_log):
            run_train_step(model, optimizer, criterion, local_inputs, local_targets)

    # 5. Check state dict saving/loading with configurable hook visualization
    with log_hooks_enabled(args.show_memory_checkpoint):
        check_sd_loading(model, mesh)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
