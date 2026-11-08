"""
Rank-aware logger and ASCII visualization utilities for Educational FSDP.

Provides:
1. `ASCIIVisualizer`: Renders memory shard layouts (e.g., `[████████░░░░░░░░]`).
2. `RankLogger`: Handles distributed rank logging, barriers, and state transitions.
"""

import torch
import torch.distributed as dist


class ASCIIVisualizer:
    """Utility for rendering ASCII memory layout and parameter state transitions."""

    full_char = "█"
    empty_char = "░"

    def __init__(self, bar_width: int = 18, rank: int = 0, world_size: int = 1) -> None:
        """
        Initializes ASCII visualizer dimensions and pre-renders rank layout bars.

        Args:
            bar_width: Character width of the output ASCII bar.
            rank: Distributed rank index of current process.
            world_size: Total process count in distributed mesh.

        """
        self.bar_width = bar_width
        self.rank = rank
        self.world_size = world_size

        self.gathered_bar = f"[{self.full_char * self.bar_width}]"
        self.non_gathered_bar = self._prerender_shard_bar()

    def _prerender_shard_bar(self) -> str:
        if self.world_size <= 0 or self.bar_width <= 0:
            return f"[{self.empty_char * self.bar_width}]"

        # Calculate character slots for this rank's shard slice
        slot_size = self.bar_width / self.world_size
        start_slot = round(self.rank * slot_size)
        end_slot = round((self.rank + 1) * slot_size)
        # Clamp slots to bounds
        start_slot = max(0, min(self.bar_width, start_slot))
        end_slot = max(start_slot, min(self.bar_width, end_slot))

        left_pad = self.empty_char * start_slot
        shard_segment = self.full_char * (end_slot - start_slot)
        right_pad = self.empty_char * (self.bar_width - end_slot)

        return f"[{left_pad}{shard_segment}{right_pad}]"

    def render_shard_bar(self, is_gathered: bool) -> str:
        """
        Renders a single fixed-width ASCII bar representing memory layout for a rank.

        Examples (world_size=2, bar_width=18):
        - Gathered full tensor: [██████████████████]
        - Rank 0 local shard:   [█████████░░░░░░░░░]
        - Rank 1 local shard:   [░░░░░░░░░█████████]

        Args:
            is_gathered: Whether the target tensor is currently full or sharded.

        Returns:
            Formatted ASCII memory layout string enclosed in brackets.

        """
        return self.gathered_bar if is_gathered else self.non_gathered_bar

    def render_transition_bars(
        self,
        shape_before: torch.Size,
        shape_after: torch.Size,
    ) -> tuple[str, str]:
        """
        Calculates transition state bars (before -> after) based on shape changes.

        Args:
            shape_before: Tensor dimensions prior to transition event.
            shape_after: Tensor dimensions following transition event.

        Returns:
            Tuple containing formatted before and after ASCII layout strings.

        """
        numel_before = shape_before.numel()
        numel_after = shape_after.numel()

        if numel_before < numel_after:
            return self.non_gathered_bar, self.gathered_bar
        if numel_before > numel_after:
            return self.gathered_bar, self.non_gathered_bar
        return self.gathered_bar, self.gathered_bar


class RankLogger:
    """Distributed rank-aware logger with ASCII lifecycle visualization."""

    # Formatting layout constants
    LINE_WIDTH = 130
    LAYER_NAME_PREFIX_WIDTH = 10
    PARAM_COLUMN_WIDTH = 20
    EVENT_COLUMN_WIDTH = 15

    def __init__(self) -> None:
        """
        Initializes the logger without touching `torch.distributed`.

        Rank, world size and the visualizer are resolved lazily, because this object is
        created at import time, before the process group exists.
        """
        self.last_layer_prefix: str | None = None
        self.log_hooks = False

    @property
    def rank(self) -> int:
        """Rank of this process, or 0 if the process group is not initialized."""
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        """Number of processes, or 1 if the process group is not initialized."""
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def visualizer(self) -> ASCIIVisualizer:
        """ASCII visualizer matching the current rank and world size."""
        return ASCIIVisualizer(rank=self.rank, world_size=self.world_size)

    def add_prefix_rank(self, message: str) -> str:
        """
        Formats string with rank prefix.

        Args:
            message: Raw log string.

        Returns:
            Formatted message prefixed with process rank details.

        """
        return f"[Rank {self.rank}/{self.world_size}] {message}"

    def log(self, message: str) -> None:
        """Logs message with rank prefix."""
        print(self.add_prefix_rank(message), flush=True)

    def log_rank0(self, message: str) -> None:
        """Logs message only on process rank 0."""
        if self.rank == 0:
            print(message, flush=True)
        if dist.is_initialized():
            dist.barrier()

    def log_section_header(self, title: str) -> None:
        """Logs a centered major section header (e.g. 'Forward pass') on Rank 0."""
        banner = f" {title} ".center(self.LINE_WIDTH, "=")
        self.log_rank0(f"\n\n{banner}\n")

    def log_header(self, title: str, symbol: str = "#", width: int = 30) -> None:
        """Logs a prominent block header (e.g. 'LOSS VALUES') on Rank 0."""
        title = f" {title} ".center(width, symbol)
        self.log_rank0(f"\n\n{title}\n")

    def log_divider(self, symbol: str = "#", width: int = 30) -> None:
        """Logs a visual divider line on Rank 0."""
        self.log_rank0(f"\n{symbol * width}")

    def log_ordered(self, message: str, add_rank_info: bool = True) -> None:
        """Guarantees sequentially ordered stdout across all ranks via process barriers."""
        if not dist.is_initialized():
            if message.strip():
                self.log(message)
            return

        for rank in range(self.world_size):
            if self.rank == rank and message.strip():
                formatted = "\n".join(
                    self.add_prefix_rank(line) if add_rank_info else line
                    for line in message.splitlines()
                )
                print(formatted, flush=True)

            dist.barrier()

    def _extract_layer_prefix(self, parameter_name: str) -> str:
        """
        Extracts top-level layer prefix from the FQN (e.g., 'layer1' from 'layer1.weight').

        Args:
            parameter_name: Fully qualified parameter name.

        Returns:
            Extracted top-level layer attribute prefix.

        """
        prefix, _, _ = parameter_name.partition(".")
        return prefix if prefix else parameter_name

    def reset_layer_prefix(self) -> None:
        """Resets layer boundary tracking at training step boundaries."""
        self.last_layer_prefix = None

    def log_transition(
        self,
        event_name: str,
        parameter_name: str,
        shape_before: torch.Size,
        shape_after: torch.Size,
    ) -> None:
        """Logs parameter shape transitions and visual ASCII shard state."""
        if not self.log_hooks:
            return

        # Render memory layout transition
        before_bar, after_bar = self.visualizer.render_transition_bars(
            shape_before=shape_before,
            shape_after=shape_after,
        )

        current_prefix = self._extract_layer_prefix(parameter_name)

        # Print section dividers when moving to a new module layer on rank 0
        if current_prefix != self.last_layer_prefix:
            layer_name_text = f" {current_prefix} ".center(
                len(current_prefix) + self.LAYER_NAME_PREFIX_WIDTH * 2, "─"
            )
            layer_name_text = f" {layer_name_text} ".center(self.LINE_WIDTH, " ")
            layer_name_text = f"\n{layer_name_text}\n"
            self.log_rank0(layer_name_text)
        self.last_layer_prefix = current_prefix

        # Print entries to a table
        log_entry_divider = "─" if self.rank == 0 else "⁻"
        shape_str = f"{tuple(shape_before)} -> {tuple(shape_after)}"

        msg = (
            f"{log_entry_divider * self.LINE_WIDTH}\n"
            f"Rank {self.rank} │ {parameter_name:<{self.PARAM_COLUMN_WIDTH}} │ "
            f"{event_name:<{self.EVENT_COLUMN_WIDTH}} │ "
            f"Shard: {before_bar} -> {after_bar} │ "
            f"Shape: {shape_str}\n"
        )

        self.log_ordered(msg, add_rank_info=False)
        self.log_rank0(f"{'─' * self.LINE_WIDTH}\n")


logger = RankLogger()
