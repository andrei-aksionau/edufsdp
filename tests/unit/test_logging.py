"""
Unit tests for edufsdp.logging covering ASCIIVisualizer layout rendering and RankLogger
output.
"""

import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import distributed as dist

from edufsdp.logging import ASCIIVisualizer, RankLogger
from tests.base.test_class import DistributedTestBase


class TestASCIIVisualizer:
    """Test suite for ASCIIVisualizer shard bar rendering and transition state mapping."""

    @pytest.mark.parametrize("world_size", (1, 2, 4))
    def test_ascii_visualizer_shard_bar_rendering(self, world_size: int) -> None:
        """Verifies rank shard bar rendering across different world sizes."""
        bar_width = 16
        for rank in range(world_size):
            visualizer = ASCIIVisualizer(bar_width=bar_width, rank=rank, world_size=world_size)
            shard_bar = visualizer.render_shard_bar(is_gathered=False)

            # Validate bar formatting and structure
            assert shard_bar.startswith("[")
            assert shard_bar.endswith("]")
            assert len(shard_bar) == bar_width + 2  # Including outer brackets

    def test_ascii_visualizer_gathered_bar_rendering(self) -> None:
        """Verifies fully gathered bar is completely filled for any rank."""
        visualizer = ASCIIVisualizer(bar_width=10, rank=1, world_size=4)
        gathered_bar = visualizer.render_shard_bar(is_gathered=True)
        assert gathered_bar == "[██████████]"

    def test_render_transition_bars(self) -> None:
        """
        Verifies correct transition bar mapping for expand
        and shrink shape transformations.
        """
        visualizer = ASCIIVisualizer(bar_width=10, rank=0, world_size=2)
        shape_sharded = torch.Size([5, 16])
        shape_gathered = torch.Size([10, 16])

        # Sharded -> Gathered (Expand)
        before, after = visualizer.render_transition_bars(shape_sharded, shape_gathered)
        assert before == visualizer.non_gathered_bar
        assert after == visualizer.gathered_bar

        # Gathered -> Sharded (Shrink)
        before, after = visualizer.render_transition_bars(shape_gathered, shape_sharded)
        assert before == visualizer.gathered_bar
        assert after == visualizer.non_gathered_bar

        # Gathered -> Gathered (No change)
        before, after = visualizer.render_transition_bars(shape_gathered, shape_gathered)
        assert before == visualizer.gathered_bar
        assert after == visualizer.gathered_bar


class TestRankLoggerUnit:
    """Test suite for RankLogger utility methods, prefixes, and mocked ranks."""

    @pytest.mark.parametrize(
        ("rank", "world_size", "provided", "expected"),
        [
            (0, 2, "test message", "[Rank 0/2] test message"),
            (1, 4, "hello world", "[Rank 1/4] hello world"),
        ],
    )
    def test_add_prefix_rank(
        self, rank: int, world_size: int, provided: str, expected: str
    ) -> None:
        """Verifies rank prefix formatting matches expectations."""
        with patch.dict(os.environ, {"RANK": str(rank), "WORLD_SIZE": str(world_size)}):
            rank_logger = RankLogger()
            assert rank_logger.add_prefix_rank(provided) == expected

    @pytest.mark.parametrize("rank", (0, 1))
    @pytest.mark.parametrize("world_size", (2, 3, 4))
    def test_log_transition_executes_without_exception(
        self, world_size: int, rank: int
    ) -> None:
        """
        Verifies log_transition runs successfully across multiple rank settings without
        raising errors.
        """
        with patch.dict(os.environ, {"RANK": str(rank), "WORLD_SIZE": str(world_size)}):
            rank_logger = RankLogger()
            rank_logger.log_hooks = True

            # Should execute successfully without throwing exceptions
            rank_logger.log_transition(
                event_name="all_gather",
                parameter_name="fc1.weight",
                shape_before=torch.Size([2, 16]),
                shape_after=torch.Size([4, 16]),
            )


@pytest.mark.multi_rank
class TestRankLoggerDistributed(DistributedTestBase):
    """Test suite for distributed logging functions requiring active process groups."""

    def test_log_rank0_only_prints_on_rank_zero(self) -> None:
        """
        Verifies log_rank0 prints on rank 0 and triggers barrier on other ranks without
        printing.
        """
        test_logger = RankLogger()

        with patch("builtins.print") as mock_print:
            test_logger.log_rank0("Rank 0 exclusive message")

            if self.rank == 0:
                mock_print.assert_called_once()
                args, _ = mock_print.call_args
                assert "Rank 0 exclusive message" in args[0]
            else:
                # Rank 1 should not print the message
                mock_print.assert_not_called()

    def test_log_ordered_sequential_output(self) -> None:
        """
        Verifies log_ordered prints strictly in rank order (Rank 0 then Rank 1) into a
        shared file.
        """
        test_logger = RankLogger()

        log_file_path = os.path.join(tempfile.gettempdir(), "edufsdp_log_ordered_test.txt")
        if self.rank == 0:
            open(log_file_path, "w").close()

        dist.barrier()

        def mock_print_and_record(*args: Any, **kwargs: Any) -> None:
            output_str = " ".join(str(arg) for arg in args)
            with open(log_file_path, "a") as shared_file:
                shared_file.write(f"{output_str}\n")

        with patch("builtins.print", side_effect=mock_print_and_record):
            test_logger.log_ordered("Ordered synchronization check")

        dist.barrier()

        if self.rank == 0:
            with open(log_file_path) as shared_file:
                lines = shared_file.readlines()

            msg_lines = [
                line.strip() for line in lines if "Ordered synchronization check" in line
            ]

            assert len(msg_lines) == 2
            assert "Rank 0/2" in msg_lines[0]
            assert "Rank 1/2" in msg_lines[1]

            if os.path.exists(log_file_path):
                os.remove(log_file_path)
