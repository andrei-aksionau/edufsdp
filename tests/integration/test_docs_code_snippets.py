"""
Scans documentation markdown files for executable `verify_via_test` code blocks
and executes them across multiple CPU processes using torch.multiprocessing.spawn.
"""

import re
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

from tests.conftest import _init_distributed_environment, get_free_port


class MarkdownSnippet(NamedTuple):
    """Container for an extracted documentation code snippet."""

    name: str
    code_str: str
    world_size: int


def _run_markdown_snippet_worker(
    rank: int,
    world_size: int,
    port: str,
    code_str: str,
) -> None:
    """Target function executed inside each worker process for markdown code snippets."""

    _init_distributed_environment(rank, world_size, port)

    # Since code snippets contain their own imports, an empty or minimal global scope suffices
    exec(code_str, {})  # ruff: ignore[exec-builtin]


def _get_markdown_code_snippets() -> list[tuple[str, str, int]]:
    """
    Scans all .md files in the docs/ directory for ```python verify_via_test blocks.

    Returns an instance of MarkdownSnippet containing name, code content and world size.
    """

    # Navigate up from tests/integration/test_docs_code_snippets.py to project root
    docs_dir = Path(__file__).parents[2] / "docs"
    if not docs_dir.exists():
        return []

    snippets = []
    # Use verbose regex with comments (re.VERBOSE) to keep the pattern readable
    pattern = re.compile(
        r"""
        ```python\s+verify_via_test  # Match opening fence and marker
        (?:\s+world_size=(\d+))?      # Optional world_size capture group
        [^\r\n]*                      # Consume remainder of the fence line
        \r?\n                         # Newline
        (.*?)                         # Capture the code block content non-greedily
        ```                           # Match closing fence
        """,
        re.DOTALL | re.VERBOSE,
    )

    for md_path in docs_dir.glob("*.md"):
        content = md_path.read_text(encoding="utf-8")
        for match in pattern.finditer(content):
            world_size_str, code_str = match.groups()
            world_size = int(world_size_str) if world_size_str else 2

            line_no = content[: match.start()].count("\n") + 1
            name = f"{md_path.name}::line_{line_no}"
            snippets.append(
                MarkdownSnippet(
                    name=name,
                    code_str=code_str.strip(),
                    world_size=world_size,
                )
            )

    return snippets


@pytest.mark.parametrize(
    "snippet",
    _get_markdown_code_snippets(),
    ids=lambda s: s.name,
)
def test_markdown_code_snippet(snippet: MarkdownSnippet) -> None:
    """
    Spawns distributed worker processes to execute an extracted documentation
    code snippet.
    """

    port = get_free_port()
    torch.multiprocessing.spawn(
        _run_markdown_snippet_worker,
        args=(snippet.world_size, port, snippet.code_str),
        nprocs=snippet.world_size,
        join=True,
    )
