"""
Integration tests validating the end-to-end execution, distributed process
spawning via torchrun, and checkpoint restoration workflow of the FSDP inspection utility.
"""

import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]

import pytest


class TestDemoScript:
    """
    Test suite for the orchestration script, verifying correct multi-process
    launcher configuration and successful model state verification.
    """

    @pytest.mark.skipif(
        shutil.which("bash") is None,
        reason="Bash shell required to run the demo script",
    )
    def test_run_demo_script_with_bash(self) -> None:
        """
        Verifies that the FSDP inspection shell script executes successfully
        via torchrun and outputs the expected success confirmation.
        """

        bash_path = shutil.which("bash")
        assert bash_path is not None

        result = subprocess.run(
            [bash_path, "./run_demo.sh", "--nprocs", "2"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0
        assert (
            "Successfully loaded 'Local State Dict' into a fresh model instance"
            in result.stdout
        )

    def test_run_demo_script_as_executable(self) -> None:
        """
        Verifies that the FSDP inspection shell script executes successfully as executable
        and doesn't throw permission errors.
        """

        result = subprocess.run(
            ["./run_demo.sh", "--nprocs", "2"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.returncode == 0
        assert (
            "Successfully loaded 'Local State Dict' into a fresh model instance"
            in result.stdout
        )
