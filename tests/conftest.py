"""
Pytest setup and multi-process test runner configuration for edufsdp.

Provides custom test markers and Pytest execution hooks to run distributed
unit tests across multiple CPU processes using `torch.multiprocessing.spawn`.
"""

import os
import platform
import socket
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import distributed as dist

if TYPE_CHECKING:
    from collections.abc import Callable


def get_free_port() -> str:
    """Allocates an available free port on local host TCP socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def _init_distributed_environment(rank: int, world_size: int, port: str) -> None:
    """
    Configures standard PyTorch distributed environment variables and initializes
    the Gloo process group with loopback interface bindings.
    """
    # Configure PyTorch Distributed environment variables
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Explicit loopback interface binding for Gloo to prevent socket warnings (Unix only)
    if platform.system() == "Darwin":
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    elif platform.system() == "Linux":
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    else:
        # On Windows, let Gloo auto-detect the interface
        os.environ.pop("GLOO_SOCKET_IFNAME", None)

    # Initialize PyTorch distributed process group
    # Fallback to 28 seconds (slightly shorter than pytest timeout) for normal runs,
    # allow override during debugging
    timeout_seconds = int(os.environ.get("DIST_TIMEOUT_SECONDS", 28))
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )


def _spawn_worker(
    rank: int,
    world_size: int,
    port: str,
    test_func: Callable[..., Any],
    test_instance: Any,
    kwargs: dict[str, Any],
) -> None:
    """
    Target function executed inside each spawned worker process.

    Populates standard PyTorch distributed environment variables, sets up loopback
    network interfaces for Gloo, and executes the target test function or method.
    """

    _init_distributed_environment(rank, world_size, port)

    # Execute class methods with bound `self` instance, or plain functions directly
    if test_instance is not None:
        if hasattr(test_instance, "setup_method"):
            test_instance.setup_method(test_func)
        try:
            test_func(test_instance, **kwargs)
        finally:
            if hasattr(test_instance, "teardown_method"):
                test_instance.teardown_method(test_func)
    else:
        test_func(**kwargs)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """
    Intercepts marked test calls to execute them across spawned worker processes.

    If a test item or its parent class is marked with `@pytest.mark.multi_rank`,
    this hook intercepts standard single-process execution and instead spawns `world_size`
    isolated CPU worker processes.
    """
    marker = pyfuncitem.get_closest_marker("multi_rank")
    if not marker:
        return None  # Fall back to standard single-process pytest execution

    world_size: int = marker.kwargs.get("world_size", 2)
    port: str = get_free_port()

    # Extract raw underlying function object and target test instance (if class method)
    test_obj: Any = pyfuncitem.obj
    test_func = getattr(test_obj, "__func__", test_obj)
    test_instance = getattr(pyfuncitem, "instance", None)

    # Extract raw parametrized values prior to fixture evaluation to prevent
    # IPC serialization issues with evaluated complex objects across processes.
    kwargs: dict[str, Any] = {}
    if hasattr(pyfuncitem, "callspec"):
        callspec: Any = pyfuncitem.callspec
        kwargs = dict(callspec.params)

    # Spawn worker processes and wait for completion across all ranks
    torch.multiprocessing.spawn(
        _spawn_worker,
        args=(world_size, port, test_func, test_instance, kwargs),
        nprocs=world_size,
        join=True,
    )

    return True  # Signal to pytest that execution has been handled
