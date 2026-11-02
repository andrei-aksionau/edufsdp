"""Base test harness and model definitions for distributed unit tests."""

import copy
from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard as reference_fully_shard

from edufsdp.fsdp import fully_shard as custom_fully_shard


class FSDPType(Enum):
    """Supported types of model sharding implementation."""

    REFERENCE = "reference"  # PyTorch's implementation
    CUSTOM = "custom"  # edufsdp implementation


class SimpleMLP(nn.Module):
    """
    Simple multi-layer perceptron module for testing sharding and hooks.

    Attributes:
        fc1: First linear projection without bias (in_features=16, out_features=4).
        fc2: Second linear projection without bias (in_features=4, out_features=8).

    """

    def __init__(self) -> None:
        """Initializes linear layers with dimensions for the world size of 2."""
        assert dist.get_world_size() == 2, "Supports only tests for world size of 2"
        super().__init__()

        self.input_dim = 16
        self.hidden_dim = 4
        self.output_dim = 8

        # Weight shape: (out_features=4, in_features=16) -> Shard shape on 2 ranks: (2, 16)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        # Weight shape: (out_features=8, in_features=4) -> Shard shape on 2 ranks: (4, 4)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Executes forward pass through linear layers.

        Args:
            x: Input tensor of shape (batch_size, 16).

        Returns:
            Output tensor of shape (batch_size, 8).

        """
        return self.fc2(self.fc1(x))


class DistributedTestBase:
    """Base class for distributed unit tests executing under `torch.multiprocessing.spawn`."""

    def setup_method(self, method: Any) -> None:
        """Sets up CPU device mesh and model instances inside worker process."""
        if not dist.is_initialized():
            return

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device("cpu")
        self.device_mesh = init_device_mesh(str(self.device), (self.world_size,))

        # Identical seed guarantees identical baseline parameters on every rank
        torch.manual_seed(42)
        self.base_model = SimpleMLP().to(self.device)

        # Target expected tensor shapes: parameter_name -> (global_shape, local_shard_shape)
        self.expected_shapes = {
            "fc1.weight": (
                torch.Size([4, 16]),
                torch.Size([4 // self.world_size, 16]),
            ),
            "fc2.weight": (
                torch.Size([8, 4]),
                torch.Size([8 // self.world_size, 4]),
            ),
        }

    def teardown_method(self, method: Any) -> None:
        """Synchronizes workers and tears down Gloo process group."""
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def fully_shard_model(self, sharding_implementation_type: FSDPType) -> nn.Module:
        """Recursively discovers and fully_shards all submodules owning parameters dynamically."""
        model = copy.deepcopy(self.base_model)

        for _, submodule in model.named_modules():
            if any(p.requires_grad for p in submodule.parameters(recurse=False)):
                if sharding_implementation_type == FSDPType.REFERENCE:
                    reference_fully_shard(submodule, mesh=self.device_mesh)
                else:
                    custom_fully_shard(submodule, device_mesh=self.device_mesh)

        if sharding_implementation_type == FSDPType.REFERENCE:
            reference_fully_shard(model, mesh=self.device_mesh)
        else:
            custom_fully_shard(model, device_mesh=self.device_mesh)

        return model
