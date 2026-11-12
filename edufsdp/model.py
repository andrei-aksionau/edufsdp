"""
Toy module definitions used for testing and demonstrating Fully Sharded Data Parallel (FSDP).

This simple Multi-Layer Perceptron (MLP) acts as a clean baseline to observe:
  1. Module tree traversal and parameter sharding (`layer1.weight`, `layer2.weight`, etc.)
  2. All-Gather collective calls before linear layers during forward/backward passes.
  3. Memory freeing (dropping full unsharded tensors) post-computation.
"""

import torch.distributed as dist
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """
    A simple 2-layer Multi-Layer Perceptron (MLP).

    This network is purposely kept small with distinct input/hidden/output dimensions
    to make parameter shape tracking easy when sharding across multi-rank nodes.

    Layer dimensions are selected automatically based on the number of ranks.
    """

    def __init__(self) -> None:
        """Initializes linear layers with dimensions scaled by the distributed world size."""
        super().__init__()

        world_size = dist.get_world_size()
        self.input_dim = 3 * world_size
        self.hidden_dim = 4 * world_size
        self.output_dim = 2 * world_size

        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Executes forward pass.

        Args:
            x: Input batch tensor of shape `(batch_size, input_dim)`.

        Returns:
            Output tensor of shape `(batch_size, output_dim)`.
        """
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)
