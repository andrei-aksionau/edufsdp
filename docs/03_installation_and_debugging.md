<p><h1 align="center">Installation, Setup & Debugging Guide</h1></p>

<br>

This document provides a practical guide for setting up the edufsdp development environment, running the distributed test suite and demonstration scripts, leveraging pre-configured VS Code debugging tools, wrapping custom modules, and troubleshooting multi-rank CPU execution hurdles.

<br>

## 1. Environment Setup & Installation

`edufsdp` relies on modern Python tooling and requires **Python 3.14** and **PyTorch 2.14** to guarantee strict version parity.

### Prerequisites

* `uv` package manager ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/))

uv handles everything out of the box - it will automatically fetch and provision the correct Python runtime, create an isolated virtual environment, and sync all locked dependencies defined in *pyproject.toml*.

### Installation Steps

1\. **Clone the Repository:**

```bash
git clone https://github.com/andrei-aksionau/edufsdp.git
cd edufsdp
```

2\. **Sync Dependencies:**

```bash
uv sync
```

<br>

## 2. Running the Demonstration Script

To watch live weight sharding visualizations, hook executions, loss reduction, and state saving in action across multiple CPU worker processes, execute the root-level shell script:

```bash
./run_inspect_fsdp.sh
```

This script automates `torchrun` across 6 CPU processes using the Gloo backend to execute `inspect_fsdp.py`, running a live training iteration with step-by-step ASCII memory state logging and checkpoint verification.

Here is an example output:

```text
========================================================== Forward Pass ==========================================================


                                                    ───────── layer1 ─────────

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Rank 0 │ layer1.weight        │ all_gather      │ Shard: [██████░░░░░░░░░░░░] -> [██████████████████] │ Shape: (4, 9) -> (12, 9)
⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
Rank 1 │ layer1.weight        │ all_gather      │ Shard: [░░░░░░██████░░░░░░] -> [██████████████████] │ Shape: (4, 9) -> (12, 9)
⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
Rank 2 │ layer1.weight        │ all_gather      │ Shard: [░░░░░░░░░░░░██████] -> [██████████████████] │ Shape: (4, 9) -> (12, 9)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Rank 0 │ layer1.bias          │ all_gather      │ Shard: [██████░░░░░░░░░░░░] -> [██████████████████] │ Shape: (4,) -> (12,)
⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
Rank 1 │ layer1.bias          │ all_gather      │ Shard: [░░░░░░██████░░░░░░] -> [██████████████████] │ Shape: (4,) -> (12,)
⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
Rank 2 │ layer1.bias          │ all_gather      │ Shard: [░░░░░░░░░░░░██████] -> [██████████████████] │ Shape: (4,) -> (12,)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Rank 0 │ layer1.weight        │ free            │ Shard: [██████████████████] -> [██████░░░░░░░░░░░░] │ Shape: (12, 9) -> (4, 9)
⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
Rank 1 │ layer1.weight        │ free            │ Shard: [██████████████████] -> [░░░░░░██████░░░░░░] │ Shape: (12, 9) -> (4, 9)
⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
Rank 2 │ layer1.weight        │ free            │ Shard: [██████████████████] -> [░░░░░░░░░░░░██████] │ Shape: (12, 9) -> (4, 9)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Rank 0 │ layer1.bias          │ free            │ Shard: [██████████████████] -> [██████░░░░░░░░░░░░] │ Shape: (12,) -> (4,)
⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
Rank 1 │ layer1.bias          │ free            │ Shard: [██████████████████] -> [░░░░░░██████░░░░░░] │ Shape: (12,) -> (4,)
⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
Rank 2 │ layer1.bias          │ free            │ Shard: [██████████████████] -> [░░░░░░░░░░░░██████] │ Shape: (12,) -> (4,)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

...
```

## 3. Running the Test Suite

The repository includes a comprehensive automated test suite combining **unit tests** (isolating individual components like parameter wrapper, hooks, logging, and state dict managers) and **integration tests** (validating end-to-end distributed lifecycles, DTensor placements, optimizer state sharding, and loss convergence).

Thanks to a custom pytest hook in `conftest.py`, tests marked with `@pytest.mark.multi_rank` automatically spawn isolated CPU worker processes via `torch.multiprocessing.spawn`. In other words, you can execute the entire test suite using standard pytest invocations:

```bash
pytest
```

### Key Verification Focus

The verification suite (`test_torch_parity.py`) asserts that `edufsdp` matches PyTorch's native distributed primitives down to tight mathematical tolerances ($\le 10^{-5}$):

* **Sharding Placements:** Validates tensor sharding layouts for weights and optimizer states against native implementations.
* **Gradient Reduction Match:** Verifies that Reduce-Scatter collective math produces identical gradients.
* **Parameter Updates & Loss Convergence:** Asserts that iterative training steps yield identical loss curves and weight updates compared to PyTorch's native `fully_shard`.
* **State Dict Restoration:** Verifies Local and Full checkpoint export/import fidelity.

<br>

## 3. VS Code Debugging Configurations

The repository includes a pre-configured `.vscode/launch.json` file designed to streamline debugging. <br>
It provides out-of-the-box configurations for launching single files as well as running multi-process distributed training via 'torch.distributed.run' (torchrun) across multiple ranks with pre-set environment variables ('PYTHONPATH' and 'GLOO_SOCKET_IFNAME'), enabling seamless breakpoints across spawned worker processes directly inside VS Code.

<br>

## 4. Wrapping Custom Modules

Wrapping models in `edufsdp` is designed to be as simple and familiar as PyTorch's native FSDP interface.

### Basic Wrapping Example

```python
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from edufsdp.fsdp import FSDP

# 1. Initialize a 1D CPU Device Mesh across available ranks
device_mesh = init_device_mesh("cpu", mesh_shape=(2,), mesh_dim_names=("cpu",))

# 2. Define a standard PyTorch model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleMLP()

# 3. Wrap the model in FSDP (as simple as native PyTorch)
wrapped_model = FSDP(model, device_mesh=device_mesh)
```

### Best Practices for Wrapping

* **Leaf vs. Container Modules:** `edufsdp` inspects parameters recursively. Ensure your parameters reside in standard leaf layers (like `nn.Linear` or `nn.Conv2d`) so that hooks bind correctly to parameter ownership boundaries.
* **In-Place Operations:** Avoid in-place tensor mutations (e.g., `x.add_()`) inside custom forward passes, as they can interfere with autograd gradient graphs and tensor sharding hooks.

<br>

## 5. Multi-Rank Debugging & Troubleshooting

Running distributed training across multiple CPU processes via the Gloo backend can occasionally present synchronization hurdles. Here are common issues and solutions:

### 1. Communication Hangs / Deadlocks

* **Symptom:** The script freezes indefinitely during forward or backward passes.
* **Cause:** Rank desynchronization. If one rank enters a communication collective (like an All-Gather hook) while another rank diverges or skips a layer, the Gloo backend blocks waiting for the missing process.
* **Solution:** Ensure that all ranks execute identical control flow paths. Avoid conditional layer execution based on rank ID unless explicitly coordinated.

### 2. Interface Binding Issues on Linux/macOS

* **Symptom:** `RuntimeError: Gloo initialization failed` or socket timeout errors on startup.
* **Cause:** PyTorch defaults to guessing the network interface, which can fail on machines with multiple active loopback or virtual network interfaces.
* **Solution:** Explicitly set the network interface environment variable before launching your script:

```bash
export GLOO_SOCKET_IFNAME=lo0  # Or lo on Linux
```

### 3. Missing Gradients (`NoneType`)

* **Symptom:** `AttributeError: 'NoneType' object has no attribute 'to_local'` during the optimizer step.
* **Cause:** The backward pass was executed on a tensor that did not require gradients, or `backward()` was not called on the loss output.
* **Solution:** Ensure `requires_grad=True` is set on input tensors and that `loss.backward()` is called before invoking `optimizer.step()`.
