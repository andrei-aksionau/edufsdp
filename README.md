
<p>
    <h4 align="center"><i>Welcome to</i></h4>
    <h1 align="center">Educational Fully Sharded Data Parallel</h2>
    <h4 align="center">(edufsdp)</h4>
</p>

<br>

<p align="center"><i>A transparent, ground-up implementation of Fully Sharded Data Parallel (FSDP) built for learning, low-level inspection, and running on CPU via the Gloo backend.</i></p>

<br>

## Overview

While PyTorch's native FullyShardedDataParallel (FSDP) is powerful and production-ready, its codebase is designed for massive scale, making it hard to peel back the layers and understand how things work step by step.

***edufsdp*** is an educational reference implementation that makes parameter sharding, autograd hooks, and collective communication completely transparent. Built to run on CPUs using PyTorch's Gloo backend, it matches the mathematical behavior of production FSDP down to $\le 10^{-5}$ precision, without the heavy infrastructure overhead.

<br>

## Key Features

**Explicit Parameter Container**
<br>
Wraps DTensor instances and seamlessly switch module state between local shards and gathered weights, while custom autograd functions handle exact All-Gather and Reduce-Scatter operations.

**Clear Forward & Backward Lifecycle Hooks**
<br>
• *Symmetric Autograd Integration* — Wraps collective communication in a custom torch.autograd.Function, using DTensor primitives to execute All-Gather (forward) and Reduce-Scatter (backward) within the computational graph.
<br>
• *Forward Pre/Post Hooks* — Reconstructs full weights right before layer execution and discards them immediately afterward to minimize the memory footprint.
<br>
• *Backward Pre/Post Hooks* — Re-gathers full weights right before gradient computation and discards them immediately afterward, keeping only sharded gradients.

**Flexible State Dict Management**
<br>
• *Local Mode* — Operates on rank-specific DTensor shards directly, bypassing network communication for maximum speed.
<br>
• *Full Mode* — Reconstructs unified global tensors across all ranks, ensuring full compatibility for single-device export.

**Mathematically Verified**
<br>
Backed by an automated verification suite that validates sharding placements of weights and optimizer states, match of gradient reduction, parameter updates, loss convergence and state dict restoration against PyTorch's native distributed primitives.

<br>

## Quickstart

1\. **Clone and Sync Dependencies**
<br>
Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to automatically manage the Python environment and sync dependencies specified in *pyproject.toml*:

```bash
git clone https://github.com/andrei-aksionau/edufsdp.git
cd edufsdp
uv sync
```

<br>

2\. **Run the Demonstration Script**
<br>
Execute the root-level inspection script to watch live weight sharding visualizations, hook executions, and state saving in action:

```bash
./run_inspect_fsdp.sh
```

<br>

## Documentation

⇨ **[Conceptual FSDP & Lifecycle](docs/01_fsdp_mechanics.md)** — Deep dive into the memory wall, parameter sharding lifecycles, and FSDP1 vs. FSDP2 mechanics.

⇨ **[Codebase Architecture](docs/02_codebase_architecture.md)** — Mapping theory to code (parameter container, custom autograd hooks, module wrapping, and state dict managers).

⇨ **[Setup & Debugging Guide](docs/03_installation_and_debugging.md)** — Guide on environment setup, wrapping arbitrary modules, multi-rank debugging, and deadlock troubleshooting.
