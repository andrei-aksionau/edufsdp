<p><h1 align="center">Codebase Architecture & Component Internals</h1></p>

<br>

The *edufsdp* codebase is structured to maintain a direct, transparent correspondence between theoretical distributed principles and actual implementation details. Rather than relying on complex production wrappers or heavy C++ extensions, the architecture uses standard Python and PyTorch primitives to expose the mechanics of parameter sharding, lifecycle hooks, and state management.

The core logic is partitioned into four primary modules:

* **sharded_parameter.py** — Manages local weight slices, global shape abstractions, and state transitions.
* **hooks.py** — Implements the custom autograd communication function and lifecycle hooks.
* **fsdp.py** — Orchestrates composable block wrapping via `fully_shard` and hook registration.
* **state_dict.py** — Handles checkpoint conversion between local shards and full global tensors.

<br>

## 1. Explicit Parameter Management

The foundational building block of the library is the 'ShardedParameter' class (`sharded_parameter.py`). In a sharded environment, each rank only owns a fraction of any given parameter tensor.

* **Local vs. Global Representation:** 'ShardedParameter' wraps the local weight slice residing on the current rank while keeping track of the tensor's global shape and sharding dimension via PyTorch's native DTensor abstractions.
* **Lazy Buffer Pattern & Storage Management:** Instead of dynamic allocations, 'ShardedParameter' maintains an internal buffer matching the global parameter shape. It encapsulates all resizing logic, hiding it from hooks. By utilizing buffer resizing down to zero, it ensures physical memory is immediately reclaimed when sharded while keeping the Python tensor shell (with all the metadata) intact for autograd strong-reference compatibility.
* **State Manipulation Methods:** 'ShardedParameter' exposes two key state manipulation methods that control whether a module sees its local shard or the fully gathered weight:
  * *set_unsharded(full_tensor)*: Mounts the gathered full parameter tensor directly onto the module's attribute dictionary (module.\_\_dict\_\_), allowing normal forward/backward matrix multiplication.
  * *set_sharded()*: Restores the attribute pointer back to the sharded DTensor and clears the buffer storage back to zero size, freeing memory immediately.

<br>

## 2. Custom Autograd & Lifecycle Hooks

Distributed training requires precise synchronization timed directly to the execution of the forward and backward passes. The `hooks.py` module manages this timing through a native autograd integration and discrete hook classes.

### Custom Autograd Function

`FSDPAllGatherFunction` inherits from `torch.autograd.Function`, integrating distributed communication natively into PyTorch's autograd graph:

* **Forward Pass:** Takes the local shard and runs `dist.all_gather_single` to collect from all ranks the full size tensor using the Gloo backend and copy it into the pre-allocated `_unsharded_buffer`.
* **Backward Pass:** Intercepts incoming global gradients, runs `dist.reduce_scatter_single` to aggregate and shard gradients across the device mesh. It then normalizes the sharded gradient by dividing by the world size ($N$), completing the Reduce-Scatter operation.

### Lifecycle Hooks

Four discrete hook classes manage parameter lifecycles by leveraging the parameter state methods:

* **Forward Pre-Hook**: Intercepts module execution before the forward pass begins, triggering the All-Gather collective via 'FSDPAllGatherFunction.apply()' to temporarily reconstruct the full parameter tensor from local shards.
* **Forward Post-Hook**: Executes immediately after the forward computation finishes, discarding the gathered full parameter tensor to reclaim peak device memory.
* **Backward Pre-Hook**: Coordinates the secondary All-Gather required during the backward pass to restore parameter weights before gradient calculation.
* **Backward Post-Hook**: Executes immediately after the backward pass computation finishes, specifically responsible for dropping and cleaning up the temporary all-gathered weights.

<br>

## 3. FSDP Orchestration & 'fully_shard' API

A composable block-level API (`fsdp.py`) centered around the `fully_shard` function controls the granularity of module wrapping across the architecture. Grouping multiple layers into a single wrapped block reduces the number of individual all-gather kernel launches and communication latency at the cost of increased peak memory consumption:

* **Composable Block Wrapping:** The user explicitly calls `fully_shard(submodule, device_mesh)` on individual layers or blocks (such as transformer blocks or linear layers).
* **Parameter Conversion:** Standard 'torch.nn.Parameter' instances are converted into managed 'ShardedParameter' objects.
* **Local Hook Binding:** Pre- and post-forward/backward hooks from `hooks.py` are attached directly to the specified submodule, ensuring that parameter gathering and gradient reduction occur strictly around that module's compute scope.

<br>

## 4. State Dictionary Manager

Saving and loading checkpoints in a distributed sharded environment requires handling two distinct operational modes, managed by StateDictManager (`state_dict.py`):

* **Local Mode** (*StateDictType.Local*)
  * Get State Dict: Extracts rank-specific DTensor shards directly from each process with zero inter-process communication.
  * Load State Dict: Copies rank-local tensor payloads directly into 'parameter.local_shard'.
* **Full Mode** (*StateDictType.Full*)
  * Get State Dict: Automatically gathers full un-sharded parameter tensors across ranks using FSDPAllGatherFunction, returning a unified global state dictionary suitable for single-device inference or export.
  * Load State Dict: Takes a global un-sharded checkpoint, distributes tensors across ranks using distribute_tensor with 'Shard(dim=0)', and loads local shards accordingly.
