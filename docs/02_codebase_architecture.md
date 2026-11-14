<p><h1 align="center">Codebase Architecture & Component Internals</h1></p>

<br>

The *edufsdp* codebase is structured to maintain a direct, transparent correspondence between theoretical distributed principles and actual implementation details. Rather than relying on complex production wrappers or heavy C++ extensions, the architecture uses standard Python and PyTorch primitives to expose the mechanics of parameter sharding, lifecycle hooks, and state management.

The core logic is partitioned into four primary modules:

* **sharded_parameter.py** — Maps the FSDP v2 parameter-storage decoupling model. It manages rank-local weight shards alongside a resizable global buffer, handling state transitions between sharded and unsharded representations.
* **hooks.py** — Implements the just-in-time communication primitives required by FSDP v2 (All-Gather during forward/backward pre-hooks and Reduce-Scatter via custom autograd functions), orchestrating lifecycle timing.
* **fsdp.py** — Provides the composable block-level wrapping API (`fully_shard`), mapping the FSDP v2 model-parallel design where arbitrary submodules can be wrapped independently.
* **state_dict.py** — Implements distributed checkpointing and state dictionary conversion, supporting both rank-local sharded states and global un-sharded exports.

<br>

## 1. Explicit Parameter Management

A wrapped FSDP model acts as a transparent drop-in replacement for a standard PyTorch model. To any external caller or standard training loop, the module looks and behaves exactly like a conventional nn.Module. All complex sharding, communication, and memory management logic is completely encapsulated within 'ShardedParameter' (`sharded_parameter.py`) and its associated hooks, allowing standard forward passes and optimizer updates to run unmodified.

* **Distributed Parameter Wrappers:** 'ShardedParameter' encapsulates the rank-local weight shard while maintaining awareness of the global tensor shape and sharding topology through native PyTorch DTensor abstractions.
* **Persistent Buffer & Storage Management:** PyTorch's autograd engine maintains strong references to tensors used during the forward pass, preventing standard garbage collection from freeing memory when unlinking attributes. To solve this, 'ShardedParameter' maintains a persistent buffer created at full global size to satisfy autograd's structural requirements. Storage is dynamically managed via low-level resizing: when sharded, the buffer storage is resized to zero ('resize_(0)') to instantly reclaim physical GPU memory while keeping the Python tensor shell and metadata intact. Conversely, right before an All-Gather communication step, the buffer storage is resized back up to its full global dimensions to accommodate incoming full-size weights.
* **State Manipulation Methods:** 'ShardedParameter' exposes two key state manipulation methods that control whether a module sees its local shard or the fully gathered weight:
  * *set_unsharded(full_tensor)*: Mounts the gathered full parameter tensor directly onto the module's attribute dictionary (module.\_\_dict\_\_), allowing normal forward/backward matrix multiplication.
  * *set_sharded()*: Restores the attribute pointer back to the sharded DTensor and clears the buffer storage back to zero size, freeing memory immediately.

<br>

## 2. Custom Autograd & Lifecycle Hooks

Distributed training requires precise synchronization timed directly to the execution of the forward and backward passes. The `hooks.py` module manages this timing through a native autograd integration and discrete hook classes.

### Custom Autograd Function

`FSDPAllGatherFunction` inherits from `torch.autograd.Function`, integrating distributed communication natively into PyTorch's autograd graph:

* **Forward Pass:** Because each rank only holds a 1/N local weight shard, a standard forward matrix multiplication cannot execute. To solve this, an All-Gather collective (`dist.all_gather_single`) is executed across all ranks using the Gloo backend. Each rank contributes its local shard, and all ranks coordinate to reconstruct the complete, global weight matrix inside a pre-allocated buffer. This temporarily restores the full parameter tensor so the module can compute activations for its local input batch.
* **Backward Pass:** Since full-size weight matrices were used during the forward pass, every rank computes a full-size gradient tensor with respect to its local input batch. These full gradients must first be globally reduced (summed) across all ranks so that every rank shares identical global gradient information. Then, because each rank only owns a rank-local weight shard, a `dist.reduce_scatter_single` operation is executed to scatter and shard the gradient, leaving each rank with only the gradient slice corresponding to its weight shard, scaled by the world size ($N$).

### Lifecycle Hooks

Four discrete hook classes manage parameter lifecycles by leveraging the parameter state methods:

* **Forward Pre-Hook**: Intercepts module execution before the forward pass begins, triggering the All-Gather collective via 'FSDPAllGatherFunction.apply()' to temporarily reconstruct the full parameter tensor from local shards.
* **Forward Post-Hook**: Executes immediately after the forward computation finishes, discarding the gathered full parameter tensor to reclaim peak device memory.
* **Backward Pre-Hook**: Coordinates the secondary All-Gather required during the backward pass to restore parameter weights before gradient calculation.
* **Backward Post-Hook**: Executes immediately after the backward pass computation finishes, specifically responsible for dropping and cleaning up the temporary all-gathered weights.

<br>

## 3. FSDP Orchestration & 'fully_shard' API

A composable block-level API (`fsdp.py`) centered around the `fully_shard` function controls the granularity of module wrapping across the architecture. Grouping multiple layers into a single wrapped block reduces the number of individual all-gather kernel launches and communication latency at the cost of increased peak memory consumption:

* **Composable Block Wrapping:** The user explicitly calls `fully_shard(submodule, device_mesh)` on individual layers or blocks (such as linear layers or transformer blocks).
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
