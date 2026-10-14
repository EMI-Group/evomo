# Devices, compilation, and performance

## Set the device and dtype

Set the default device before creating algorithms, problems, and tensors, or explicitly pass
the same device to the relevant constructors:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
```

The default device affects subsequent allocations without an explicit device. Control the
scope of this global setting in an existing application. A `device` argument usually controls
placement; it does not mean an algorithm automatically preserves every input dtype.
Check internal initialization for each algorithm. With a nondefault dtype, inspect the problem,
bounds, state, and result dtypes. If needed, set the default floating-point dtype before
construction and restore it after use.

## Compile a workflow

Validate eager execution first, then compile `step` after initialization:

```python
workflow.init_step()
compiled_step = torch.compile(workflow.step, fullgraph=True)
for _ in range(100):
    compiled_step()
```

EvoX also provides `evox.core.compile`. Some algorithm tests use
`compile(workflow.step, dynamic=False)`. Choose a method validated for your algorithm and
runtime environment. Fixed population size, decision dimension, objective count, and return
structure help reduce recompilation.

The first invocation includes graph capture and compilation. Successful `fullgraph=True`
execution establishes full graph capture for that call, not a single GPU kernel, CUDA Graph
support, or `vmap` compatibility. Some algorithm, operator backend, and compiler combinations
may not support full graph capture; verify them on your system.

## Analyze results outside the loop

Keep dynamic boolean filtering, `.item()`, printing, NumPy conversion, and file writing outside
the iteration loop. Reading Python scalars in GPU hot paths introduces synchronization.
Data-dependent branches and dynamic shapes can also affect compilation. Prefer fixed-shape
masks, `torch.where`, and batched operations in custom problems.

## Timing a run

The following assumes that the workflow is initialized and `step_function` is an eager or
compiled step. Warmup advances algorithm state; to compare equivalent starting points,
construct separate workflows or restore equivalent states.

```python
import time

for _ in range(5):
    step_function()

cuda = workflow.algorithm.pop.device.type == "cuda"
if cuda:
    torch.cuda.synchronize(workflow.algorithm.pop.device)
start = time.perf_counter()
for _ in range(50):
    step_function()
if cuda:
    torch.cuda.synchronize(workflow.algorithm.pop.device)
seconds_per_step = (time.perf_counter() - start) / 50
print(seconds_per_step)
```

Repeat measurements and report the device, PyTorch/EvoX versions, dtype, actual population
size, objective count, problem, compilation options, warmup count, and synchronization method.
This timing includes problem evaluation, algorithm updates, and enabled monitor overhead.
Report speed and solution quality separately. Quality comparisons require multiple seeds,
equal evaluation budgets, and consistent metric definitions.

## Operator backends and vectorization

Non-dominated sorting defaults to PyTorch. See the [backend guide](../non_dominate_backends.md)
for optional Triton requirements and limitations. Selecting a standalone backend does not
replace operators inside algorithms automatically. The Triton backend has no `vmap` rules.
For other algorithms, verify the specific state and operator combination before using `vmap`;
a successful compiled single run does not establish support for batched independent experiments.
