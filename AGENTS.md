# Agent guide

## Project overview

EvoMO is a PyTorch library for tensorized, GPU-accelerated multiobjective evolutionary
optimization, built on EvoX. The current codebase uses PyTorch; the historical JAX
implementation lives on a separate branch.

## Repository map

- `src/evomo/algorithms/`: optimization algorithms and their public exports.
- `src/evomo/operators/`: reusable evolutionary operators and selection utilities.
- `src/evomo/problems/`: numerical, constrained, and neuroevolution benchmarks.
- `src/evomo/workflows/`: integration between algorithms, problems, and monitors.
- `src/evomo/metrics/`: solution-quality metrics.
- `unit_test/`: algorithm, operator, and problem tests.
- `benchmarks/` and `experiments/`: benchmarking and experimental work.
- `docs/`: supporting documentation.
- `pyproject.toml`: dependencies, packaging, and Ruff configuration.

## Environment and commands

Use Python 3.10 or newer and the intended project environment. Core dependencies are
PyTorch >= 2.6.0 and EvoX >= 1.2.1. Check the existing environment before installing
dependencies; preserve its CPU/CUDA configuration.

For an editable installation and basic development tools:

```sh
python -m pip install -e .
python -m pip install pytest ruff
```

Install optional visualization or neuroevolution dependencies only when needed.
The `test` extra does not itself install pytest or Ruff.

Run relevant tests from the repository root, for example:

```sh
python -m pytest unit_test/problems/test_dtlz.py -q
python -m pytest unit_test/operators/test_non_dominate.py -q
python -m pytest unit_test/algorithms/test_moea.py -q
```

Check changed Python files with `python -m ruff check <paths>` and
`python -m ruff format --check <paths>`. Follow the existing style: 128-character
line length and LF line endings. Avoid repository-wide formatting for a scoped fix.

## Implementation guidance

- Follow existing EvoX `Algorithm` and `Mutable` state patterns. Preserve public
  interfaces and exports unless the task requires changing them.
- Preserve tensor device and dtype. Avoid accidental CPU allocations or host/device
  transfers inside GPU execution paths.
- Prefer batched tensor operations. In compiled hot paths, avoid unnecessary
  `.item()`, tensor-dependent Python branches, and dynamic-size indexing; use
  fixed-shape masks and scatter/gather operations where appropriate.
- Initialize a workflow with `init_step()` before compiling its `step()`.
  `torch.compile(fullgraph=True)` is useful for checking graph capture, but does not
  prove single-kernel execution, CUDA graph support, or `vmap` compatibility.
- Preserve algorithm semantics while tensorizing. Non-dominated ranks are zero-based;
  do not replace rank values with a selection mask.
- Constrained problems may return `(fitness, constraint_violation)`. Preserve this
  information through supported workflows and selection; do not silently drop it.
- Keep optional dependencies optional. Read `docs/non_dominate_backends.md` before
  changing non-dominated sorting backends; PyTorch is the default backend.

## Validation and experiments

- Run targeted checks appropriate to the change. For algorithm changes, cover
  relevant edge cases such as duplicate points, constant objectives, degenerate
  normalization, and constraint handling. Preserve existing tests.
- Test eager and compiled execution when changing compilation-sensitive code.
  Report unsupported environments or compiler/toolchain failures separately from
  algorithm failures; do not claim untested device or compilation support.
- For performance comparisons, separate compilation and warmup from steady-state
  timing, synchronize CUDA around timed regions, and repeat measurements. Report
  device, software versions, population size, objective count, and timing method.
- Compare stochastic solution quality across multiple seeds with equal evaluation
  budgets and consistent metric definitions. Speed alone does not establish quality,
  and compiled/eager runs need not be bitwise identical.
- Keep temporary scripts and generated data isolated from production code. Avoid
  committing benchmark outputs, local environments, or unrelated scratch files.
  Remove only task-owned temporary artifacts unless broader cleanup is requested.
- Keep changes scoped and preserve unrelated user edits. Summarize what changed,
  which checks ran, and any remaining limitations. For documentation-only changes,
  check the diff and referenced paths; runtime tests are normally unnecessary.
