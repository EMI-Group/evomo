# Troubleshooting

## Import errors or the wrong environment

Check the interpreter and installed package locations:

```sh
python -c "import sys; print(sys.executable)"
python -m pip show torch evox evomo
```

Use the same interpreter to install dependencies, run examples, and run tests. The repository
uses a `src/` layout. Run `python -m pip install -e .` from the root before running examples,
rather than changing `sys.path` temporarily.

## CUDA unavailable or mismatched devices

If `torch.cuda.is_available()` is false, check the PyTorch build and driver, then follow the
official installation instructions. For device mismatch errors, inspect `.device` on bounds,
problem constants, algorithm state, inputs, and reference fronts. Set the default device
before constructing problems and algorithms. MoRobtrol also requires checking `jax.devices()`.

## Incorrect objective shapes

Check that `n_objs` matches `fitness.shape[1]` and input columns match the problem's decision
dimension. Reference-vector algorithms may adjust population size, so the requested `pop_size`
is not always the output row count.

## A constrained problem returns an unexpected tuple

Use `UnifiedWorkflow` and an algorithm that uses violations in selection. Do not bypass the
error by taking only `evaluate(...)[0]`, which changes the problem's meaning. Follow the
complete [constraint example](constraints.md).

## NaN or Inf objectives

Check bounds, division by zero, normalization denominators, invalid mathematical domains,
and state updates. Reference-vector selection may use NaN to mark slots without associated
candidates; tensor row count alone does not measure valid solutions. During analysis, filter
population and objective rows together. Filtering does not repair evaluation errors.
If every row is invalid, stop metric computation and investigate the cause.

## Slow compilation, compiler errors, or recompilation

Run eagerly first to distinguish algorithm errors from compiler or toolchain errors. Ensure
`init_step()` runs before compilation, and keep shapes and return structure stable during
iterations. The first invocation may include compilation costs; exclude it from steady-state
timing. Record complete exceptions and versions for system, toolchain, or operator failures,
and use eager execution to continue algorithm validation when needed.

## Unexpected or inconsistent metrics

Verify that inputs are objective values, GD/IGD inputs are nonempty, and constrained inputs
contain only feasible solutions. Check objective directions, normalization, reference front
sampling, HV reference points, sample counts, and seeds. The current `gd` divides a fixed
norm by point count; comparisons using different normalization definitions are not equivalent.

## Out of GPU memory

Inspect pairwise comparisons in non-dominated sorting, GD/IGD distance matrices, HV sample
comparisons, and full monitor histories. Reduce population size, reference points, or HV
samples, and disable unnecessary history recording. Record how these changes affect evaluation
budgets or metric precision.

## Long Windows paths

For path-related `FileNotFoundError` errors during compilation or dependency caching, inspect
the actual missing path. If the cause is the system's path length limit, use shorter checkout
or cache paths, or enable long path support. Diagnose other missing-file errors from their
actual paths.

## Reporting an issue

Provide a minimal reproducer, actual import paths, Python/PyTorch/EvoX versions, device, dtype,
population and objective counts, and the complete exception. State the outcomes of eager and
compiled execution separately. Submit reproducible reports to the
[EvoMO issue tracker](https://github.com/EMI-Group/evomo/issues).
