# Quickstart: NSGA-II on DTLZ2

This example minimizes the three objectives of DTLZ2 in a 12-dimensional decision space,
with all variables bounded by `[0, 1]`. It chooses CUDA when available and otherwise uses CPU,
sets a random seed, initializes the workflow, and performs 50 iterations. It then extracts
non-dominated solutions from the final population and computes IGD.

## Complete example

```{literalinclude} ../examples/quickstart.py
:language: python
```

After installing the source checkout, run this command from the repository root:

```sh
python docs/examples/quickstart.py
```

## Understanding the result

`workflow.algorithm.pop` stores decision variables and `workflow.algorithm.fit` stores the
corresponding objective values. Rows of these tensors refer to the same candidate solutions.
Algorithms use minimization internally. Since this example does not change objective directions,
these values are also DTLZ2's original objectives.

`non_dominate_rank(fitness) == 0` extracts the first non-dominated front of the current set.
Filtering happens after optimization and is kept outside the compiled `step`.
`problem.pf()` supplies reference front samples. Lower IGD means the reference points are
closer on average to the resulting solution set. Hardware, versions, and random operations
may change the numerical result, so the example does not prescribe a fixed score.

The 50 iterations illustrate the interface; they do not establish convergence or algorithm performance.

## Next steps

- To change algorithms, check [algorithm parameters](algorithms.md) and the actual population size.
- To change problems, check [benchmarks](problems.md) and update dimensions, objective count, and valid bounds.
- For feasibility handling, follow the [constrained optimization example](constraints.md).
- Before compiling the workflow, read [compilation and performance](performance.md).
