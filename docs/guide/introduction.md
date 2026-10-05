# Introduction to EvoMO

## Multiobjective optimization: inputs and outputs

A multiobjective problem optimizes several objectives that may conflict. For example,
faster robot motion may require more energy. An evolutionary algorithm maintains a population
of candidate solutions and updates it through evaluation, selection, crossover, or mutation.

Under minimization, solution A dominates B if A is no worse in every objective and strictly
better in at least one. Points that are not dominated by other points in the current set form
the first non-dominated front. This does not establish that they lie on the problem's true
Pareto front; reference fronts, metrics, and repeated runs help assess solution quality.

EvoMO uses the following batched tensor conventions:

| Data | Shape | Meaning |
| --- | --- | --- |
| `population` / `pop` | `(N, D)` | N candidate solutions with D decision variables each |
| `fitness` / `fit` | `(N, M)` | M objective values per candidate |
| `lb`, `ub` | `(D,)` | Lower and upper bounds for each decision variable |
| `cv` | `(N, C)` | Nonnegative violations for C constraints; some operators also accept total violations of shape `(N,)` |
| `rank` | `(N,)` | Non-domination ranks starting at 0 |

Keep input tensors, algorithm state, problems, and reference fronts on consistent devices
with compatible dtypes.

## Relationship with EvoX

EvoMO uses EvoX's `Algorithm`, `Problem`, `Workflow`, and state management abstractions.
EvoX supplies general operators, monitors, and neural network parameter adapters.
EvoMO provides its own multiobjective algorithms, benchmarks, selection operators, metrics,
and `UnifiedWorkflow`, which preserves constraint violations.

```python
from evomo.algorithms import NSGA2
from evomo.problems.numerical import DTLZ2
from evomo.workflows import UnifiedWorkflow

from evox.workflows import EvalMonitor
```

`evomo.algorithms.NSGA2` and `evox.algorithms.NSGA2` are different implementations.
Record the actual import path when reproducing an experiment.

## Parts of a run

1. **Problem:** implements batched `evaluate`, returning objectives and, for constrained problems, violations.
2. **Algorithm:** stores the population and objectives, then generates and selects the next population.
3. **Workflow:** connects the algorithm and problem, handling objective directions, transforms, and monitor callbacks.
4. **Analysis:** extracts the final population or monitor history, filters feasible non-dominated solutions, and computes metrics.

Call `workflow.init_step()` to initialize a run, then call `workflow.step()` repeatedly.
The examples assume that algorithm evaluation is connected through a workflow.

## Version scope

This documentation describes the current PyTorch implementation. The documentation build reads
the version from `pyproject.toml`. The historical JAX implementation is on the `v0.0.1-dev`
branch and pairs with EvoX 0.9.0; its state and invocation conventions do not apply to these examples.
