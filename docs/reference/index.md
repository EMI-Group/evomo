# API reference

These pages are generated from the current source, including constructor signatures, methods,
and existing docstrings. Generated API documentation is shared in English by both language
guides. All 57 public algorithm constructors include parameter descriptions, types, defaults,
and implementation notes. Prefer public module exports for imports; internal helpers are not stable interfaces.

```{toctree}
:maxdepth: 1

parameters
citation
```

```{toctree}
:maxdepth: 2

Algorithms <../autoapi/evomo/algorithms/index>
Numerical problems <../autoapi/evomo/problems/numerical/index>
Constrained problems <../autoapi/evomo/problems/constrained/index>
Neuroevolution <../autoapi/evomo/problems/neuroevolution/index>
Workflows <../autoapi/evomo/workflows/index>
Selection operators <../autoapi/evomo/operators/selection/index>
Metrics <../autoapi/evomo/metrics/index>
Utilities <../autoapi/evomo/utils/index>
```

```{toctree}
:hidden:

../autoapi/evomo/problems/index
../autoapi/evomo/operators/index
```

## Selection operator overview

Import common operators from `evomo.operators.selection`. Objectives use minimization.

| Interface | Input | Output and purpose |
| --- | --- | --- |
| `non_dominate_rank(x, cv=None)` | `(N, M)` objectives and optional violations | Complete zero-based ranks of shape `(N,)` |
| `crowding_distance(costs, mask)` | Objectives and a boolean `(N,)` mask, or `None` | Crowding distances of shape `(N,)` |
| `nd_environmental_selection(x, f, topk, cv=None)` | Population, objectives, survivor count, and optional violations | `(selected_pop, selected_fit, selected_rank, selected_distance, selected_cv)` |
| `ref_vec_guided(x, f, v, theta)` | Population, objectives, reference vectors, and angle penalty parameter | `(next_pop, next_fit)`; unassociated slots may contain NaN |
| `get_non_dominate_backend(backend="torch")` | `"torch"` or `"triton"` | Backend module; does not change algorithms automatically |

For unconstrained environmental selection, the fifth return value is `None`. `topk` must be
within the valid population size. Environmental selection ranks only through the cutoff
front, but ranks returned for selected individuals are complete. Use `non_dominate_rank` when
you need ranks for every input point.

Backend modules also expose `dominate_relation(x, y, cv_x=None, cv_y=None)`, which returns a
dominance matrix between two sets of objectives. This function is not a top-level public
export of `evomo.operators.selection`; access it through the backend module.

## Evaluation and state helpers

- `parse_evaluate(eval_out)` returns `(fitness, cv)`; pure objective tensors give `cv=None`.
- `at_least_2d` handles single and batched inputs; see its generated return contract.
- `get_pareto_front` and `unique_rows_sorted` filter and deduplicate results; consider dynamic output sizes when compiling.
- `load_pareto_front_from_file` loads packaged constrained-problem reference fronts.
- `register_lazy_buffer` supports state whose shape is known only after the first evaluation.

See the [workflow guide](../guide/workflows.md) and [constraint guide](../guide/constraints.md)
for their corresponding semantics.
