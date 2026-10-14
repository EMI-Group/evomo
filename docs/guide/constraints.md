# Constrained multiobjective optimization

## Violation conventions

`CMOP.evaluate(population)` returns `(fitness, cv)`. Objectives have shape `(N, M)` and
violations have shape `(N, n_iq + n_eq)`, with one column per constraint.

- Inequalities use `g(x) <= 0`; the violation is `max(g(x), 0)`.
- Equality violations are `max(abs(h(x)) - constr_eq_eps, 0)`; the default tolerance is `1e-4`.
- A solution is feasible when all violations are zero. Current selection compares total violations.

A negative raw inequality value means the constraint is satisfied; do not use it directly as
a violation. Currently, `CMOP.evaluate` always returns a pair, regardless of the constructor's
`return_cv` parameter.

## NSGA-II on DOC1

```{literalinclude} ../examples/constrained.py
:language: python
```

```sh
python docs/examples/constrained.py
```

NSGA-II preserves violations during initialization and selection in each generation. Feasible
solutions take priority over infeasible solutions. Infeasible solutions are first compared
by total violation. Objective dominance applies between feasible solutions or when total
violations are equal.

## Extracting a feasible front

First filter by violations and finite objective values, then extract rank-zero points from
the remaining objectives. A front computed from objectives alone is not a feasible front.
A small value for one objective also does not establish feasibility.

If no feasible solutions remain, report feasibility rates or violation statistics. GD and IGD
require nonempty inputs, so feasible-front distance metrics cannot be reported in that case.

## Using another algorithm

A constrained run requires all three conditions:

1. The problem returns both objectives and violations.
2. The workflow preserves both, for example through `UnifiedWorkflow`.
3. The algorithm uses violations in comparison, selection, and state updates.

The first two conditions are insufficient on their own. Some algorithms treat `evaluate` as
a pure objective tensor. Even implementations that unpack its return value need to be checked
for actual constraint-aware selection. This guide uses `evomo.algorithms.NSGA2`; it does not
establish constraint support for every algorithm.
