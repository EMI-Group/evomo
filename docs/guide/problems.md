# Benchmarks and custom problems

## Numerical benchmarks

Import concrete problem classes from `evomo.problems.numerical`:

| Family | Public classes | Parameters and bounds |
| --- | --- | --- |
| ZDT | `ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6` | Decision dimension is `n`; typically two objectives; ZDT4's trailing variables use `[-5, 5]` |
| DTLZ | `DTLZ1`–`DTLZ7` | `d`, `m`, and `ref_num`; standard variable bounds are `[0, 1]` |
| WFG | `WFG1`–`WFG9` | `d`, `m`, `k`, and `ref_num`; variable i uses `[0, 2i]`, with i starting at 1 |
| MaF | `MAF1`–`MAF15` | Parameters include `d` and `m`; check dimension and bound conventions for each variant |
| LSMOP | `LSMOP1`–`LSMOP9` | Specify `d` and `m`; the first `m-1` variables typically use `[0, 1]`, and trailing variables use `[0, 10]` |

Numerical problems generally return an objective tensor of shape `(N, M)`. Unlike constrained
problems, not every numerical class exposes `lb` and `ub`. Construct bounds from the problem
definition when needed.

`pf()` supplies a reference front or samples of it for analysis. `ref_num` is not always an
exact output count or strict upper bound; inspect the returned shape. Some reference fronts
use numerical samples or packaged data, so not every `pf()` result is an exact analytical front.

## Constrained benchmarks

`evomo.problems.constrained` exports:

| Family | Variants |
| --- | --- |
| LIRCMOP | `LIRCMOP1`–`LIRCMOP14` |
| DASCMOP | `DASCMOP1`–`DASCMOP9` |
| CTP | `CTP1`–`CTP8` |
| MW | `MW1`–`MW14` |
| FCP | `FCP1`–`FCP5` |
| DOC | `DOC1`–`DOC9` |
| SDC | `SDC1`–`SDC15` |
| LSCM | `LSCM1`–`LSCM12` |

These classes use `evomo.problems.constrained.base.CMOP` and expose decision dimension `d`,
objective count `m`, and bounds `lb` and `ub`. Some variants have fixed dimensions; passing
`d` or `m` does not universally change a problem's definition. `evaluate` returns `(fitness, cv)`.
See [constrained optimization](constraints.md) for a complete example.

## A custom batched problem

Subclass EvoX's `Problem` and implement `evaluate` for an entire population at once.
This example minimizes squared distances to two centers:

```{literalinclude} ../examples/custom_problem.py
:language: python
```

```sh
python docs/examples/custom_problem.py
```

The first output dimension must correspond to input candidates, and the second must match
the algorithm's `n_objs`. Derive computations from input tensors instead of converting to
NumPy or evaluating candidates one by one. Register constants as buffers during initialization,
or create them on the input's device with its dtype.

A custom constrained problem can return `(fitness, cv)`, where `cv` contains nonnegative
violations rather than unprocessed constraint functions. Connect it with `UnifiedWorkflow`
and an algorithm that uses violations in selection.
