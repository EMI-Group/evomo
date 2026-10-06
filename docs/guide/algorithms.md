# Algorithms

Import algorithms from `evomo.algorithms`. The names below match the current public exports.
The [API reference](../reference/index.md) is generated from source and provides constructor
signatures, defaults, methods, and existing paper references.

## Common starting points

| Class | Main mechanism | Details to check |
| --- | --- | --- |
| `NSGA2` | Non-dominated sorting and crowding distance | A starting point for the numerical and constrained examples |
| `NSGA3` | Non-dominated sorting and reference points | Reference point count relative to objectives and population size |
| `MOEAD` | Decomposition and neighborhood updates | Aggregation and neighborhood parameters |
| `TensorMOEAD` | Tensorized decomposition and neighborhood updates | `aggregate_op`, actual population size, and neighborhood size |
| `RVEA`, `RVEAa` | Reference vector guided selection | Set `max_gen` consistently with the iteration budget |
| `IBEA` | Indicator-based selection | Indicator choice and objective scaling |
| `HypE` | Selection based on hypervolume estimation | Sampling and reference-related parameters |
| `LMOCSO` | Competitive swarm optimization for large-scale problems | Decision dimension and algorithm-specific parameters |

This table describes interfaces, not an experimental ranking. Compare algorithms using equal
evaluation budgets and multiple random seeds.

## Constructor parameters and state

Algorithms accept `pop_size`, `n_objs`, `lb`, and `ub`; some also expose `device`. Bounds must be
one-dimensional tensors with matching shapes, devices, and dtypes. `n_objs` must match the
number of columns returned by the problem's `evaluate`. See the
[parameter guide](../reference/parameters.md) for device rules, operator contracts, and algorithm-specific settings.

Common state attributes are `pop` and `fit`. Algorithms may also maintain reference vectors,
ideal points, archives, velocities, ranks, or violations. Attributes specific to one algorithm
are not a universal interface.

Reference vector or weight sampling may adjust `pop_size`. Read
`workflow.algorithm.pop.shape[0]` for the actual population size after construction.
`TensorMOEAD` requires the sampled population size to be greater than 10.

For example, replace the quickstart's algorithm construction with:

```python
from evomo.algorithms import TensorMOEAD

algorithm = TensorMOEAD(
    pop_size=100,
    n_objs=problem.m,
    lb=torch.zeros(problem.d, device=device),
    ub=torch.ones(problem.d, device=device),
    aggregate_op=("pbi", "pbi"),
    device=device,
)
```

`TensorMOEAD` accepts the aggregation names `pbi`, `tchebycheff`, `tchebycheff_norm`,
`modified_tchebycheff`, and `weighted_sum`.

## Replacing operators

Some algorithms accept `selection_op`, `mutation_op`, and `crossover_op`. Their calling
contracts are defined by each algorithm. Check its `step` implementation for input shapes,
output shapes, and semantics before substituting an operator.

For NSGA-II, `selection_op` chooses mating parents and defaults to EvoX's
`tournament_selection_multifit`. It does not perform environmental selection or select a
non-dominated sorting backend. See the [backend guide](../non_dominate_backends.md).

## Other public algorithms

The following table preserves the exact exported class names:

| Group | Exported names |
| --- | --- |
| MOEA/D and related implementations | `BCEMOEAD`, `MOEADAWA`, `MOEADDE`, `MOEADDU`, `MOEADDYTS`, `MOEADFRRMAB`, `MOEAURAW`, `MOEAD_DCWV`, `MOEAD_DRA`, `MOEAD_PaS` |
| Sparse and large-scale implementations | `LSMOF`, `SparseEA`, `SparseEA2`, `TSSparseEA`, `WOF` |
| Other implementations (1) | `AGEMOEA`, `BCE_IBEA`, `BiGE`, `CLIA`, `CMOEA_MS`, `CMOPSO`, `CoMMEA`, `DMMOEA`, `EFRRR`, `GDE3`, `GrEA`, `GWASFGA`, `KnEA` |
| Other implementations (2) | `MaOEACSS`, `NSBiDiCo`, `NSGAII_SDR`, `OSP_NSDE`, `PESA2`, `PICEAg`, `PREA`, `SIBEA`, `SMPSO`, `SNSGA2`, `SPEAR`, `SSCEA` |
| Other implementations (3) | `TELSO`, `TSNSGAII`, `ThetaDEA`, `Two_Arch2`, `VaEA`, `WASFGA`, `eMOEA`, `tDEA_CPBI` |

An algorithm name containing a constraint-related term does not establish compatibility with
arbitrary constrained problems. Start with the documented NSGA-II constraint workflow and
verify whether other implementations use violations in selection.
