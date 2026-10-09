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

`IBEA` and `HypE` accept `(fitness, cv)` through `UnifiedWorkflow` and retain selected
violations in `cv`. These are feasibility-priority extensions of the existing tensorized
algorithms; PlatEMO's base IBEA and HypE implementations do not consume constraints.
Mating first compares total positive violation, then indicator fitness or estimated
hypervolume contribution. Tied keys remain tied. IBEA removes the largest violations
first, resolving equal violations with its iterative indicator deletion. Constrained
IBEA also guards constant objective ranges and zero indicator scales.

HypE sorts constrained fronts: feasible solutions are ranked by objective dominance,
then infeasible solutions by violation, with equal violations sharing a front. It keeps
complete fronts and uses fixed-shape masked hypervolume estimates for batch selection
within the cutoff front. The existing unconstrained batch selection is preserved; this
does not reproduce PlatEMO's iterative hypervolume deletion.

`RVEAa` retains violations through first-front filtering, reference-vector selection,
and final truncation. Its partition selection follows PlatEMO RVEAa: feasible APD
winners take priority, otherwise the minimum violation wins. In constrained runs,
the first-front filter jointly compares objectives and total violation as an extra
objective. This extends PlatEMO's objective-only prefilter, protects feasible points
from infeasible dominators and preserves objective/violation tradeoffs before partition
selection, avoiding premature collapse to a single minimum-violation individual.
Final fixed-shape batch truncation prioritizes violation, then nearest angular crowding;
it retains at most the sampled population size and pads other slots with NaNs in
population, objectives and violations. This is a batch approximation of angular
truncation, not PlatEMO's iterative truncation. Unconstrained filtering and truncation
remain unchanged. Custom environmental selection uses `(pop, fit, vectors, theta, cv)`
and returns `(pop, fit, cv)` for constrained runs, matching RVEA's constraint interface.

`TensorMOEAD` accepts `(fitness, cv)` through `UnifiedWorkflow` and retains selected
violations in `cv`. Both update stages prioritize total positive violation: a child
can replace a neighbor with higher violation, or with equal violation and a strictly
better first aggregation value. When children compete for one subproblem, the second
stage minimizes violation first and then its aggregation value. This uses the
constraint-comparison approach of PlatEMO C-MOEA/D while retaining TensorMOEA/D's batch
update framework and strict improvement rule. Ideal-point and objective-range updates
include all evaluated points. Constraint comparisons use fixed-shape tensor operations;
unconstrained evaluations retain the original aggregation-only path.

`NSGA3` accepts `(fitness, cv)` through `UnifiedWorkflow` and preserves selected
violations in `cv`. Constrained runs follow PlatEMO NSGA-III: mating tournaments
compare total positive violation; environmental sorting ranks feasible solutions
by objective dominance, followed by infeasible solutions ordered by violation.
Equal-violation infeasible solutions share a front, with reference-point niching
resolving the cutoff front. Violations are sorted with fixed-shape tensor operations
instead of peeling one front per distinct violation. The historical feasible ideal
point is saved in `z_min`; before any feasible evaluation, normalization uses an
all-ones fallback. Degenerate normalization retains the existing safe fallback.
Unconstrained evaluations retain the original rank-based mating and normalization.
Custom mating operators keep the `(pop_size, keys)` interface: `keys` contains rank
for unconstrained runs and total positive violation for constrained runs.

`RVEA` also accepts `(fitness, cv)` through `UnifiedWorkflow`. Its `cv` state stays
aligned with population and objective slots, including NaN padding for empty reference
vectors. Within each reference-vector partition it follows PlatEMO RVEA: select the
feasible solution with minimum angle-penalized distance (APD), or, if no solution in
that partition is feasible, select the solution with minimum total positive violation.
This is a local feasibility rule; infeasible survivors in other partitions are allowed.
The selection uses fixed-shape tensor masks. A custom RVEA `selection_op` must accept
`(pop, fit, vectors, theta, cv)` and return `(pop, fit, cv)` for constrained evaluations;
the four-argument, two-result interface is preserved for unconstrained evaluations.

## Remaining algorithm families

All 57 public algorithm classes now accept objective-only or `(fitness, cv)` evaluations.
Objective-only evaluations avoid the constraint-comparison path. Independent search and
batch-update corrections are listed below and can change objective-only trajectories.
Constraint state follows the same selected indices as decisions, objectives, masks, personal bests and
archives. Positive violations are summed on the tensor device; negative inequality values
are clamped to zero for comparison. The stored `cv` retains the evaluator's original values.
For column-form violations, test feasibility with `cv.clamp_min(0).sum(dim=1) == 0`,
and exclude nonfinite objectives; for 1D totals, use `cv.clamp_min(0) == 0`. A problem must
consistently return the same evaluation structure.

| Family | Additional classes | Constraint selection |
| --- | --- | --- |
| Front and dominance selection | `AGEMOEA`, `BiGE`, `GDE3`, `GrEA`, `KnEA`, `NSGAII_SDR`, `OSP_NSDE`, `SIBEA`, `SNSGA2`, `VaEA` | Constrained dominance/fronts, followed by the existing crowding, geometry, grid or indicator rule. Equal positive violations share a front. |
| Sparse decisions | `DMMOEA`, `SparseEA`, `SparseEA2`, `TSSparseEA` | Joint objective/total-violation fronts for variable probes; constrained fronts for survivors. Raw decisions, masks and violations stay aligned. |
| Decomposition | `MOEAD`, `MOEADAWA`, `MOEAD_DCWV`, `MOEADDE`, `MOEAD_DRA`, `MOEADDU`, `MOEADDYTS`, `MOEADFRRMAB`, `MOEAD_PaS`, `MOEAURAW`, `BCEMOEAD` | C-MOEA/D comparison: lower total violation, then the original scalar comparison at equal violation. Neighborhood/replacement limits remain. Batched constrained proposals use deterministic reductions instead of conflicting writes. |
| Indicators and archives | `BCE_IBEA`, `Two_Arch2`, `SSCEA`, `PICEAg`, `PREA`, `eMOEA`, `PESA2` | Constraint-aware population/archive selection; native indicators, goals, epsilon grids or diversity resolve ties. Each archive retains its own violation state. |
| Reference and scalar selection | `CLIA`, `ThetaDEA`, `tDEA_CPBI`, `SPEAR`, `MaOEACSS`, `WASFGA`, `GWASFGA` | Violation priority combined with the existing clustering, reference or scalar selection. |
| Swarms | `CMOPSO`, `SMPSO`, `LMOCSO`, `TELSO` | Violations follow swarm state and competition. SMPSO also updates personal bests and its archive by constrained dominance. LMOCSO/TELSO use feasible APD winners or minimum violation within each reference partition; infeasible survivors in other partitions are allowed. |
| Other population structures | `CMOEA_MS`, `CoMMEA`, `NSBiDiCo`, `EFRRR`, `LSMOF`, `TSNSGAII`, `WOF` | Violations propagate through both populations, auxiliary archives, stage selections, directional/weight probes or transformed decisions as applicable. |

`CMOEA_MS` follows PlatEMO's two scoring stages: initially it optimizes shifted density
and normalized violation jointly; once the feasible fraction exceeds `lambda_` and 10% of
`max_gen` has elapsed, it uses objective/CV strength fitness. Constraint columns are
normalized separately, then averaged. `max_gen=100` and `lambda_=0.5` are optional constructor
parameters; `max_gen` controls the stage transition, while the caller controls termination.
The early stage can retain infeasible objective/violation tradeoffs. Constrained `CoMMEA`
keeps a full second mating population: violation takes priority, followed by epsilon
eligibility and dual-space crowding. The candidate pool supplies fallbacks when epsilon
filtering would leave a singleton. `NSBiDiCo` likewise uses
joint objective/violation fronts for its auxiliary archive, with constrained fronts for its
main population. Its archive retains feasible entries when present, rather than copying
PlatEMO's infeasible-only archive and bidirectional mating literally.

These implementations reference [PlatEMO](https://github.com/BIMK/PlatEMO/tree/d25e65d1ffba58dbf4d7e1b5259786187d12968a/PlatEMO/Algorithms/Multi-objective%20optimization),
particularly constrained `NDSort`, C-MOEA/D, CMOEA-MS and reference-partition selection.
PlatEMO's base versions of many algorithms (including most MOEA/D variants, indicators,
grids and swarms) only compare objectives. Their constraint support here is an extension
using those constrained comparison rules. Existing EvoMO approximations are retained:
this is not an exact reproduction of every PlatEMO operator, archive, normalization or
truncation rule. In particular, `tDEA_CPBI` uses violation priority with EvoMO's existing
cluster selection rather than porting PlatEMO's constraint-normalization pipeline.
Sparse variable probes collapse constraint columns to total violation for joint ranking;
PlatEMO uses individual constraint columns. Zero normalization spans are guarded.

Constrained `PESA2` fills fronts up to the requested population size, then applies grid
truncation only to the cutoff front; retaining only the minimum-CV front can collapse
an infeasible mating population to one point. Constrained `eMOEA` samples dominated
replacement targets randomly to spread simultaneous proposals across incumbents.
`DMMOEA` uses real-valued polynomial mutation in constrained search, and duplicate
candidate pools are padded rather than leaving front selection without enough entries.
`MOEAD_DRA` also uses polynomial mutation for constrained offspring. The utilities in
`MOEAD_DRA`, `MOEADAWA`, `MOEADDYTS` and `MOEADFRRMAB` credit relative violation changes
first, and objective progress at equal violation; constrained utility remains in [0, 1].
`OSP_NSDE` retains ordinary variation until feasibility and applies its existing approximate
forecast periodically afterward; this does not port PlatEMO's full ARX/HV trigger.

`SMPSO` samples leaders only from occupied archive slots. Its mutation gates particles
with probability 0.15 and applies the operator's built-in 1/D variable probability once.
These two corrections also apply to objective-only runs, so their trajectories can change.
Evaluate its feasible archive as well as the current swarm when reporting quality.
NaN violations compare as positive infinity while the stored raw values are preserved.

The EvoX polynomial-mutation argument `pro_m` is the expected mutation count per individual;
the operator divides it by D. `BiGE`, `CMOPSO`, `KnEA`, `MaOEACSS`, `CMOEA_MS`, `NSGAII_SDR`,
`EFRRR`, `TSNSGAII`, `TSSparseEA` and `WOF` use `pro_m=1` for a 1/D variable probability,
correcting their former 1/D-squared probability. `eMOEA`, `MOEADAWA` and `PREA` supply SBX
parents in first-half/second-half order. PREA uses N offspring per generation, matching
PlatEMO's `OperatorGAhalf`; compare runs by evaluations rather than generations.
Objective-only batch updates in `eMOEA`, `MOEADDE`, `MOEAD_DCWV` and `MOEADFRRMAB` resolve
duplicate targets with a shared last-source reduction for decisions and objectives;
separate conflicting CUDA writes could otherwise select different sources for each tensor.
These corrections apply to constrained and objective-only execution as appropriate.

A subsequent source audit corrected PREA's ratio-indicator direction and refreshed its
indicator minima after each diversity deletion. CMOPSO computes leader crowding within
every front, samples two distinct leaders, keeps the first on equal angles and uses iterative distance truncation.
CMOEA-MS applies the same two-stage scoring to objective-only and constrained evaluations:
the early shifted density uses normalized Euclidean distance, while late strength fitness
and truncation use raw-objective cosine distance. KnEA uses distinct maximum-objective
extreme points, signed hyperplane distance and the previous knee fraction to update its
radius. A pseudoinverse guards singular extreme-point matrices.

MOEADAWA accepts `max_evaluations`, including initialization, for adaptation scheduling.
Set it to the actual run budget; the default is `100 * actual_population_size`, and it
does not terminate the run. Utility updates follow ten population-equivalent evaluations;
archive and weight adaptation start at `rate_evol=0.8`. `weight_update_interval=100`
corresponds to the source's interval divided by five subgenerations. Parent pools are
randomized without replacement and also determine replacement targets. Adaptation uses
iterative neighbor-distance products and preserves the weights of retained subproblems.
Offspring within a subgeneration remain batched; PlatEMO generates them sequentially.
Neighborhoods are refreshed after weight changes, whereas the reference main loop keeps
its initial neighborhood matrix.

Other existing approximations remain relevant when comparing source algorithms:
eMOEA updates epsilon archives in batches, MOEADDYTS simplifies its five variation
operators and per-child bandit updates, MOEADFRRMAB batches operator credit assignment,
and TSSparseEA approximates the reference's grouped binary optimization. Constraint
comparison tests do not establish equivalence of these complete search procedures.

Constraint comparisons use batched tensor operations and fixed-shape reductions where
applicable. Several complete algorithms already use data-dependent loops, dynamic
archives or boolean indexing. Those existing paths can prevent `torch.compile(fullgraph=True)`;
constraint support alone does not establish whole-step compilation, CUDA graph or `vmap`
support. In the tested Windows/PyTorch 2.11 environment, EFRRR captures a full graph
but CPU Inductor corrupts fitness dtype; this also reproduces with the original unconstrained
EFRRR. Its CUDA Inductor and CPU AOT eager checks pass. Initialize the workflow before compiling. `MOEAD` and `LMOCSO` have an explicit
`device` constructor argument; pass it when using CUDA bounds.

A constrained run requires the problem to return violations, the workflow to preserve the
tuple (for example `UnifiedWorkflow`), and the selected algorithm to consume the violations.
Distance metrics must use finite feasible solutions. A zero feasibility rate is a valid
experimental outcome; report IGD/GD as unavailable, rather than computing them on infeasible
points. Benchmark stochastic quality with several seeds and equal evaluation budgets;
probe evaluations and sampled population sizes can make generation counts unequal in cost.

Run the core algorithm state, selection and regression tests with:

```sh
python -m pytest unit_test/algorithms/test_moea_constraints.py -q
python -m pytest unit_test/algorithms/test_moea_regressions.py -q
python -m pytest unit_test/algorithms/test_moea_selection.py -q
```

The algorithm tests cover CPU/CUDA, scalar and column violations, empty constraints,
constant objectives, equal-violation fronts, feasible and lower-violation survivor priority,
archive/mask alignment, empty exploration and tight epsilon filtering. Separate tests
execute selected complete constrained steps with Inductor; graph capture alone is insufficient.
