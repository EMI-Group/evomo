# Algorithm parameter guide

Every public algorithm's generated API page documents its constructor parameters in English,
including types, shapes, defaults, and their meaning in the current implementation.
This guide explains shared conventions and the parameters that are easiest to confuse.
The [algorithm catalogue](../guide/algorithms.md) lists all public class names.

Recommendations such as positive horizons are usage requirements, not a promise that each
constructor validates every input. Check the class's API for its exact signature.

## Population, objectives, bounds, and device

| Parameter | Type and shape | Meaning |
| --- | --- | --- |
| `pop_size` | Positive `int` | Requested candidate count. Some reference-vector algorithms use the sampled vector count instead; inspect `algorithm.pop.shape[0]`. `GDE3` requires at least 4, and `MOEAD`/`TensorMOEAD` require a sampled size above 10. |
| `n_objs` | `int`, normally at least 2 | Number of minimized objectives, matching the problem's fitness columns. It is independent of decision dimension. |
| `lb`, `ub` | Floating tensors, `(D,)` | Per-variable bounds. Match shapes, dtypes, and devices, with `lb <= ub`. `D` is inferred from these tensors. |
| `device` | `torch.device` or `None`, where explicitly supported | `None` means `torch.get_default_device()` for constructors that expose this argument. Those constructors move bounds to the selected device. Most other algorithms allocate on `lb.device`. |
| `**kwargs` | Extra named settings, where accepted | Generally ignored, including an unrecognized `device` keyword. `LSMOF` reads four named settings described below. An accepted keyword does not necessarily enable a feature. |

The constructors with an explicit `device` argument are `HypE`, `IBEA`, `LMOCSO`, `MOEAD`,
`NSGA2`, `NSGA3`, `RVEA`, `RVEAa`, and `TensorMOEAD`. For other algorithms, put both bounds
on the intended device before constructing the algorithm.

After initialization, common state tensors are `pop` with decision dimension `D` and `fit`
with objective dimension `n_objs`. Algorithms can also keep archives, sparse masks, velocities,
reference vectors, or NaN placeholders; inspect their state and filter invalid results when needed.
`step()` updates state and returns `None`. Initialize with `workflow.init_step()` first.

Only the documented `NSGA2` evaluation path consumes `(fitness, constraint_violation)` in this
algorithm collection. The other current implementations expect an objective tensor. See the
[constraint guide](../guide/constraints.md) before connecting a constrained problem.

## Decomposition and differential evolution

Here `N` denotes the actual population size after weight sampling.

| Parameter | Algorithms and defaults | Meaning and limits |
| --- | --- | --- |
| `T` | `MOEAD_DCWV`, `MOEAD_DRA`, `MOEADDU`, `MOEADDYTS`, `MOEADFRRMAB`: `20` | Nearest-weight mating neighborhood size, including self; positive values are capped at `N`. |
| `T` | `BCEMOEAD`, `MOEAD_PaS`: `None` | Resolve to `ceil(N / 10)`, then clamp with `min(max(2, T), N)`. |
| `T` | `MOEADAWA`: `None` | Resolve to `ceil(N / 10)`; explicit values must satisfy `1 <= T <= N` because they are not capped. |
| `delta` | `MOEADDE`, `MOEADDU`, `MOEADFRRMAB`, `MOEAURAW`: `0.9` | Probability of drawing mating parents locally, in `[0, 1]`; other draws use the full population. |
| `nr` | `MOEADDE`, `MOEAD_DRA`, `MOEADDYTS`, `MOEADFRRMAB`, `MOEAURAW`: `2` | Maximum incumbents replaced per offspring; this is separate from mating neighborhood size. |
| `nr` | `BCEMOEAD`, `MOEADAWA`: `None` | Resolve to `ceil(N / 100)`; `BCEMOEAD` also clamps the result to `[1, N]`. |
| `nEP` | `MOEAURAW`: `200`; `MOEADAWA`: `None` | External archive capacity; `MOEADAWA` resolves `None` to `ceil(1.5 * N)`. |
| `F` | `GDE3`, `MOEADDE`: `0.5` | Positive differential step scale applied to parent differences. |
| `CR` | `GDE3`: `0.5` | Mutant-component probability in binomial crossover, in `[0, 1]`; a mutant component is also forced per trial. |
| `CR` | `MOEADDE`: `1.0` | Accepted and stored, but currently unused. Its variation step forms differential offspring and applies polynomial mutation without binomial crossover. |
| `K` | `EFRRR`: `2`; `MOEADDU`: `5` | Respectively, closest reference weights allowed in restricted ranking, or cosine-nearest subproblems considered for updating. Positive and no larger than the available vector count. |
| `window_size` | `MOEADFRRMAB`: `None` | Reward-window entry count, default `ceil(N / 2)`; use a positive value. A workflow step performs five internal subgenerations, so the window does not count workflow steps. |
| `p` | `MOEAD_DCWV`: `-1.0` | `-1.0` enables adaptive weight transformation. Other values disable that branch; their numeric values are not applied as fixed transformations by the current code. |
| `aggregate_op` | `TensorMOEAD`: `("pbi", "pbi")` | Pair of aggregation names for incumbent comparison and eligible-offspring selection. Options: `pbi`, `tchebycheff`, `tchebycheff_norm`, `modified_tchebycheff`, `weighted_sum`. |

## Schedules, selection, and sparsity

| Parameter | Algorithms and defaults | Meaning and limits |
| --- | --- | --- |
| `max_gen` | `LMOCSO`, `RVEA`, `RVEAa`, `MOEAD_PaS`, `SSCEA`: `100` | Positive generation horizon used in adaptation or penalty schedules. The caller controls termination. |
| `max_fe` | `TELSO`, `TSNSGAII`: `10000`; also an `LSMOF` keyword | Positive evaluation horizon. `TSNSGAII` and `LSMOF` switch phases at half this horizon; `TELSO` uses it to scale learning and penalty progress. It does not terminate the workflow. |
| `alpha` | `LMOCSO`, `RVEA`, `RVEAa`: `2.0` | Exponent of `(gen / max_gen) ** alpha`; larger positive values delay angular penalty growth. |
| `fr` | `RVEA`, `RVEAa`: `0.1` | Positive adaptation rate, using interval `max(round(1 / fr), 1)` steps. The default interval is 10 steps, independent of `max_gen`. |
| `t_max` | `OSP_NSDE`: `100` | Trajectory-buffer capacity including the initial entry. Keep subsequent step count below this capacity; the buffers do not wrap. |
| `kappa` | `BCE_IBEA`, `IBEA`: `0.05` | Positive scale for exponential indicator fitness; smaller values strengthen the effect of indicator differences. |
| `n_sample` | `HypE`: `10000` | Positive Monte Carlo sample count per hypervolume estimate; increasing it reduces sampling noise at additional cost. |
| `div` | `GrEA`, `PESA2`: `10` | Positive grid division count per objective; larger values give finer objective-space cells. |
| `type` | `CMOEA_MS`: `1` | `1` uses SBX and polynomial mutation; any other value enters the fixed-scale differential-variation branch. |
| `t` | `MaOEACSS`: `0.0` | Threshold on normalized convergence-score differences when eliminating a close angular pair. |
| `eps` | `CoMMEA`: `0.2` | Multiplicative epsilon tolerance using `(1 + eps) * front1_fit`; its behavior depends on objective scale and sign. |
| `epsilon` | `eMOEA`: `0.05` | Positive additive objective-grid width shared by all objectives; expressed in objective-value units. |
| `sLower`, `sUpper` | `SNSGA2`: `0.1`, `0.9` | Initial zero-component fraction endpoints, with `0 <= sLower <= sUpper <= 1`. They control initialization, not a permanent sparsity guarantee. |
| `gamma` | `WOF`: `10` | Positive decision-group count, also the weight-vector dimension. Normally no larger than `D` to avoid empty groups. |
| `point` | `WASFGA`: `None` | Preferred objective-space point, `(n_objs,)` or `(1, n_objs)`; `None` creates zeros. Explicit points are not moved to the bounds' device automatically. |
| `data_type` | `NSGA3`: `None` | `torch.bool` selects Boolean initialization; other values use continuous initialization. It does not cast to an arbitrary dtype. Boolean runs need compatible variation operators. |

`LSMOF` additionally reads `wD=5` reference solutions, `SubN=20` weight candidates,
`wmax=0.1` decision-reconstruction scale, and `max_fe=10000` from `**kwargs`.
The first phase evaluates `2 * wD * SubN` decisions per step. Use positive counts and ensure
enough distinct reference solutions for `wD`. `wmax` scales offsets along unit directions;
it is not an evaluation-budget fraction. Initial evaluations count toward its phase switch.

## Custom operator contracts

| Argument | Calling contract | Defaults and exceptions |
| --- | --- | --- |
| `mutation_op` | `(offspring, lb, ub) -> offspring`; shape `(B, D)` | Where exposed, `None` selects EvoX's `polynomial_mutation`. Preserve decision dimension and device. |
| `crossover_op` | `(parents) -> offspring`; two-dimensional decisions | `None` selects `simulated_binary`; `MOEAD` and `TensorMOEAD` use `simulated_binary_half`, producing one child per parent pair. |
| `selection_op` in `NSGA2`, `NSGA3` | `(pop_size, sort_keys) -> parent_indices` | Mating selection, default `tournament_selection_multifit`. It does not replace environmental selection. |
| `selection_op` in `RVEA`, `RVEAa`, `LMOCSO` | `(pop, fit, vectors, theta) -> (pop, fit)` | Environmental selection. The default is EvoMO's `ref_vec_guided` for RVEA variants and EvoX's version for LMOCSO. Preserve reference-vector output slots. |
| `selection_op` in `HypE` | Constructor accepts it | Currently overwritten by `tournament_selection`; a custom value has no effect. |
| `selection_op` in `MOEAD`, `TensorMOEAD` | Constructor stores it | Currently not called by `step()`; neighborhood mating is handled internally. |

## MOEADFRRMAB example

The example uses explicit bounds on the selected device, an automatically sized reward
window, and three workflow steps. It prints actual population and window sizes. These steps
are not an equal-evaluation-budget benchmark.

```{literalinclude} ../examples/moead_frrmab.py
:language: python
```

See the [MOEADFRRMAB API](../autoapi/evomo/algorithms/moead_frrmab/index.rst)
for all constructor fields and lifecycle methods.
