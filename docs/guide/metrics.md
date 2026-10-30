# Solution quality metrics

Import `gd`, `igd`, and `hv` from `evomo.metrics`. These functions evaluate sets in objective
space, not decision variables. Remove nonfinite values before calling them. For constrained
problems, also filter feasible solutions. Use consistent objective directions, scales,
reference fronts, and reference points when comparing runs.

## GD and IGD

Let A be the resulting solution set and P the reference front. Let `d(a, P)` denote the minimum
Euclidean distance from point a to P. The current implementations use:

$$
\operatorname{GD}(A,P)=\frac{\sqrt{\sum_{a\in A}d(a,P)^2}}{|A|}
$$

$$
\operatorname{IGD}_p(A,P)=\left(\frac{1}{|P|}\sum_{q\in P}d(q,A)^p\right)^{1/p}
$$

`gd(objs, pf)` uses this fixed definition, rather than the arithmetic mean or root mean square
of nearest-point distances. `igd(objs, pf, p=1)` defaults to the arithmetic mean of distances
from reference points to the solution set. Lower values are better. Inputs must be nonempty,
and IGD's `p` must be positive.

```python
from evomo.metrics import gd, igd

reference_front = problem.pf().to(device=pareto_fitness.device, dtype=pareto_fitness.dtype)
gd_score = gd(pareto_fitness, reference_front)
igd_score = igd(pareto_fitness, reference_front)
```

Use this after result filtering in the quickstart. `torch.cdist` constructs a distance matrix
between the two sets, so large solution sets and reference fronts can require substantial
memory. Objectives with larger scales can dominate distances. If you normalize objectives,
apply the same transform to both sets and record the normalization rule.

## HV: Monte Carlo estimation

`hv(objs, ref, num_sample=100000)` estimates hypervolume for minimization. `ref` has shape `(M,)`.
Choose a reference point worse than the solutions of interest and keep it fixed across
comparisons. Points outside it contribute no volume. Higher HV is better for fixed scales
and a fixed reference point.

```python
import torch
from evomo.metrics import hv

objectives = torch.tensor([[0.2, 0.8], [0.5, 0.4], [0.8, 0.2]])
reference = objectives.new_tensor([1.1, 1.1])
torch.manual_seed(42)
print(hv(objectives, reference, num_sample=10000))
```

This is a stochastic estimate, not an exact hypervolume calculation. Increasing `num_sample`
increases runtime and memory costs; comparisons scale with sample count, solution count, and objective count.
All samples are evaluated together using broadcast tensor operations.
Float16, bfloat16 and integral inputs are promoted
to at least float32, and float64 inputs retain float64 arithmetic. An empty set returns floating
scalar zero. Nonfinite objective rows and reference-boundary points contribute no volume;
a nonfinite reference returns NaN. Invalid input shapes, nonpositive sample counts and
noninteger sample counts raise errors.

In many objectives, a positive dominated volume can occupy a tiny fraction of the
sampling box. A finite Monte Carlo run may then return zero because no sample hits
that region; zero does not prove that the exact HV is zero.

This function reports volume in the supplied objective scales. [PlatEMO's `HV`](https://github.com/BIMK/PlatEMO/blob/master/PlatEMO/Metrics/HV.m) first normalizes
objectives using its reference front and computes exact volume below four objectives, so its
reported values require matching normalization before comparison.

For maximization, negate both objectives and the reference point, for example
`hv(-rewards, -reference_rewards)`. For mixed directions, transform both inputs using the
same per-objective directions. Do not pass restored maximization objectives directly to HV
under its minimization convention.

## Reproducible comparisons

Record algorithm seeds, evaluation budgets, the extraction scope (final population or historical
archive), reference front sampling, HV reference points, and sample counts. Across multiple
seeds, report distributions or means with dispersion rather than treating one run as a stable result.
