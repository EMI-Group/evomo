from typing import Optional

import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    prefer_by_constraint,
    rank_with_constraints,
    take_violation,
    total_violation,
)
from evomo.utils import parse_evaluate, register_lazy_buffer

# Recommended grid divisions, indexed by the number of objectives
_DEFAULT_DIV = (0, 0, 45, 15, 10, 9, 9, 8, 8, 10, 12)


class GrEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, div: Optional[int] = None, **kwargs):
        """Initialize the GrEA population and optimization state.

        :param pop_size: Required. Requested number of candidate solutions. Use a positive integer; algorithm-specific
            minimums and reference-vector sampling are described below.
        :type pop_size: int
        :param n_objs: Required. Number of objectives, matching the second dimension of the problem's objective tensor.
            Objectives are minimized; this library targets two or more objectives.
        :type n_objs: int
        :param lb: Required. Lower decision bounds of shape ``(D,)``, where ``D`` is the number of decision variables.
            Use floating-point bounds with the same shape, dtype and device as ``ub``, and ``lb <= ub`` elementwise.
        :type lb: torch.Tensor
        :param ub: Required. Upper decision bounds of shape ``(D,)``. The decision dimension is inferred from the
            bounds, rather than passed separately. Match ``lb`` in shape, dtype and device.
        :type ub: torch.Tensor
        :param div: Default: ``None``. Positive number of grid divisions per objective. More divisions give finer
            objective-space cells; grid occupancy or grid distance is used during selection. ``None`` uses the
            setting recommended for the number of objectives (45, 15, 10, 9, 9, 8, 8, 10, 12 for 2 to 10 objectives,
            as in PlatEMO).
        :type div: int
        :param kwargs: Default: ``{}``. Extra keyword arguments are accepted for constructor compatibility but are not
            read by this implementation. In particular, passing ``device=...`` here does not move tensors; place both
            bounds on the intended device before construction.
        :type kwargs: dict

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            accepts an objective tensor or a ``(fitness, constraint_violation)`` tuple.
            Constrained evaluations retain violations through selection and state updates.

            Tensor allocation uses ``lb.device``. There is no explicit device parameter; prepare both bounds on the intended
            device.
        """
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        if div is None:
            div = _DEFAULT_DIV[min(n_objs, 10)]
        self.div = Parameter(torch.tensor(div, dtype=torch.int32, device=device))
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))

    def _grid(self, fit: torch.Tensor):
        """Grid location of each solution, with the grid's lower bound and cell width."""
        div = self.div.to(fit.dtype)
        f_min = torch.min(fit, dim=0).values
        f_max = torch.max(fit, dim=0).values
        lb = f_min - (f_max - f_min) / 2 / div
        ub = f_max + (f_max - f_min) / 2 / div
        d = (ub - lb) / div
        g_loc = torch.floor((fit - lb) / d)
        # A constant objective has zero cell width; all its solutions share cell 0
        g_loc = torch.nan_to_num(g_loc, nan=0.0)
        return g_loc, lb, d

    @staticmethod
    def _grid_difference(g_loc: torch.Tensor) -> torch.Tensor:
        """GD(i, j) = sum_k |G_k(i) - G_k(j)|, with an infinite diagonal."""
        gd = torch.sum(torch.abs(g_loc.unsqueeze(1) - g_loc.unsqueeze(0)), dim=-1)
        return gd.fill_diagonal_(torch.inf)

    @staticmethod
    def _grid_dominance(g_loc: torch.Tensor) -> torch.Tensor:
        """Returns a matrix where out[i, j] is True if i grid-dominates j."""
        return (g_loc.unsqueeze(1) <= g_loc.unsqueeze(0)).all(-1) & (g_loc.unsqueeze(1) < g_loc.unsqueeze(0)).any(-1)

    def _mating_selection(self) -> torch.Tensor:
        """Binary tournament on Pareto dominance, then grid dominance, then the smaller GCD."""
        N = self.pop_size
        device = self.pop.device
        g_loc, _, _ = self._grid(self.fit)
        gcd = torch.clamp(self.n_objs - self._grid_difference(g_loc), min=0).sum(dim=1)
        p1 = torch.randint(0, N, (N,), device=device)
        p2 = torch.randint(0, N, (N,), device=device)
        f1, f2 = self.fit[p1], self.fit[p2]
        g1, g2 = g_loc[p1], g_loc[p2]
        dominate = (f1 < f2).any(dim=1).int() - (f1 > f2).any(dim=1).int()
        g_dominate = (g1 < g2).any(dim=1).int() - (g1 > g2).any(dim=1).int()
        tie = (dominate == 0) & (g_dominate == 0)
        first = (dominate == 1) | (g_dominate == 1) | (tie & (gcd[p1] <= gcd[p2]))
        if self.cv is not None:
            total = total_violation(self.cv)
            first = prefer_by_constraint(first, total[p1], total[p2])
        return torch.where(first, p1, p2)

    def _last_selection(self, fit: torch.Tensor, k: int) -> torch.Tensor:
        """Choose ``k`` solutions of the last front one by one, by GR, then GCD, then GCPD (Yang et al., 2013)."""
        n, m = fit.shape
        g_loc, lb, d = self._grid(fit)
        gr = g_loc.sum(dim=1)
        gcd = torch.zeros(n, dtype=fit.dtype, device=fit.device)
        gcpd = torch.sqrt(torch.sum(((fit - (lb + g_loc * d)) / d) ** 2, dim=1))
        gcpd = torch.nan_to_num(gcpd, nan=0.0)
        gd = self._grid_difference(g_loc)
        dominates = self._grid_dominance(g_loc)
        zeros = torch.zeros_like(gcd)

        remain = torch.ones(n, dtype=torch.bool, device=fit.device)
        for _ in range(k):
            # Best remaining solution: smallest GR, ties broken by GCD, then by GCPD
            best = remain & (gr == torch.where(remain, gr, torch.inf).min())
            best = best & (gcd == torch.where(best, gcd, torch.inf).min())
            q = torch.argmin(torch.where(best, gcpd, torch.inf))
            remain[q] = False

            gcd = gcd + torch.clamp(m - gd[q], min=0)

            # GR adjustment: same cell +(M + 2), grid-dominated by q +M, and a distance-based penalty PD for
            # the neighbours of q and the solutions they grid-dominate
            eq = (gd[q] == 0) & remain
            gq = dominates[q] & remain
            ngq = remain & ~gq
            near = (gd[q] < m) & ngq & ~eq
            direct = torch.where(near, m - gd[q], zeros)
            # Grid dominance is transitive, so PD(r) is the largest direct penalty of a neighbour dominating r
            inherited = torch.where(dominates & near.unsqueeze(1), direct.unsqueeze(1), 0).amax(dim=0)
            pd = torch.maximum(direct, torch.where(remain & ~(gq | eq), inherited, zeros))
            gr = gr + eq * (m + 2) + gq * m + torch.where(ngq & ~eq, pd, zeros)
        return ~remain

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        N = self.pop_size

        # 1. Mating Selection
        mating_idx = self._mating_selection()

        # 2. Variation
        parents = self.pop[mating_idx]
        offspring = simulated_binary(parents)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit, off_cv = parse_evaluate(self.evaluate(offspring))

        # 3. Environmental Selection: complete fronts first, then the last front by grid-based selection
        c_pop = torch.cat([self.pop, offspring], dim=0)
        c_fit = torch.cat([self.fit, off_fit], dim=0)
        c_cv = cat_violation(self.cv, off_cv)
        c_rank = rank_with_constraints(c_fit, c_cv)
        last_rank = torch.sort(c_rank).values[N - 1]
        survivor_mask = c_rank < last_rank
        last = torch.where(c_rank == last_rank)[0]
        chosen = self._last_selection(c_fit[last], N - int(survivor_mask.sum()))
        survivor_mask[last[chosen]] = True

        self.pop = c_pop[survivor_mask]
        self.fit = c_fit[survivor_mask]
        self.cv = take_violation(c_cv, survivor_mask)


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = GrEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    jit_state_step()

    torch.cuda.synchronize()
    exec_start = time.perf_counter()

    for i in range(1, 50):
        jit_state_step()

        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            fit = fit[~torch.any(torch.isnan(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")
