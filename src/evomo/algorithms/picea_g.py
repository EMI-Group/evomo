import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, lexsort

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    rank_with_constraints,
)
from evomo.utils import parse_evaluate, register_lazy_buffer


class PICEAg(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the PICEAg population and optimization state.

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
        self.dim = lb.numel()

        # PICEA-g specific parameters
        self.n_goal_val = 100 * n_objs

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Coevolved goals
        self.goal = Mutable(torch.zeros((self.n_goal_val, n_objs), device=device))

        # Archive
        self.archive = Mutable(torch.zeros((pop_size, self.dim), device=device))
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        register_lazy_buffer(self, "archive_cv", device_like="pop")

    def _get_goal_satisfaction(self, fit: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        # Bug #29: Vectorization Override
        return (fit.unsqueeze(1) <= goals.unsqueeze(0)).all(dim=-1)

    def _gene_goal(self, combined_fit: torch.Tensor):
        # Blueprint 3.A: Goal Generation
        f_min = torch.min(combined_fit, dim=0)[0]
        f_max = torch.max(combined_fit, dim=0)[0]
        new_goals = f_min + torch.rand((self.n_goal_val, self.n_objs), device=self.lb.device) * (f_max * 1.2 - f_min)
        return new_goals

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        # Initial goals
        self.goal = self._gene_goal(self.fit)
        # Initial Archive
        self.archive = self.pop.clone()
        self.archive_fit = self.fit.clone()
        self.archive_cv = None if self.cv is None else self.cv.clone()

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        device = self.lb.device
        N = self.pop_size

        # 1. Mating
        mating_pool = torch.randint(0, N, (N,), device=device)
        offspring = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit, off_cv = parse_evaluate(self.evaluate(offspring))

        # 3. Archive Update (Blueprint 3.B: Brutal Static Truncation)
        combined_arc_pop = torch.cat([self.archive, offspring], dim=0)
        combined_arc_fit = torch.cat([self.archive_fit, off_fit], dim=0)

        # Filter for Pareto Rank 1
        combined_arc_cv = cat_violation(self.archive_cv, off_cv)
        ranks = rank_with_constraints(combined_arc_fit, combined_arc_cv)
        rank1_mask = ranks == 0
        rank1_pop = combined_arc_pop[rank1_mask]
        rank1_fit = combined_arc_fit[rank1_mask]

        # Brutal Truncation
        # Fix for Bug #41: Avoid in-place fill_diagonal_ which breaks JIT view chains
        dist = torch.cdist(rank1_fit, rank1_fit, p=2)
        eye_mask = torch.eye(dist.shape[0], device=device, dtype=torch.bool)
        dist = torch.where(eye_mask, torch.full_like(dist, torch.inf), dist)

        # Handle potential empty rank1_fit (though unlikely in MOO)
        min_d = torch.where(rank1_mask.any(), torch.min(dist, dim=1)[0], torch.zeros(rank1_pop.shape[0], device=device))

        # Sort by min_d descending
        idx_arc = torch.argsort(min_d, descending=True)
        # Keep top N (or all if less than N)
        num_to_keep = torch.minimum(torch.tensor(N, device=device), torch.tensor(rank1_pop.shape[0], device=device))
        self.archive = rank1_pop[idx_arc[:num_to_keep]]
        self.archive_fit = rank1_fit[idx_arc[:num_to_keep]]
        if combined_arc_cv is not None:
            self.archive_cv = combined_arc_cv[rank1_mask][idx_arc[:num_to_keep]]

        # 4. Environmental Selection (Blueprint 3.C)
        # Refresh goals
        new_goals = self._gene_goal(torch.cat([self.fit, off_fit], dim=0))
        combined_goal = torch.cat([self.goal, new_goals], dim=0)
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        # Goal Satisfaction Matrix
        S = self._get_goal_satisfaction(combined_fit, combined_goal)  # [N_total, N_goal_total]

        # Fitness Calculation (Bug #12: Safe Division)
        ng = S.sum(dim=0).float()  # [N_goal_total]
        inv_ng = 1.0 / (ng + 1e-6)
        Fs = (S.float() * inv_ng.unsqueeze(0)).sum(dim=1)  # [N_total]

        Fg = 1.0 / (1.0 + (ng - 1.0) / (N - 1.0) + 1e-6)
        # Bug #41: JIT Compliance (torch.where for ng==0)
        Fg = torch.where(ng == 0, torch.full_like(Fg, 0.5), Fg)

        # Selection Strategy - Solutions
        combined_cv = cat_violation(self.cv, off_cv)
        sol_ranks = rank_with_constraints(combined_fit, combined_cv)
        # Bug #25: Lexsort Axis (Primary key LAST)
        sel_idx = lexsort(torch.stack([-Fs, sol_ranks.float()]))
        self.pop = combined_pop[sel_idx[:N]]
        self.fit = combined_fit[sel_idx[:N]]
        if combined_cv is not None:
            self.cv = combined_cv[sel_idx[:N]]

        # Selection Strategy - Goals
        goal_idx = torch.argsort(Fg, descending=True)
        self.goal = combined_goal[goal_idx[: self.n_goal_val]]


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = PICEAg(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
