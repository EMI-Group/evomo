import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import nd_environmental_selection
from evomo.operators.selection.constraint_handling import (
    cat_violation,
    constrained_crowding_selection,
    constraint_keys,
    rank_with_constraints,
    take_violation,
)
from evomo.utils import parse_evaluate, register_lazy_buffer


class CoMMEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, eps: float = 0.2, **kwargs):
        """Initialize the CoMMEA population and optimization state.

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
        :param eps: Default: ``0.2``. Multiplicative epsilon tolerance for comparing the second population with the
            first population's front. Comparisons use ``(1 + eps) * front1_fit``. This is distinct from the additive
            grid width used by ``eMOEA``; choose it with the objective scale and sign in mind.
        :type eps: float
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
        self.eps = Parameter(torch.tensor(eps, device=device))

        # Initialize State (Mutables)
        # Pop 1: Global Convergence
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.spea2_fit1 = Mutable(torch.zeros(pop_size, device=device))

        # Pop 2: Multimodal Diversity
        self.pop2 = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit2 = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.spea2_fit2 = Mutable(torch.zeros(pop_size, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        register_lazy_buffer(self, "cv2", device_like="pop")

    def _harmonic_crowding(self, pop: torch.Tensor, k: int = 3) -> torch.Tensor:
        # Section 3C: Harmonic Crowding Distance (Decision Space)
        dist = torch.cdist(pop, pop, p=2)
        sorted_dist, _ = torch.sort(dist, dim=1)
        # Bug #12: Add 1e-6 to denominator
        crowd = k / (torch.sum(1.0 / (sorted_dist[:, 1 : k + 1] + 1e-6), dim=1) + 1e-6)
        return crowd

    def _cal_spea2_fitness(self, pop: torch.Tensor, fit: torch.Tensor, local_niche: bool) -> torch.Tensor:
        # Section 3A & 3B: SPEA2 Fitness Calculation
        N = pop.shape[0]
        D_mat = torch.cdist(pop, pop, p=2)

        # Bug #24: Pareto Dominance
        dom_obj = (fit.unsqueeze(1) <= fit.unsqueeze(0)).all(-1) & (fit.unsqueeze(1) < fit.unsqueeze(0)).any(-1)

        if local_niche:
            # Section 3A: LocalC Niche Dominance
            avg_dist = D_mat.mean()
            R = torch.where(avg_dist > 8.0, avg_dist / 2.0, avg_dist / 4.0)
            niche_mask = D_mat < R
            dom_mat = dom_obj & niche_mask
        else:
            dom_mat = dom_obj

        # Strength S
        S = dom_mat.sum(dim=1)
        # Raw Fitness R (Bug #23: Use matmul)
        Raw = (dom_mat.t().float() @ S.float().unsqueeze(1)).squeeze()

        # Density D
        dist_sorted, _ = torch.sort(D_mat, dim=1)
        k = int(torch.sqrt(torch.tensor(N, device=pop.device)).floor())
        # Bug #12: Safe division
        Density = 1.0 / (dist_sorted[:, k] + 2.0)

        return Raw + Density

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        self.fit2, self.cv2 = parse_evaluate(self.evaluate(self.pop2))
        self.spea2_fit1 = self._cal_spea2_fitness(self.pop, self.fit, local_niche=False)
        self.spea2_fit2 = self._cal_spea2_fitness(self.pop2, self.fit2, local_niche=True)

    def step(self) -> None:
        # 1. Mating Selection (Pop 1 & Pop 2)
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        idx1 = tournament_selection_multifit(self.pop_size, constraint_keys([self.spea2_fit1], self.cv), tournament_size=2)
        idx2 = tournament_selection_multifit(self.pop_size, constraint_keys([self.spea2_fit2], self.cv2), tournament_size=2)

        # 2. Variation
        off1 = simulated_binary(self.pop[idx1], pro_c=1.0, dis_c=20.0)
        off1 = polynomial_mutation(off1, self.lb, self.ub)
        off1 = clamp(off1, self.lb, self.ub)

        off2 = simulated_binary(self.pop2[idx2], pro_c=1.0, dis_c=20.0)
        off2 = polynomial_mutation(off2, self.lb, self.ub)
        off2 = clamp(off2, self.lb, self.ub)

        # 3. Evaluation
        fit_off1, cv_off1 = parse_evaluate(self.evaluate(off1))
        fit_off2, cv_off2 = parse_evaluate(self.evaluate(off2))

        # 4. Environmental Selection 1 (Global Convergence)
        merged_pop1 = torch.cat([self.pop, off1], dim=0)
        merged_fit1 = torch.cat([self.fit, fit_off1], dim=0)
        if self.cv is None:
            self.pop, self.fit, _, _, _ = nd_environmental_selection(merged_pop1, merged_fit1, self.pop_size)
        else:
            merged_cv1 = cat_violation(self.cv, cv_off1)
            indices, _, _ = constrained_crowding_selection(merged_fit1, merged_cv1, self.pop_size)
            self.pop, self.fit, self.cv = merged_pop1[indices], merged_fit1[indices], merged_cv1[indices]
        self.spea2_fit1 = self._cal_spea2_fitness(self.pop, self.fit, local_niche=False)

        # 5. Environmental Selection 2 (Epsilon-Diversity for Pop 2)
        merged_pop2 = torch.cat([self.pop2, off2], dim=0)
        merged_fit2 = torch.cat([self.fit2, fit_off2], dim=0)

        merged_cv2 = cat_violation(self.cv2, cv_off2)
        rank = rank_with_constraints(merged_fit2, merged_cv2)
        front1_mask = rank == 0
        front1_fit = merged_fit2[front1_mask]

        # Epsilon-dominance check (Section 3D)
        # Bug #29: Vectorized broadcasting
        is_eps_dominated = ((merged_fit2.unsqueeze(1) >= (1 + self.eps) * front1_fit.unsqueeze(0)).all(-1)).any(1)

        # Keep Front 1 OR solutions not epsilon-dominated
        keep_mask = front1_mask | (~is_eps_dominated)

        if merged_cv2 is not None:
            # Keep a full mating population even when epsilon filtering leaves
            # one point. CV takes priority, epsilon eligibility breaks ties,
            # and the existing dual-space diversity orders the remaining pool.
            objective_crowding = crowding_distance(merged_fit2, torch.ones_like(keep_mask))
            decision_crowding = self._harmonic_crowding(merged_pop2)
            selected = lexsort(constraint_keys([-(objective_crowding + decision_crowding), (~keep_mask).float()], merged_cv2))[
                : self.pop_size
            ]
            self.pop2 = merged_pop2[selected]
            self.fit2 = merged_fit2[selected]
            self.cv2 = merged_cv2[selected]
            self.spea2_fit2 = self._cal_spea2_fitness(self.pop2, self.fit2, local_niche=True)
            return

        # Pruning if count > N
        survivor_pop = merged_pop2[keep_mask]
        survivor_fit = merged_fit2[keep_mask]

        # Calculate dual-space crowding
        obj_crowd = crowding_distance(survivor_fit, torch.ones(survivor_fit.shape[0], dtype=torch.bool, device=self.lb.device))
        dec_crowd = self._harmonic_crowding(survivor_pop)
        total_crowd = obj_crowd + dec_crowd

        # Bug #25: Lexsort primary key last
        survivor_cv = take_violation(merged_cv2, keep_mask)
        indices = lexsort(constraint_keys([-total_crowd], survivor_cv))
        final_idx = indices[: self.pop_size]

        self.pop2 = survivor_pop[final_idx]
        self.fit2 = survivor_fit[final_idx]
        self.cv2 = take_violation(survivor_cv, final_idx)
        self.spea2_fit2 = self._cal_spea2_fitness(self.pop2, self.fit2, local_niche=True)


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = CoMMEA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    # 1. Trigger JIT compilation (First step)
    jit_state_step()

    # 2. Pure execution (Remaining 49 steps)
    torch.cuda.synchronize()
    exec_start = time.perf_counter()

    for i in range(1, 50):
        jit_state_step()

        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            # Simple NaN filtering for metric calculation
            fit = fit[~torch.any(torch.isnan(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")
