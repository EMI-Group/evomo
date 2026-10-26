import math

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    rank_with_constraints,
    take_violation,
)
from evomo.utils import parse_evaluate, register_lazy_buffer


class VaEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the VaEA population and optimization state.

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
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)  # [N,D]
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))  # [N,M]

        register_lazy_buffer(self, "cv", device_like="pop")

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))  # [N,M]

    def _cosine_distance_matrix(self, x: torch.Tensor) -> torch.Tensor:
        # [Bug #12] Safe Division & [Bug #29] Vectorization
        norm = torch.norm(x, dim=1, keepdim=True)
        similarity = (x @ x.t()) / (norm @ norm.t() + 1e-6)
        # Clamp to prevent NaNs in acos
        return torch.acos(similarity.clamp(-1 + 1e-7, 1 - 1e-7))

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        device = self.pop.device
        N = self.pop_size
        M = self.n_objs

        # 1. Mating: VaEA selects the parents uniformly at random
        mating_pool = torch.randint(0, N, (N,), device=device)
        parents = self.pop[mating_pool]

        offspring = simulated_binary(parents, pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit, off_cv = parse_evaluate(self.evaluate(offspring))

        # 3. Environmental Selection
        C_pop = torch.cat([self.pop, offspring], dim=0)
        C_fit = torch.cat([self.fit, off_fit], dim=0)

        # Fronts before the last front F_k are kept; F_k is filled by niching
        C_cv = cat_violation(self.cv, off_cv)
        rank = rank_with_constraints(C_fit, C_cv)
        last_rank = torch.sort(rank).values[N - 1]
        kept = torch.where(rank < last_rank)[0]
        last = torch.where(rank == last_rank)[0]
        cand = torch.cat([kept, last])

        # Normalization over S_t = F_1 u ... u F_k
        S_fit = C_fit[cand]
        z_min = torch.min(S_fit, dim=0).values
        z_max = torch.max(S_fit, dim=0).values
        norm_fit = (S_fit - z_min) / (z_max - z_min + 1e-6)
        conv = torch.sum(norm_fit, dim=1)
        angle_mat = self._cosine_distance_matrix(norm_fit)

        choose = torch.zeros(cand.numel(), dtype=torch.bool, device=device)
        choose[: kept.numel()] = True
        if kept.numel() == 0:
            # The solution with the smallest angle to each axis, then the M best converged solutions
            unit = S_fit / (torch.norm(S_fit, dim=1, keepdim=True) + 1e-6)
            choose[torch.argmax(unit, dim=0)] = True
            choose[torch.argsort(conv)[: min(M, cand.numel())]] = True

        threshold = math.pi / (2 * (N + 1))
        while int(choose.sum()) < N:
            select_idx = torch.where(choose)[0]
            remain_idx = torch.where(~choose)[0]
            # Maximum-vector-angle-first: add the remaining solution with the largest angle to the selected ones
            rho = torch.argmax(torch.min(angle_mat[remain_idx][:, select_idx], dim=1).values)
            choose[remain_idx[rho]] = True
            # Worse elimination: the remaining solution with the smallest angle to the selected ones replaces
            # its nearest selected solution if they are within the threshold and it converges better
            remain_idx = torch.cat([remain_idx[:rho], remain_idx[rho + 1 :]])
            if remain_idx.numel() > 0:
                select_idx = torch.where(choose)[0]
                min_angles, nearest = torch.min(angle_mat[remain_idx][:, select_idx], dim=1)
                mu = torch.argmin(min_angles)
                r = select_idx[nearest[mu]]
                same_front = C_cv is None or rank[cand[r]] == rank[cand[remain_idx[mu]]]
                if min_angles[mu] < threshold and conv[r] > conv[remain_idx[mu]] and same_front:
                    choose[r] = False
                    choose[remain_idx[mu]] = True

        final_idx = cand[choose]
        self.pop = C_pop[final_idx]
        self.fit = C_fit[final_idx]
        self.cv = take_violation(C_cv, final_idx)


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = VaEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
