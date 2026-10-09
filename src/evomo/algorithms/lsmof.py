import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    rank_with_constraints,
    take_violation,
)
from evomo.utils import parse_evaluate, register_lazy_buffer, unique_rows_sorted


class LSMOF(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize two-phase weight-space optimization and NSGA-II refinement.

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
        :param kwargs: Default: ``{}``. Optional settings: ``wD=5`` (reference solutions), ``SubN=20`` (weight
            population size), ``wmax=0.1`` (decision reconstruction scale), and ``max_fe=10000`` (evaluation horizon).
            Other keywords are ignored. The full meanings and limits are listed below.
        :type kwargs: dict

        .. rubric:: Additional keyword settings

        ``wD`` (int, default ``5``)
            Reference solution count for building directions from both bounds. Use a positive value no larger than the
            available distinct population. The reconstruction assumes that exactly this many reference solutions are
            selected.

        ``SubN`` (int, default ``20``)
            Positive weight-population size. The first phase evaluates ``2 * wD * SubN`` reconstructed decision vectors
            per step.

        ``wmax`` (float, default ``0.1``)
            Positive scale multiplying weighted unit-direction offsets in decision reconstruction. This is a decision-
            space step scale, not a fraction of the evaluation budget.

        ``max_fe`` (int, default ``10000``)
            Positive evaluation horizon. The second phase begins when the evaluation counter reaches ``max_fe // 2``;
            the initial evaluations are included. Termination remains controlled by the caller.


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
        self.D = lb.numel()

        # LSMOF Specific Parameters
        self.wD = kwargs.get("wD", 5)  # Number of reference solutions
        self.SubN = kwargs.get("SubN", 20)  # Sub-population for weight optimization
        self.wmax = Parameter(torch.tensor(kwargs.get("wmax", 0.1), device=device))
        self.max_fe = kwargs.get("max_fe", 10000)
        self.switch_fe = self.max_fe // 2

        # Initialize State
        self.pop = Mutable(torch.rand(pop_size, self.D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        self.archive_pop = Mutable(self.pop.clone())
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        self.fe_counter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        self.rank = Mutable(torch.full((pop_size,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device))
        self.dis = Mutable(torch.full((pop_size,), -torch.inf, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        register_lazy_buffer(self, "archive_cv", device_like="pop")

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        self.fe_counter = self.fe_counter + self.pop_size
        self.archive_fit = self.fit.clone()
        self.archive_cv = None if self.cv is None else self.cv.clone()
        self.archive_pop = self.pop.clone()
        if self.cv is None:
            _, _, self.rank, self.dis = self._environmental_selection(self.pop, self.fit, self.pop_size)
        else:
            self.pop, self.fit, self.rank, self.dis = self._environmental_selection(self.pop, self.fit, self.pop_size, self.cv)

    def _reconstruct_decisions(
        self, weights: torch.Tensor, directions: torch.Tensor, base: torch.Tensor, wmax: torch.Tensor
    ) -> torch.Tensor:
        # weights: (SubN, 2*wD), directions: (2*wD, D), base: (2*wD, D)
        # Broadcasting: (SubN, 2*wD, 1) * (1, 2*wD, D) -> (SubN, 2*wD, D)
        offset = torch.einsum("sw,wd->swd", weights, directions) * wmax
        dec = base.unsqueeze(0) + offset
        return dec.reshape(-1, self.D)

    def _environmental_selection(self, pop, fit, N, cv=None):
        combined_pop, combined_idx = unique_rows_sorted(pop)
        combined_fit = fit[combined_idx]

        combined_cv = take_violation(cv, combined_idx)
        rank = rank_with_constraints(combined_fit, combined_cv)
        N_total = combined_fit.shape[0]
        mask = torch.zeros(N_total, dtype=torch.bool, device=pop.device)
        distances = torch.full((N_total,), -1.0, device=pop.device)

        num_selected = 0

        # Peeling Loop
        for i in range(N_total):
            curr_front_mask = rank == i
            count = torch.sum(curr_front_mask.int())

            is_empty = count == 0
            is_done = num_selected >= N

            if not is_done and not is_empty:
                if num_selected + count <= N:
                    mask |= curr_front_mask
                    distances[curr_front_mask] = crowding_distance(combined_fit, curr_front_mask)[curr_front_mask]
                    num_selected += count
                else:
                    dist = crowding_distance(combined_fit, curr_front_mask)
                    idx_in_front = torch.where(curr_front_mask)[0]
                    sorted_sub_idx = torch.argsort(dist[curr_front_mask], descending=True)
                    selected_indices = idx_in_front[sorted_sub_idx[: N - num_selected]]
                    mask[selected_indices] = True
                    distances[selected_indices] = dist[selected_indices]
                    num_selected = N

        if N == self.pop_size:
            self.cv = take_violation(combined_cv, mask)
        survivor_pop = combined_pop[mask]
        survivor_fit = combined_fit[mask]
        survivor_rank = rank[mask]
        survivor_dis = distances[mask]
        return survivor_pop, survivor_fit, survivor_rank, survivor_dis

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        device = self.pop.device

        if self.fe_counter < self.switch_fe:
            # Phase A: Bi-directional Weight Optimization
            # 1. Reference Selection
            ref_pop, _, _, _ = self._environmental_selection(self.pop, self.fit, self.wD, self.cv)

            # 2. Direction Matrix
            Direct_L = (ref_pop - self.lb) / (torch.norm(ref_pop - self.lb, dim=1, keepdim=True) + 1e-6)
            Direct_U = (ref_pop - self.ub) / (torch.norm(ref_pop - self.ub, dim=1, keepdim=True) + 1e-6)
            Direct = torch.cat([Direct_L, Direct_U], dim=0)  # (2*wD, D)
            Base = torch.cat([self.lb.repeat(self.wD, 1), self.ub.repeat(self.wD, 1)], dim=0)  # (2*wD, D)

            # 3. Internal DE for Weights
            weights = torch.rand(self.SubN, 2 * self.wD, device=device)
            decisions = self._reconstruct_decisions(weights, Direct, Base, self.wmax)
            decisions = clamp(decisions, self.lb, self.ub)
            off_fit, off_cv = parse_evaluate(self.evaluate(decisions))
            self.fe_counter = self.fe_counter + decisions.shape[0]

            # Update Archive and Pop
            merged_pop = torch.cat([self.pop, decisions], dim=0)
            merged_fit = torch.cat([self.fit, off_fit], dim=0)
            self.pop, self.fit, self.rank, self.dis = self._environmental_selection(
                merged_pop, merged_fit, self.pop_size, cat_violation(self.cv, off_cv)
            )

        else:
            # Phase B: Standard NSGA-II Refinement
            mating_pool = tournament_selection_multifit(self.pop_size, [-self.dis, self.rank.float()], tournament_size=2)
            parents = self.pop[mating_pool]
            offspring = simulated_binary(parents)
            offspring = polynomial_mutation(offspring, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)

            off_fit, off_cv = parse_evaluate(self.evaluate(offspring))
            self.fe_counter = self.fe_counter + self.pop_size

            merged_pop = torch.cat([self.pop, offspring], dim=0)
            merged_fit = torch.cat([self.fit, off_fit], dim=0)
            self.pop, self.fit, self.rank, self.dis = self._environmental_selection(
                merged_pop, merged_fit, self.pop_size, cat_violation(self.cv, off_cv)
            )


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = LSMOF(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
