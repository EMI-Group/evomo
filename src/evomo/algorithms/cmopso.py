import torch
from evox.core import Algorithm, Mutable
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, lexsort

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    prefer_by_constraint,
    rank_with_constraints,
    total_violation,
)
from evomo.operators.selection.distance_truncation import crowding_distance_by_rank, distance_truncation
from evomo.utils import parse_evaluate, register_lazy_buffer


class CMOPSO(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the CMOPSO population and optimization state.

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
        self.v = Mutable(torch.zeros((pop_size, D), device=device))  # [N,D] Persistent Velocity

        register_lazy_buffer(self, "cv", device_like="pop")

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        device = self.pop.device
        N = self.pop_size
        D = self.lb.numel()

        # 1. Leader Selection (Mating)
        rank = rank_with_constraints(self.fit, self.cv)
        # Crowding distance is calculated per front.
        cd = crowding_distance_by_rank(self.fit, rank)

        # Sort to find top 10 leaders (Bug #25)
        indices = lexsort(torch.stack([-cd, rank.float()]))
        LeaderSetIdx = indices[:10]

        # 2. Tournament & Angle-Based Competition
        # Ensure we have 10 leaders to pick from; if not, pad with 0
        num_leaders = LeaderSetIdx.shape[0]
        first = torch.randint(0, num_leaders, (N,), device=device)
        second = (first + torch.randint(1, max(2, num_leaders), (N,), device=device)) % num_leaders
        c1_idx, c2_idx = LeaderSetIdx[first], LeaderSetIdx[second]

        C1_fit = self.fit[c1_idx]
        C2_fit = self.fit[c2_idx]

        # Cosine Similarity (Bug #12: Safe Division)
        norm_pop = torch.norm(self.fit, dim=-1)
        norm_c1 = torch.norm(C1_fit, dim=-1)
        norm_c2 = torch.norm(C2_fit, dim=-1)

        cos1 = (self.fit * C1_fit).sum(-1) / (norm_pop * norm_c1 + 1e-6)
        cos2 = (self.fit * C2_fit).sum(-1) / (norm_pop * norm_c2 + 1e-6)

        # Winner selection (Bug #41: torch.where)
        # PlatEMO keeps the first leader when the angles are equal.
        winner_mask = cos1 >= cos2
        if self.cv is not None:
            total = total_violation(self.cv)
            winner_mask = prefer_by_constraint(winner_mask, total[c1_idx], total[c2_idx])
        winner_idx = torch.where(winner_mask, c1_idx, c2_idx)

        # 3. Velocity and Position Update
        r1 = torch.rand((N, D), device=device)
        r2 = torch.rand((N, D), device=device)

        off_v = r1 * self.v + r2 * (self.pop[winner_idx] - self.pop)
        off_pop = self.pop + off_v

        # Boundary Clamping (Bug #38)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # Polynomial Mutation
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub, pro_m=1.0, dis_m=20.0)

        # 4. Evaluation
        off_fit, off_cv = parse_evaluate(self.evaluate(off_pop))

        # 5. Environmental Selection
        X = torch.cat([self.pop, off_pop], dim=0)
        F = torch.cat([self.fit, off_fit], dim=0)
        V_all = torch.cat([self.v, off_v], dim=0)

        CV_all = cat_violation(self.cv, off_cv)
        all_rank = rank_with_constraints(F, CV_all)

        # Vectorized MaxFNo identification (Bug #41)
        # Max possible rank is 2*N
        rank_bins = torch.arange(2 * N + 1, device=device).view(-1, 1)
        counts = (all_rank == rank_bins).sum(dim=1).float()
        cum_counts = torch.cumsum(counts, dim=0)

        # Find the first front index where cum_sum >= N
        front_overflow_mask = cum_counts >= N
        MaxFNo = torch.nonzero(front_overflow_mask)[0, 0]

        # Individuals in fronts < MaxFNo
        keep_mask = all_rank < MaxFNo
        kept_indices = torch.where(keep_mask)[0]
        needed_count = N - kept_indices.numel()

        # Truncation for the last front (Bug #30)
        last_front_mask = all_rank == MaxFNo
        F_last = F[last_front_mask]
        X_last = X[last_front_mask]
        V_last = V_all[last_front_mask]

        # Normalization (Bug #12)
        f_min = F[all_rank == 0].min(0)[0]
        f_max = F[all_rank == 0].max(0)[0]
        norm_F = (F_last - f_min) / (f_max - f_min + 1e-6)

        # Distance Matrix calculation without in-place mutation
        dist_matrix = torch.cdist(norm_F, norm_F)
        sentinel_inf = 1e18
        # Use torch.eye to create a mask for the diagonal and set to sentinel_inf
        diag_mask = torch.eye(dist_matrix.shape[0], device=device, dtype=torch.bool)
        dist_matrix = torch.where(diag_mask, torch.tensor(sentinel_inf, device=device), dist_matrix)

        trunc_indices = distance_truncation(dist_matrix, needed_count)

        # Combine survivors
        survivor_pop = torch.cat([X[keep_mask], X_last[trunc_indices]], dim=0)
        survivor_fit = torch.cat([F[keep_mask], F_last[trunc_indices]], dim=0)
        survivor_v = torch.cat([V_all[keep_mask], V_last[trunc_indices]], dim=0)

        # Update State
        if CV_all is not None:
            self.cv = torch.cat([CV_all[keep_mask], CV_all[last_front_mask][trunc_indices]])[:N]
        self.pop = survivor_pop[:N]
        self.fit = survivor_fit[:N]
        self.v = survivor_v[:N]


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    # CMOPSO must be replaced by your actual class name
    algo = CMOPSO(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
