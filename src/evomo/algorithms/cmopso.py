import torch
from evox.core import Algorithm, Mutable
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class CMOPSO(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
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

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size
        D = self.lb.numel()

        # 1. Leader Selection (Mating)
        rank = non_dominate_rank(self.fit)
        # Crowding distance is calculated per front.
        mask_f1 = rank == 0
        cd = crowding_distance(self.fit, mask_f1)

        # Sort to find top 10 leaders (Bug #25)
        indices = lexsort(torch.stack([-cd, rank.float()]))
        LeaderSetIdx = indices[:10]

        # 2. Tournament & Angle-Based Competition
        # Ensure we have 10 leaders to pick from; if not, pad with 0
        num_leaders = LeaderSetIdx.shape[0]
        c1_idx = LeaderSetIdx[torch.randint(0, num_leaders, (N,), device=device)]
        c2_idx = LeaderSetIdx[torch.randint(0, num_leaders, (N,), device=device)]

        C1_fit = self.fit[c1_idx]
        C2_fit = self.fit[c2_idx]

        # Cosine Similarity (Bug #12: Safe Division)
        norm_pop = torch.norm(self.fit, dim=-1)
        norm_c1 = torch.norm(C1_fit, dim=-1)
        norm_c2 = torch.norm(C2_fit, dim=-1)

        cos1 = (self.fit * C1_fit).sum(-1) / (norm_pop * norm_c1 + 1e-6)
        cos2 = (self.fit * C2_fit).sum(-1) / (norm_pop * norm_c2 + 1e-6)

        # Winner selection (Bug #41: torch.where)
        winner_idx = torch.where(cos1 > cos2, c1_idx, c2_idx)

        # 3. Velocity and Position Update
        r1 = torch.rand((N, D), device=device)
        r2 = torch.rand((N, D), device=device)

        off_v = r1 * self.v + r2 * (self.pop[winner_idx] - self.pop)
        off_pop = self.pop + off_v

        # Boundary Clamping (Bug #38)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # Polynomial Mutation
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub, pro_m=1.0 / D, dis_m=20.0)

        # 4. Evaluation
        off_fit = self.evaluate(off_pop)

        # 5. Environmental Selection
        X = torch.cat([self.pop, off_pop], dim=0)
        F = torch.cat([self.fit, off_fit], dim=0)
        V_all = torch.cat([self.v, off_v], dim=0)

        all_rank = non_dominate_rank(F)

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
        num_kept = keep_mask.sum().to(torch.int32)
        needed_count = N - num_kept

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

        min_dists, _ = torch.min(dist_matrix, dim=1)
        # Sort by distance descending
        trunc_indices = torch.argsort(min_dists, descending=True)[:needed_count]

        # Combine survivors
        survivor_pop = torch.cat([X[keep_mask], X_last[trunc_indices]], dim=0)
        survivor_fit = torch.cat([F[keep_mask], F_last[trunc_indices]], dim=0)
        survivor_v = torch.cat([V_all[keep_mask], V_last[trunc_indices]], dim=0)

        # Update State
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
