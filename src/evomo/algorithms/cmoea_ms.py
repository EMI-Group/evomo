import torch
import torch.nn.functional as F
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp


class CMOEA_MS(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, type: int = 1, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        # Rename 'type' to 'op_type' to avoid collision with torch.nn.Module.type
        self.op_type = Parameter(torch.tensor(type, dtype=torch.int32, device=device))
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.fitness = Mutable(torch.zeros(pop_size, device=device))
        self.iter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def _cal_sde(self, norm_fit: torch.Tensor) -> torch.Tensor:
        N_total = norm_fit.shape[0]
        P_i = norm_fit.unsqueeze(1)
        P_j = norm_fit.unsqueeze(0)
        S = torch.max(P_j, P_i)

        dist = 1.0 - F.cosine_similarity(P_i, S, dim=-1)

        mask = torch.eye(N_total, device=norm_fit.device).bool()
        dist = torch.where(mask, torch.full_like(dist, float("inf")), dist)

        # k = floor(sqrt(N))
        k = int(torch.sqrt(torch.tensor(N_total, dtype=torch.float32)))
        sorted_dist, _ = torch.sort(dist, dim=1)
        # Use k-1 for 0-based indexing
        sde = 1.0 / (sorted_dist[:, k - 1] + 2.0)
        return sde

    def _cal_fitness(self, norm_fit: torch.Tensor) -> torch.Tensor:
        X = norm_fit.unsqueeze(1)
        Y = norm_fit.unsqueeze(0)
        dom_mat = (X <= Y).all(dim=-1) & (X < Y).any(dim=-1)

        S = dom_mat.sum(dim=1)
        R = dom_mat.t().float() @ S.float()
        sde = self._cal_sde(norm_fit)

        return R + sde

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        f_min = self.fit.min(dim=0, keepdim=True)[0]
        f_max = self.fit.max(dim=0, keepdim=True)[0]
        norm_fit = (self.fit - f_min) / (f_max - f_min + 1e-6)
        self.fitness = self._cal_fitness(norm_fit)

    def step(self) -> None:
        self.iter = self.iter + 1
        device = self.pop.device
        N = self.pop_size

        mating_idx = tournament_selection_multifit(N, [self.fitness], tournament_size=2)
        parents = self.pop[mating_idx]

        if self.op_type == 1:
            off_pop = simulated_binary(parents, pro_c=1.0, dis_c=20.0)
            off_pop = polynomial_mutation(off_pop, self.lb, self.ub, pro_m=1.0 / self.lb.numel(), dis_m=20.0)
        else:
            # DE logic: current-to-best style or similar
            # Using parents as a base for variation
            off_pop = parents + 0.5 * (
                self.pop[torch.randint(0, N, (N,), device=device)] - self.pop[torch.randint(0, N, (N,), device=device)]
            )

        off_pop = clamp(off_pop, self.lb, self.ub)
        off_fit = self.evaluate(off_pop)

        Q_pop = torch.cat([self.pop, off_pop], dim=0)
        Q_fit = torch.cat([self.fit, off_fit], dim=0)

        f_min = Q_fit.min(dim=0, keepdim=True)[0]
        f_max = Q_fit.max(dim=0, keepdim=True)[0]
        norm_fit = (Q_fit - f_min) / (f_max - f_min + 1e-6)

        fitness_total = self._cal_fitness(norm_fit)

        mask_nd = fitness_total < 1.0
        num_nd = torch.sum(mask_nd.int())

        # Unified Selection Logic
        sorted_fit_val, sorted_idx = torch.sort(fitness_total)
        is_peeling = num_nd > N

        # Peeling Logic
        P_i = norm_fit.unsqueeze(1)
        P_j = norm_fit.unsqueeze(0)
        dist_mat = 1.0 - F.cosine_similarity(P_i, P_j, dim=-1)
        diag_mask = torch.eye(2 * N, device=device).bool()
        dist_mat = torch.where(diag_mask, torch.full_like(dist_mat, float("inf")), dist_mat)

        curr_mask = mask_nd.clone()
        curr_num = num_nd

        # Run loop for max possible removals (N)
        for _ in range(N):
            active = (curr_num > N) & is_peeling

            # Distance only between survivors
            temp_dist = torch.where(
                curr_mask.unsqueeze(0) & curr_mask.unsqueeze(1), dist_mat, torch.full_like(dist_mat, float("inf"))
            )

            # Find individual with minimum distance to any other survivor
            min_dists, _ = torch.min(temp_dist, dim=1)
            min_dists = torch.where(curr_mask, min_dists, torch.full_like(min_dists, float("inf")))
            idx_to_remove = torch.argmin(min_dists)

            # Update
            curr_mask[idx_to_remove] = torch.where(active, torch.tensor(False, device=device), curr_mask[idx_to_remove])
            curr_num = torch.where(active, curr_num - 1, curr_num)

        # Final Selection
        # If peeling: take the N survivors from curr_mask
        # If not peeling: take top N from sorted_idx
        if_peeling_indices = torch.argsort(curr_mask.int(), descending=True)[:N]
        if_not_peeling_indices = sorted_idx[:N]

        survivor_idx = torch.where(is_peeling, if_peeling_indices, if_not_peeling_indices)

        self.pop = Q_pop[survivor_idx]
        self.fit = Q_fit[survivor_idx]
        self.fitness = fitness_total[survivor_idx]


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    # CMOEA_MS must be replaced by your actual class name
    algo = CMOEA_MS(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
