import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, randint

from evomo.operators.selection import nd_environmental_selection, non_dominate_rank
from evomo.utils import unique_rows_sorted


class BCE_IBEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, kappa: float = 0.05, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.kappa = Parameter(torch.tensor(kappa, device=device))
        D = lb.numel()

        # Initialize State (Mutables)
        # PC: Pareto Population, NPC: Non-Pareto Population
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        self.npc_pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.npc_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.npc_fit = self.evaluate(self.npc_pop)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size

        # 1. PC Selection Logic (Niche Radius calculation)
        dist_pc = torch.cdist(self.fit, self.fit)
        sorted_dist, _ = torch.sort(dist_pc, dim=1)
        r = torch.mean(sorted_dist[:, 2])  # 3rd nearest neighbor

        # 2. Exploration Logic (Mating Selection for PC)
        dist_npc_pc = torch.cdist(self.npc_fit, self.fit)
        neighbors_count = torch.sum(dist_npc_pc < r, dim=1)  # Count NPC neighbors for each PC
        isolated_mask = neighbors_count <= 1

        # Generate NewPC from isolated PC individuals
        # If no isolated individuals, use all PC (Deadlock prevention via logic)
        pc_to_mate = torch.where(isolated_mask.any(), isolated_mask, torch.ones_like(isolated_mask))
        pc_indices = torch.where(pc_to_mate)[0]
        # Tournament selection for mating pool
        mating_idx = pc_indices[randint(0, pc_indices.numel(), (N,), device=device)]
        off_pc = self.mutation(self.crossover(self.pop[mating_idx]), self.lb, self.ub)
        off_pc_fit = self.evaluate(off_pc)

        # 3. NPC Mating (Standard IBEA style selection)
        npc_mating_idx = randint(0, N, (N,), device=device)
        off_npc = self.mutation(self.crossover(self.npc_pop[npc_mating_idx]), self.lb, self.ub)
        off_npc_fit = self.evaluate(off_npc)

        # 4. Environmental Selection for PC
        merged_pc_pop = torch.cat([self.pop, off_pc, off_npc], dim=0)
        merged_pc_fit = torch.cat([self.fit, off_pc_fit, off_npc_fit], dim=0)
        u_pc_pop, u_pc_idx = unique_rows_sorted(merged_pc_pop)
        u_pc_fit = merged_pc_fit[u_pc_idx]
        self.pop, self.fit = _pc_selection(u_pc_pop, u_pc_fit, N)

        # 5. Environmental Selection for NPC
        merged_npc_pop = torch.cat([self.npc_pop, off_pc, off_npc], dim=0)
        merged_npc_fit = torch.cat([self.npc_fit, off_pc_fit, off_npc_fit], dim=0)
        u_npc_pop, u_npc_idx = unique_rows_sorted(merged_npc_pop)
        u_npc_fit = merged_npc_fit[u_npc_idx]
        self.npc_pop, self.npc_fit = _ibea_selection(u_npc_pop, u_npc_fit, N, self.kappa)

    def crossover(self, x):
        return simulated_binary(x)

    def mutation(self, x, lb, ub):
        return clamp(polynomial_mutation(x, lb, ub), lb, ub)


def _ibea_selection(pop, fit, K, kappa):
    N = pop.shape[0]
    f_min = torch.min(fit, dim=0)[0]
    f_max = torch.max(fit, dim=0)[0]
    norm_objs = (fit - f_min) / (f_max - f_min + 1e-6)

    # Indicator Matrix (Additive epsilon-indicator)
    diff = norm_objs.unsqueeze(1) - norm_objs.unsqueeze(0)
    indicator = torch.max(diff, dim=-1)[0]
    C = torch.max(torch.abs(indicator))

    # Initial Fitness
    fitness = torch.sum(-torch.exp(-indicator / (C * kappa + 1e-6)), dim=0)

    # Peeling
    mask = torch.ones(N, dtype=torch.bool, device=pop.device)
    sentinel = torch.finfo(torch.float32).max

    for _ in range(N - K):
        # Find worst among active
        temp_fit = torch.where(mask, fitness, sentinel)
        worst = torch.argmin(temp_fit)
        mask[worst] = False
        # Update fitness of remaining
        fitness += torch.exp(-indicator[worst, :] / (C * kappa + 1e-6))

    return pop[mask], fit[mask]


def _pc_selection(pop, fit, K):
    # 1. Non-dominated Filtering
    rank = non_dominate_rank(fit)
    mask_rank1 = rank == 0

    # If rank 1 is not enough, take more from subsequent ranks
    if torch.sum(mask_rank1) <= K:
        # Standard ND selection if rank 1 is small
        new_pop, new_fit, _, _, _ = nd_environmental_selection(pop, fit, K)
        return new_pop, new_fit

    # Pruning Rank 1
    p_pop = pop[mask_rank1]
    p_fit = fit[mask_rank1]
    N_p = p_pop.shape[0]

    dist_matrix = torch.cdist(p_fit, p_fit)
    sorted_dist, _ = torch.sort(dist_matrix, dim=1)
    r = torch.mean(sorted_dist[:, 2])

    R = torch.clamp(dist_matrix / (r + 1e-6), max=1.0)
    P = torch.prod(R + 1e-6, dim=1)

    active_mask = torch.ones(N_p, dtype=torch.bool, device=pop.device)
    for _ in range(N_p - K):
        temp_P = torch.where(active_mask, P, torch.finfo(torch.float32).max)
        worst = torch.argmin(temp_P)
        active_mask[worst] = False
        P = P / (R[worst, :] + 1e-6)

    return p_pop[active_mask], p_fit[active_mask]


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = BCE_IBEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
