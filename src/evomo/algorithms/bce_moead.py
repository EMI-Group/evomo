import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, randint

from evomo.operators.selection import nd_environmental_selection, non_dominate_rank
from evomo.utils import unique_rows_sorted


class BCEMOEAD(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, T: int = 20, nr: int = 2, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.T = T
        self.nr = nr

        # 1. Weights & Neighbors (Bug #13, #19)
        W, n_actual = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_actual
        self.T = min(T, self.pop_size)
        W = W.to(device)
        self.W = Mutable(W)

        dist = torch.cdist(W, W)
        self.B = Mutable(torch.topk(dist, self.T, largest=False).indices)

        # 2. Populations (PC and NPC)
        self.pop = Mutable(torch.rand(self.pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

        self.npc_pop = Mutable(torch.rand(self.pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.npc_fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

        self.z = Mutable(torch.zeros((1, n_objs), device=device))
        self.nND = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.npc_fit = self.evaluate(self.npc_pop)
        self.z = torch.min(self.npc_fit, dim=0, keepdim=True).values

        # Initial PC update
        rank = non_dominate_rank(self.fit)
        self.nND = torch.sum(rank == 1)

    def _cal_tchebycheff(self, fit: torch.Tensor, z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # Bug #12: Safe division
        return torch.max(torch.abs(fit - z) / (W + 1e-6), dim=-1).values

    def _niche_deletion(self, fit: torch.Tensor, pop: torch.Tensor, r: torch.Tensor, N: int):
        # Bug #41: JIT compliant peeling loop
        f_min = torch.min(fit, dim=0).values
        f_max = torch.max(fit, dim=0).values
        norm_fit = (fit - f_min) / (f_max - f_min + 1e-6)

        dist_matrix = torch.cdist(norm_fit, norm_fit)
        R = torch.clamp(dist_matrix / (r + 1e-6), max=1.0)

        active_mask = torch.ones(fit.shape[0], dtype=torch.bool, device=fit.device)

        # Peeling Loop
        for _ in range(fit.shape[0] - N):
            # Calculate density only for active members
            # R_active: [Current_N, Current_N]
            R_curr = R[active_mask][:, active_mask]
            density = 1 - torch.prod(R_curr, dim=1)

            # Map local argmax back to global index
            local_idx = torch.argmax(density)
            global_indices = torch.where(active_mask)[0]
            idx_to_remove = global_indices[local_idx]
            active_mask[idx_to_remove] = False

        return pop[active_mask], fit[active_mask]

    def step(self) -> None:
        device = self.lb.device
        N = self.pop_size

        # --- Sub-step 2.1: Exploration Trigger ---
        f_min = torch.min(self.fit, dim=0).values
        f_max = torch.max(self.fit, dim=0).values
        norm_pc = (self.fit - f_min) / (f_max - f_min + 1e-6)
        norm_npc = (self.npc_fit - f_min) / (f_max - f_min + 1e-6)

        dist_pc = torch.cdist(norm_pc, norm_pc)
        # r0 is mean of 4th nearest neighbor distances
        r0 = torch.topk(dist_pc, 4, largest=False).values[:, 3].mean()
        r = (self.nND.float() / N) * r0

        dist_pc_npc = torch.cdist(norm_pc, norm_npc)
        neighbor_count = (dist_pc_npc <= r).sum(dim=1)
        exploration_mask = neighbor_count <= 1

        # Variation on isolated individuals
        # If no isolated, use full pop to maintain batch size for operators
        mating_pop = torch.where(exploration_mask.any(), exploration_mask, torch.ones_like(exploration_mask))
        offspring = simulated_binary(self.pop[mating_pop], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        # --- Sub-step 2.2: NPC Update (Global) ---
        self.z = torch.min(torch.cat([self.z, off_fit], dim=0), dim=0, keepdim=True).values

        # Tchebycheff for offspring vs all NPC weights
        # off_fit: [N_off, M], z: [1, M], W: [N, M]
        # diff: [N_off, 1, M]
        diff = torch.abs(off_fit.unsqueeze(1) - self.z.unsqueeze(0))
        g_off = torch.max(diff / (self.W.unsqueeze(0) + 1e-6), dim=-1).values  # [N_off, N]

        # Current NPC Tchebycheff
        g_old = self._cal_tchebycheff(self.npc_fit, self.z, self.W)  # [N]

        # Vectorized replacement (simplified: each offspring tries to replace its best matching weight)
        # For each weight, find the best offspring
        best_off_idx = torch.argmin(g_off, dim=0)  # [N]
        best_g_val = torch.gather(g_off, 0, best_off_idx.unsqueeze(0)).squeeze(0)

        replace_mask = best_g_val < g_old
        self.npc_pop[replace_mask] = offspring[best_off_idx[replace_mask]]
        self.npc_fit[replace_mask] = off_fit[best_off_idx[replace_mask]]

        # --- Sub-step 2.3: NPC Update (Neighborhood) ---
        # Standard MOEA/D neighborhood update for each offspring
        for i in range(offspring.shape[0]):
            indices = self.B[randint(0, N, (1,), device=device).item()]  # Pick a random neighborhood
            g_neigh_old = self._cal_tchebycheff(self.npc_fit[indices], self.z, self.W[indices])
            g_neigh_off = self._cal_tchebycheff(off_fit[i].unsqueeze(0), self.z, self.W[indices])

            replace_neigh = g_neigh_off < g_neigh_old
            # Limit replacements to nr (Bug #29)
            count_mask = torch.cumsum(replace_neigh.long(), dim=0) <= self.nr
            final_replace = replace_neigh & count_mask

            self.npc_pop[indices[final_replace]] = offspring[i]
            self.npc_fit[indices[final_replace]] = off_fit[i]

        # --- Sub-step 2.4: PC Selection ---
        combined_pop = torch.cat([self.pop, offspring, self.npc_pop], dim=0)
        combined_fit = torch.cat([self.fit, off_fit, self.npc_fit], dim=0)

        # Unique rows (Bug #3)
        combined_pop, u_idx = unique_rows_sorted(combined_pop)
        combined_fit = combined_fit[u_idx]

        rank = non_dominate_rank(combined_fit)
        is_nd = rank == 1
        nd_pop = combined_pop[is_nd]
        nd_fit = combined_fit[is_nd]

        self.nND = torch.sum(is_nd).to(torch.int32)

        if nd_pop.shape[0] <= N:
            # Fill with best from other ranks if needed
            self.pop, self.fit, _, _, _ = nd_environmental_selection(combined_pop, combined_fit, N)
        else:
            # Niche-based deletion
            self.pop, self.fit = self._niche_deletion(nd_fit, nd_pop, r, N)


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = BCEMOEAD(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
