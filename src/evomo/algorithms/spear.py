import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class SPEAR(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Reference Directions - uniform_sampling returns (actual_N, M)
        w, actual_n = uniform_sampling(pop_size, n_objs)
        self.pop_size = actual_n
        self.n_objs = n_objs

        self.w = Mutable(w.to(device))

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

        # Niche Radius (self.theta)
        cos_sim = self.w @ self.w.T
        angles = torch.acos(torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6))
        # Bug #1: Use a mask instead of in-place fill_diagonal_ to avoid JIT mutation errors
        mask_diag = torch.eye(self.pop_size, device=device, dtype=torch.bool)
        angles_masked = torch.where(mask_diag, torch.tensor(float("inf"), device=device), angles)
        self.theta = Mutable(torch.max(torch.min(angles_masked, dim=1).values))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size

        # 1. Mating Selection (Nearest neighbor in objective space)
        dist_mat = torch.cdist(self.fit, self.fit)
        # Fix: Avoid in-place fill_diagonal_ which crashes torch.compile
        mask_diag = torch.eye(N, device=device, dtype=torch.bool)
        dist_mat = torch.where(mask_diag, torch.tensor(float("inf"), device=device), dist_mat)

        candidates = torch.randint(0, N, (N, 20), device=device)
        cand_dists = torch.gather(dist_mat, 1, candidates)
        best_idx_in_cand = torch.argmin(cand_dists, dim=1)
        mating_pool_idx = torch.gather(candidates, 1, best_idx_in_cand.unsqueeze(1)).squeeze()

        # 2. Variation
        # GAhalf logic: use pop and mating pool
        crossovered = simulated_binary(torch.cat([self.pop, self.pop[mating_pool_idx]], dim=0))
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        # 3. Environmental Selection
        joint_pop = torch.cat([self.pop, offspring], dim=0)
        joint_fit = torch.cat([self.fit, off_fit], dim=0)

        # 3.1 Normalization (Rank-1 Based)
        rank = non_dominate_rank(joint_fit)
        front1_fit = joint_fit[rank == 1]

        # Bug #18: Normalization Trigger
        f_min = torch.min(front1_fit, dim=0).values
        f_max = torch.max(front1_fit, dim=0).values
        f_range = f_max - f_min

        # Global condition check
        if 0.05 * torch.max(f_range) < torch.min(f_range):
            norm_fit = (joint_fit - f_min) / (f_range + 1e-6)
        else:
            norm_fit = joint_fit

        # 3.2 Association & Density
        norm_w = self.w / (torch.norm(self.w, dim=1, keepdim=True) + 1e-6)
        norm_f = norm_fit / (torch.norm(norm_fit, dim=1, keepdim=True) + 1e-6)
        cos_sim_assoc = norm_f @ norm_w.T
        angles_assoc = torch.acos(torch.clamp(cos_sim_assoc, -1 + 1e-6, 1 - 1e-6))
        min_angles, ei = torch.min(angles_assoc, dim=1)
        density = min_angles / (min_angles + self.theta)

        # 3.3 Strength Pareto Fitness (Dual-Level)
        f = joint_fit
        # Bug #24: Dominance check
        dom = (f.unsqueeze(1) <= f.unsqueeze(0)).all(-1) & (f.unsqueeze(1) < f.unsqueeze(0)).any(-1)

        # Global Strength & Raw Fitness (Bug #23)
        Sg = dom.sum(dim=1)
        Rg = dom.T.float() @ Sg.float()

        # Local Strength & Raw Fitness
        niche_mask = ei.unsqueeze(1) == ei.unsqueeze(0)
        Sl = (dom & niche_mask).sum(dim=1)
        Rl = (dom & niche_mask).T.float() @ Sl.float()

        # Final FV
        # Count occurrences of each niche in the joint population
        niche_counts = torch.bincount(ei, minlength=N)[ei]
        fv = torch.where(niche_counts > 1, Rl + density + Rg, Rl + density)

        # 3.4 Selection Strategy (Niche-Clearing Peeling)
        choose = torch.zeros(joint_fit.shape[0], dtype=torch.bool, device=device)
        remaining = torch.ones(joint_fit.shape[0], dtype=torch.bool, device=device)

        # Peeling Loop (Bug #41: JIT compliant loop)
        while torch.sum(choose) < N:
            active_indices = torch.where(remaining)[0]
            active_ei = ei[remaining]
            active_fv = fv[remaining]

            # Find best in each unique niche
            winners = self._get_niche_winners(active_ei, active_fv, active_indices, N)

            # Filter valid winners (where niche index was present)
            valid_winner_mask = winners > -1
            valid_winners = winners[valid_winner_mask]

            num_needed = N - torch.sum(choose)
            num_winners = torch.sum(valid_winner_mask)

            # Deadlock Breaker (Bug #9)
            if num_winners == 0:
                rem_fv = fv.clone()
                rem_fv[~remaining] = float("inf")
                # Ensure k is at most the number of remaining elements
                k_val = torch.minimum(num_needed, torch.sum(remaining))
                _, top_k_idx = torch.topk(rem_fv, k=k_val.to(torch.long), largest=False)
                choose[top_k_idx] = True
                remaining[top_k_idx] = False
            else:
                # Select winners (up to num_needed)
                if num_winners > num_needed:
                    # If too many winners, sort them by FV to pick best
                    winner_fvs = fv[valid_winners]
                    _, sub_idx = torch.topk(winner_fvs, k=num_needed.to(torch.long), largest=False)
                    valid_winners = valid_winners[sub_idx]

                choose[valid_winners] = True
                remaining[valid_winners] = False

        self.pop = joint_pop[choose]
        self.fit = joint_fit[choose]

    def _get_niche_winners(
        self, active_ei: torch.Tensor, active_fv: torch.Tensor, active_indices: torch.Tensor, N: int
    ) -> torch.Tensor:
        device = active_ei.device
        winners = torch.full((N,), -1, dtype=torch.long, device=device)

        if active_ei.shape[0] == 0:
            return winners

        # Sort by niche ID then by FV (Bug #25)
        # Primary key (FV) last in stack for lexsort
        sort_idx = lexsort(torch.stack([active_fv, active_ei.float()]))
        sorted_ei = active_ei[sort_idx]
        sorted_indices = active_indices[sort_idx]

        # Find first occurrence of each unique niche (Bug #3)
        diff = torch.cat([torch.tensor([1], device=device), sorted_ei[1:] - sorted_ei[:-1]])
        first_mask = diff != 0

        unique_niche_indices = sorted_indices[first_mask]
        unique_niche_ids = sorted_ei[first_mask]

        # Map winners to their niche slots
        # Ensure unique_niche_ids are within bounds [0, N-1]
        valid_niche_mask = unique_niche_ids < N
        winners[unique_niche_ids[valid_niche_mask].to(torch.long)] = unique_niche_indices[valid_niche_mask]
        return winners


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SPEAR(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
