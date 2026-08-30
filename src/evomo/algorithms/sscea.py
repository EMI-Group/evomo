import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, nanmax, nanmin

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class SSCEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, max_gen: int = 100):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.max_gen = Parameter(max_gen)

        # Initialize State (Mutables)
        # Convergence Archive (CA) - Dynamic Size
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Diversity Archive (DA) - Fixed Size
        self.archive_pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Subspace Masks
        self.dv_mask = Mutable(torch.zeros(self.dim, dtype=torch.bool, device=device))
        self.cv_mask = Mutable(torch.zeros(self.dim, dtype=torch.bool, device=device))
        self.gen_counter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def _variable_clustering(self):
        # Logic: Sensitivity analysis via perturbations
        n_sel = 5
        n_per = 50
        device = self.lb.device

        # Randomly pick solutions from current CA
        indices = torch.randint(0, self.pop.shape[0], (n_sel,), device=device)
        seeds = self.pop[indices]

        # Perturb each variable and measure objective change
        # Shape: [dim, n_sel, n_per]
        delta = 0.1 * (self.ub - self.lb)
        sensitivity = torch.zeros(self.dim, device=device)

        for j in range(self.dim):
            # Vectorized perturbation for dimension j
            p_pop = seeds.repeat_interleave(n_per, dim=0)
            noise = (torch.rand(n_sel * n_per, device=device) - 0.5) * delta[j]
            p_pop[:, j] = clamp(p_pop[:, j] + noise, self.lb[j], self.ub[j])

            # Evaluate (Note: In a real workflow, evaluate is external,
            # but for clustering we use a local approximation or the problem)
            p_fit = self.evaluate(p_pop)
            # Standard deviation across perturbations as sensitivity proxy
            p_fit_reshaped = p_fit.view(n_sel, n_per, self.n_objs)
            sensitivity[j] = torch.mean(torch.std(p_fit_reshaped, dim=1))

        # Split based on median sensitivity
        threshold = torch.median(sensitivity)
        cv_mask = sensitivity >= threshold
        dv_mask = ~cv_mask
        return cv_mask, dv_mask

    def _update_ca_size(self):
        fe_ratio = self.gen_counter.float() / self.max_gen.float()
        n = float(self.pop_size)
        ca_size = (n / 10.0 + (n - n / 10.0) * (fe_ratio**2)).int()
        return torch.clamp(ca_size, min=5)

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.archive_fit = self.evaluate(self.archive_pop)
        cv, dv = self._variable_clustering()
        self.cv_mask = cv
        self.dv_mask = dv

    def step(self) -> None:
        self.gen_counter = self.gen_counter + 1
        device = self.lb.device

        # 1. Subspace Variation
        # Combine CA and DA for mating
        combined_pop = torch.cat([self.pop, self.archive_pop], dim=0)
        parents_idx = torch.randint(0, combined_pop.shape[0], (self.pop_size,), device=device)
        parents = combined_pop[parents_idx]

        off_pop = simulated_binary(parents, pro_c=1.0, dis_c=20.0)
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # Conditional Subspace Selection
        fe_ratio = self.gen_counter.float() / self.max_gen.float()
        condition = (fe_ratio < 0.5) | (torch.rand(1, device=device) < 0.5)

        # Apply masks via broadcasting
        final_off = torch.where(
            condition, off_pop * self.cv_mask + parents * (~self.cv_mask), off_pop * self.dv_mask + parents * (~self.dv_mask)
        )

        off_fit = self.evaluate(final_off)

        # 2. Update CA: Indicator-based Peeling
        merged_ca_pop = torch.cat([self.pop, final_off], dim=0)
        merged_ca_fit = torch.cat([self.fit, off_fit], dim=0)

        # Unique rows
        merged_ca_pop, u_idx = unique_rows_sorted(merged_ca_pop)
        merged_ca_fit = merged_ca_fit[u_idx]

        # Normalization
        f_min = nanmin(merged_ca_fit, dim=0)[0]
        f_max = nanmax(merged_ca_fit, dim=0)[0]
        f_norm = (merged_ca_fit - f_min) / (f_max - f_min + 1e-6)

        # Indicator Matrix I (N x N)
        # I_ij = max(f_i - f_j)
        indicator = torch.max(f_norm.unsqueeze(1) - f_norm.unsqueeze(0), dim=-1).values
        C = torch.max(torch.abs(indicator))

        # Initial Fitness F
        # F_i = sum_{j!=i} -exp(-I_ji / (C*0.05))
        # Note: I[j, i] is f_norm[j] - f_norm[i]
        kernel = torch.exp(-indicator / (C * 0.05 + 1e-6))
        F = -torch.sum(kernel, dim=0)

        # Peeling Loop
        target_ca_size = self._update_ca_size()
        current_indices = torch.ones(merged_ca_pop.shape[0], dtype=torch.bool, device=device)
        num_to_remove = merged_ca_pop.shape[0] - target_ca_size

        # Sentinel for removed
        sentinel_fit = torch.tensor(float("inf"), device=device)

        for _ in range(num_to_remove):
            # Find worst (min fitness in this formulation)
            active_F = torch.where(current_indices, F, sentinel_fit)
            worst = torch.argmin(active_F)
            current_indices[worst] = False
            # Update remaining: F_i = F_i + exp(-I_worst_i / ...)
            F = F + kernel[worst, :]

        self.pop = merged_ca_pop[current_indices]
        self.fit = merged_ca_fit[current_indices]

        # 3. Update DA: Diversity Selection
        merged_da_pop = torch.cat([self.archive_pop, final_off], dim=0)
        merged_da_fit = torch.cat([self.archive_fit, off_fit], dim=0)
        merged_da_pop, u_idx_da = unique_rows_sorted(merged_da_pop)
        merged_da_fit = merged_da_fit[u_idx_da]

        # Rank-1 Extraction
        rank = non_dominate_rank(merged_da_fit)
        mask_r1 = rank == 1
        da_pop_r1 = merged_da_pop[mask_r1]
        da_fit_r1 = merged_da_fit[mask_r1]

        # Extreme Selection (ASF)
        W = torch.eye(self.n_objs, device=device) + 1e-6
        # asf = max(f/w) + 0.1 * sum(f/w)
        asf = torch.max(da_fit_r1.unsqueeze(1) / W.unsqueeze(0), dim=-1).values + 0.1 * (da_fit_r1 @ W.T / 1e-6)
        extreme_indices = torch.argmin(asf, dim=0)

        # Cosine-based Angle Selection
        selected_mask = torch.zeros(da_pop_r1.shape[0], dtype=torch.bool, device=device)
        selected_mask[extreme_indices] = True

        norm_fit = da_fit_r1 / (torch.norm(da_fit_r1, dim=1, keepdim=True) + 1e-6)

        # Fill DA up to pop_size
        for _ in range(self.pop_size - torch.sum(selected_mask)):
            selected_norm = norm_fit[selected_mask]
            # Cosine Similarity via einsum
            cos_sim = torch.einsum("id,jd->ij", norm_fit, selected_norm)
            angles = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
            min_angles = torch.min(angles, dim=1).values
            # Mask already selected
            min_angles = torch.where(selected_mask, -1.0, min_angles)
            best_next = torch.argmax(min_angles)
            selected_mask[best_next] = True

        self.archive_pop = da_pop_r1[selected_mask]
        self.archive_fit = da_fit_r1[selected_mask]


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SSCEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
