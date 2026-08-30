import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, randint

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class Two_Arch2(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size  # This acts as N (DA size)
        self.CAsize = pop_size  # CA size is usually same as DA size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.p = 1.0 / n_objs

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(self.CAsize, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.CAsize, n_objs), torch.inf, device=device))

        self.archive = Mutable(torch.rand(self.pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.archive_fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.archive_fit = self.evaluate(self.archive)

    def _minkowski_dist(self, A: torch.Tensor, B: torch.Tensor, p: float) -> torch.Tensor:
        # A: [N, M], B: [K, M] -> [N, K]
        diff = torch.abs(A.unsqueeze(1) - B.unsqueeze(0))
        dist = torch.pow(torch.sum(torch.pow(diff, p), dim=-1) + 1e-6, 1.0 / p)
        return dist

    def _update_ca(self, combined_pop, combined_fit):
        N_total = combined_fit.shape[0]
        if N_total <= self.CAsize:
            return combined_pop, combined_fit

        # Normalization
        f_min = torch.min(combined_fit, dim=0).values
        f_max = torch.max(combined_fit, dim=0).values
        norm_fit = (combined_fit - f_min) / (f_max - f_min + 1e-6)

        # Indicator Matrix I_eps+ (N, N)
        # I(i,j) = max(fit(i) - fit(j))
        indicator = torch.max(norm_fit.unsqueeze(1) - norm_fit.unsqueeze(0), dim=-1).values

        # Fitness F
        C = torch.max(torch.abs(indicator), dim=0).values
        # F_i = sum_{j} -exp(-I(j,i) / (C_i * 0.05))
        # Note: MATLAB uses I(j,i) in the sum for F_i
        F = torch.sum(-torch.exp(-indicator / (C.unsqueeze(0) + 1e-6) / 0.05), dim=0) + 1.0

        active_mask = torch.ones(N_total, device=combined_fit.device, dtype=torch.bool)
        num_to_remove = N_total - self.CAsize

        for _ in range(num_to_remove):
            # Find worst among active
            temp_F = torch.where(active_mask, F, torch.tensor(float("inf"), device=F.device))
            worst = torch.argmin(temp_F)
            active_mask[worst] = False
            # Update F: F_j = F_j + exp(-I(worst, j) / (C_worst * 0.05))
            F = F + torch.exp(-indicator[worst, :] / (C[worst] + 1e-6) / 0.05)

        return combined_pop[active_mask], combined_fit[active_mask]

    def _update_da(self, combined_pop, combined_fit):
        # 1. Non-dominated filter
        rank = non_dominate_rank(combined_fit)
        nd_mask = rank == 0
        nd_pop = combined_pop[nd_mask]
        nd_fit = combined_fit[nd_mask]

        N_nd = nd_pop.shape[0]
        if N_nd <= self.pop_size:
            return nd_pop, nd_fit

        # 2. Extreme Point Preservation
        min_idx = torch.argmin(nd_fit, dim=0)
        max_idx = torch.argmax(nd_fit, dim=0)
        extreme_indices = torch.unique(torch.cat([min_idx, max_idx]))

        is_selected = torch.zeros(N_nd, device=nd_pop.device, dtype=torch.bool)
        is_selected[extreme_indices] = True
        num_selected = is_selected.sum()

        if num_selected > self.pop_size:
            # Randomly delete excess extremes
            selected_indices = torch.where(is_selected)[0]
            perm = torch.randperm(num_selected, device=nd_pop.device)
            keep = selected_indices[perm[: self.pop_size]]
            final_mask = torch.zeros(N_nd, device=nd_pop.device, dtype=torch.bool)
            final_mask[keep] = True
            return nd_pop[final_mask], nd_fit[final_mask]

        # 3. Max-Min Greedy Selection
        dist_mat = self._minkowski_dist(nd_fit, nd_fit, self.p)
        # Distance from each candidate to the selected set
        # For candidates, min distance to any selected point
        min_dists = torch.min(dist_mat[:, is_selected], dim=1).values

        num_to_add = self.pop_size - num_selected.item()
        for _ in range(int(num_to_add)):
            # Score is min_dist for non-selected, -1 for selected
            score = torch.where(~is_selected, min_dists, torch.tensor(-1.0, device=nd_pop.device))
            best_cand = torch.argmax(score)

            is_selected[best_cand] = True
            # Update min_dists with the new point
            new_dists = dist_mat[:, best_cand]
            min_dists = torch.min(min_dists, new_dists)

        return nd_pop[is_selected], nd_fit[is_selected]

    def step(self) -> None:
        device = self.lb.device
        # 1. Mating Selection
        # ParentC: Tournament from CA + Random from DA
        n_half = (self.pop_size + 1) // 2
        idx_a = randint(0, self.CAsize, (n_half,), device=device)
        idx_b = randint(0, self.CAsize, (n_half,), device=device)
        fit_a, fit_b = self.fit[idx_a], self.fit[idx_b]

        # Pareto Dominance: a dominates b
        dom_ab = (fit_a <= fit_b).all(dim=-1) & (fit_a < fit_b).any(dim=-1)
        dom_ba = (fit_b <= fit_a).all(dim=-1) & (fit_b < fit_a).any(dim=-1)

        # If a dominates b, take a. If b dominates a, take b. Else random.
        rand_mask = torch.rand(n_half, device=device) < 0.5
        winner_idx = torch.where(dom_ab, idx_a, torch.where(dom_ba, idx_b, torch.where(rand_mask, idx_a, idx_b)))

        da_idx = randint(0, self.pop_size, (n_half,), device=device)
        parent_c = torch.cat([self.pop[winner_idx], self.archive[da_idx]], dim=0)

        # ParentM: Random from CA
        idx_m = randint(0, self.CAsize, (self.pop_size,), device=device)
        parent_m = self.pop[idx_m]

        # 2. Variation
        offspring_c = simulated_binary(parent_c, pro_c=1.0, dis_c=20.0)
        offspring_m = polynomial_mutation(parent_m, self.lb, self.ub, pro_m=1.0, dis_m=20.0)
        # Note: pro_m in MATLAB for OperatorGA is usually 1/D, but here it's applied to ParentM

        offspring = torch.cat([offspring_c, offspring_m], dim=0)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        # 3. Update Archives
        combined_pop = torch.cat([self.pop, self.archive, offspring], dim=0)
        combined_fit = torch.cat([self.fit, self.archive_fit, off_fit], dim=0)

        combined_pop, u_idx = unique_rows_sorted(combined_pop)
        combined_fit = combined_fit[u_idx]

        # Update CA
        new_ca_pop, new_ca_fit = self._update_ca(combined_pop, combined_fit)
        self.pop = new_ca_pop
        self.fit = new_ca_fit

        # Update DA
        new_da_pop, new_da_fit = self._update_da(combined_pop, combined_fit)

        # Ensure DA size is exactly pop_size
        n_da = new_da_pop.shape[0]
        if n_da < self.pop_size:
            # Deadlock breaker: fill with CA members if DA is too small
            shortfall = self.pop_size - n_da
            fill_idx = torch.arange(min(shortfall, self.pop.shape[0]), device=device)
            self.archive = torch.cat([new_da_pop, self.pop[fill_idx]], dim=0)[: self.pop_size]
            self.archive_fit = torch.cat([new_da_fit, self.fit[fill_idx]], dim=0)[: self.pop_size]
        else:
            self.archive = new_da_pop[: self.pop_size]
            self.archive_fit = new_da_fit[: self.pop_size]


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = Two_Arch2(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
