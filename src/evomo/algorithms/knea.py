import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class KnEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.knee_points = Mutable(torch.zeros(pop_size, dtype=torch.bool, device=device))

        # Adaptive neighborhood ratio and knee point ratio per front
        # Using 2*pop_size to handle combined population in step
        self.r = Mutable(torch.full((2 * pop_size,), 0.1, device=device))
        self.t = Mutable(torch.full((2 * pop_size,), 0.0, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        # Initial knee point identification for the starting population
        is_knee, dists = self._get_knee_points(self.fit, torch.tensor(0.1, device=self.fit.device))
        self.knee_points = is_knee

    def _get_knee_points(self, f_fit: torch.Tensor, r_val: torch.Tensor):
        N_f, M = f_fit.shape
        device = f_fit.device

        # 1. Extreme Points & Hyperplane
        extreme_idx = torch.argmin(f_fit, dim=0)
        E = f_fit[extreme_idx]

        # Solve E * w = 1
        ones = torch.ones((M, 1), device=device)
        w = torch.linalg.lstsq(E, ones).solution

        # Distance to hyperplane
        dist = torch.abs(f_fit @ w - 1.0).squeeze(-1) / (torch.norm(w) + 1e-6)

        # 2. Neighborhood Update
        f_min = torch.min(f_fit, dim=0).values
        f_max = torch.max(f_fit, dim=0).values
        R = r_val * (f_max - f_min)

        # 3. Vectorized Selection (Iterative suppression)
        remain = torch.ones(N_f, dtype=torch.bool, device=device)
        is_knee = torch.zeros(N_f, dtype=torch.bool, device=device)

        # Sort by distance descending
        sorted_indices = torch.argsort(dist, descending=True)

        # JIT-friendly loop: iterate over sorted candidates
        for i in range(N_f):
            idx = sorted_indices[i]
            # If the point is still remaining, it's a knee point
            is_current_knee = remain[idx]
            is_knee[idx] = is_current_knee

            # Suppress neighbors if it was selected as a knee point
            in_neighborhood = torch.all(torch.abs(f_fit - f_fit[idx]) <= R + 1e-6, dim=1)
            remain = torch.where(is_current_knee, remain & ~in_neighborhood, remain)

        return is_knee, dist

    def _update_adaptive_params(self, f_idx: torch.Tensor, t_val: torch.Tensor):
        # Bug #12: Safe division
        denom = torch.tensor(float(self.n_objs), device=self.r.device)
        ratio = (1.0 - t_val / 0.5) / (denom + 1e-6)
        self.r[f_idx] = self.r[f_idx] / (torch.exp(ratio) + 1e-6)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size

        # 1. Mating Selection
        # Weighted Distance for Crowding
        D_mat = torch.cdist(self.fit, self.fit, p=2)
        vals, _ = torch.topk(D_mat, k=4, largest=False)
        d1, d2, d3 = vals[:, 1], vals[:, 2], vals[:, 3]
        crowd = 3 * d1 + 2 * d2 + 1 * d3

        # Tournament
        rank = non_dominate_rank(self.fit)
        mating_pool = tournament_selection_multifit(N, [rank, -self.knee_points.float(), -crowd], tournament_size=2)

        # Variation
        crossovered = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(crossovered, self.lb, self.ub, pro_m=1.0 / self.pop.shape[1], dis_m=20.0)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(offspring)

        # 3. Environmental Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        # NDSort
        fronts = non_dominate_rank(combined_fit)

        # Peeling logic
        new_pop = torch.zeros_like(self.pop)
        new_fit = torch.zeros_like(self.fit)
        new_knee = torch.zeros(N, dtype=torch.bool, device=device)

        current_count = 0

        # We iterate through fronts to fill the population
        # JIT-safe: iterate up to 2*N (worst case)
        for f_no in range(2 * N):
            mask = fronts == f_no
            num_in_front = torch.sum(mask.int())

            # If front is empty or we are already full, skip
            is_active = (num_in_front > 0) & (current_count < N)

            if is_active:
                f_fit = combined_fit[mask]
                f_pop = combined_pop[mask]

                # Calculate knee points for this front
                is_knee_f, dist_f = self._get_knee_points(f_fit, self.r[f_no])

                # Update adaptive parameters
                t_val = torch.sum(is_knee_f.float()) / (num_in_front.float() + 1e-6)
                self._update_adaptive_params(torch.tensor(f_no, device=device), t_val)

                if current_count + num_in_front <= N:
                    # Select all
                    indices = torch.arange(current_count, current_count + num_in_front, device=device)
                    new_pop[indices] = f_pop
                    new_fit[indices] = f_fit
                    new_knee[indices] = is_knee_f
                    current_count += num_in_front
                else:
                    # Partial selection from the last front
                    num_needed = N - current_count
                    # Priority: Knee points first, then hyperplane distance
                    # lexsort: primary key last. Primary: is_knee, Secondary: dist
                    sel_indices = lexsort(torch.stack([-dist_f, -is_knee_f.float()]))

                    final_indices = sel_indices[:num_needed]
                    fill_indices = torch.arange(current_count, N, device=device)

                    new_pop[fill_indices] = f_pop[final_indices]
                    new_fit[fill_indices] = f_fit[final_indices]
                    new_knee[fill_indices] = is_knee_f[final_indices]
                    current_count = N

        self.pop = new_pop
        self.fit = new_fit
        self.knee_points = new_knee


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = KnEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
