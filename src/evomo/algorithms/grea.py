import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class GrEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, div: int = 10, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.div = Parameter(torch.tensor(div, dtype=torch.int32, device=device))
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

    def _calc_grid_metrics(self, fit: torch.Tensor):
        f_min = torch.min(fit, dim=0)[0]
        f_max = torch.max(fit, dim=0)[0]
        d = (f_max - f_min) / self.div.float()
        lb = f_min - d / 2

        # GLoc calculation with Bug #12 compliance
        g_loc = torch.floor((fit - lb) / (d + 1e-6)).to(torch.int32)

        # GCPD calculation
        gcpd = torch.sqrt(torch.sum(((fit - lb) / (d + 1e-6) - g_loc.float()) ** 2, dim=-1))

        # Pairwise Grid Distance Matrix (GD)
        # GD(i,j) = sum(abs(g_loc_i - g_loc_j))
        gd_matrix = torch.sum(torch.abs(g_loc.unsqueeze(1) - g_loc.unsqueeze(0)), dim=-1)

        # Grid Crowding Distance (GCD)
        # GCD_i = sum(max(M - GD(i,j), 0))
        gcd = torch.sum(torch.clamp(self.n_objs - gd_matrix, min=0), dim=-1)

        # Grid Ranking (GR)
        # GR_i = count(j grid-dominates i) + count(GD(i,j) < M)
        # Grid Dominance: (g_loc_j <= g_loc_i).all() & (g_loc_j < g_loc_i).any()
        g_dom = (g_loc.unsqueeze(0) <= g_loc.unsqueeze(1)).all(-1) & (g_loc.unsqueeze(0) < g_loc.unsqueeze(1)).any(-1)
        gr_dom = g_dom.sum(dim=0).to(torch.int32)
        gr_dist = (gd_matrix < self.n_objs).sum(dim=-1).to(torch.int32) - 1  # exclude self
        gr = gr_dom + gr_dist

        return g_loc, gcd, gr, gcpd, gd_matrix

    def _grid_dominance(self, g_loc: torch.Tensor) -> torch.Tensor:
        # Returns matrix where out[i, j] is True if i grid-dominates j
        return (g_loc.unsqueeze(0) <= g_loc.unsqueeze(1)).all(-1) & (g_loc.unsqueeze(0) < g_loc.unsqueeze(1)).any(-1)

    def step(self) -> None:
        device = self.pop.device
        # 1. Mating Selection
        g_loc, gcd, gr, gcpd, _ = self._calc_grid_metrics(self.fit)
        rank = non_dominate_rank(self.fit)

        # Tournament Selection (Bug #25: Primary key last)
        # Criteria: Pareto Rank (min), Grid Rank (min), GCD (min)
        mating_idx = tournament_selection_multifit(self.pop_size, [rank.float(), gr.float(), gcd.float()], tournament_size=2)

        # 2. Variation
        parents = self.pop[mating_idx]
        offspring = simulated_binary(parents)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        # 3. Environmental Selection
        c_pop = torch.cat([self.pop, offspring], dim=0)
        c_fit = torch.cat([self.fit, off_fit], dim=0)

        # Non-dominated Sort
        c_rank = non_dominate_rank(c_fit)

        # Front Filling
        survivor_mask = torch.zeros(c_pop.shape[0], dtype=torch.bool, device=device)
        current_rank = 0
        num_selected = 0

        # Find the last front to be processed
        while num_selected < self.pop_size:
            in_front = c_rank == current_rank
            count_in_front = in_front.sum()
            if num_selected + count_in_front <= self.pop_size:
                survivor_mask = survivor_mask | in_front
                num_selected += count_in_front
                current_rank += 1
            else:
                # This is the LastFront
                break

        # Peeling Logic for LastFront (Bug #9, #29, #41)
        if num_selected < self.pop_size:
            last_front_indices = torch.where(c_rank == current_rank)[0]
            lf_fit = c_fit[last_front_indices]
            lf_g_loc, lf_gcd, lf_gr, lf_gcpd, lf_gd_matrix = self._calc_grid_metrics(lf_fit)

            # Active mask for peeling within the last front
            active_mask = torch.ones(last_front_indices.shape[0], dtype=torch.bool, device=device)
            num_remaining = last_front_indices.shape[0]
            target_size = self.pop_size - num_selected

            # Peeling Loop: Remove individuals until target_size is reached
            while num_remaining > target_size:
                # Mask inactive indices with very small values to not be picked as max
                sort_gr = torch.where(active_mask, lf_gr.float(), -1e9)
                sort_gcd = torch.where(active_mask, lf_gcd.float(), -1e9)
                sort_gcpd = torch.where(active_mask, lf_gcpd.float(), -1e9)

                # Primary key (GR) last
                sort_idx = lexsort(torch.stack([sort_gcpd, sort_gcd, sort_gr]))
                q = sort_idx[-1]

                # Vectorized Update (Bug #29)
                active_mask[q] = False
                num_remaining -= 1

                # Update GCD: GCD_p = GCD_p - max(M - GD(p, q), 0)
                update_gcd = torch.clamp(self.n_objs - lf_gd_matrix[:, q], min=0)
                lf_gcd = lf_gcd - update_gcd

                # Update GR:
                # 1. If q grid-dominates p, GR_p = GR_p - 1
                # 2. If GD(p, q) < M, GR_p = GR_p - 1
                q_g_loc = lf_g_loc[q]
                p_is_dominated = (q_g_loc <= lf_g_loc).all(-1) & (q_g_loc < lf_g_loc).any(-1)
                p_is_close = lf_gd_matrix[:, q] < self.n_objs

                lf_gr = lf_gr - p_is_dominated.to(torch.int32) - p_is_close.to(torch.int32)

            # Add survivors from LastFront
            survivor_mask[last_front_indices[active_mask]] = True

        self.pop = c_pop[survivor_mask]
        self.fit = c_fit[survivor_mask]


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = GrEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
