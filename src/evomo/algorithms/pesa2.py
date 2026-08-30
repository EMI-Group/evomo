import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection import nd_environmental_selection, non_dominate_rank
from evomo.utils import unique_rows_sorted


class PESA2(Algorithm):
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
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)  # [N,D]
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))  # [N,M]

    def _get_grid_info(self, fit: torch.Tensor):
        # 1. Calculate bounds
        fmin = torch.min(fit, dim=0)[0]
        fmax = torch.max(fit, dim=0)[0]

        # 2. Calculate grid step (Bug #12: Safe Division)
        step = (fmax - fmin) / (self.div.float() + 1e-6)

        # 3. Compute coordinates
        gloc = torch.floor((fit - fmin) / (step + 1e-6))
        gloc = torch.clamp(gloc, 0, self.div - 1)

        # 4. Flatten to 1D IDs (Bug #3: Unique Stability)
        _, gid = unique_rows_sorted(gloc)

        # 5. Density (Vectorized)
        crowd_g = torch.bincount(gid)

        return gid, crowd_g

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

    def step(self) -> None:
        device = self.pop.device

        # 1. Mating Selection (Grid-based Tournament)
        gid, crowd_g = self._get_grid_info(self.fit)
        # Binary tournament on individuals using their grid's density
        mating_pool = tournament_selection_multifit(self.pop_size, [crowd_g[gid].float()], tournament_size=2)

        # 2. Variation
        crossovered = simulated_binary(self.pop[mating_pool])
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 3. Evaluation
        off_fit = self.evaluate(offspring)

        # 4. Environmental Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        # Pareto Filter (Bug #24: Dominance Logic)
        rank = non_dominate_rank(combined_fit)
        is_rank1 = rank == 1
        num_rank1 = torch.sum(is_rank1.int())

        # Deadlock Breaker (Bug #40): If Rank 1 < N, use standard NDSort logic to fill
        if num_rank1 < self.pop_size:
            # Use standard NDSort logic to get exactly N individuals
            # We use crowding distance as a secondary metric for the last front
            self.pop, self.fit, _, _, _ = nd_environmental_selection(combined_pop, combined_fit, self.pop_size)
        else:
            # Brutal Static Truncation (Bug #30) on Rank 1 set
            rank1_pop = combined_pop[is_rank1]
            rank1_fit = combined_fit[is_rank1]

            gid_r1, crowd_g_r1 = self._get_grid_info(rank1_fit)

            density_score = crowd_g_r1[gid_r1].float()
            random_tie_breaker = torch.rand(density_score.shape[0], device=device)
            final_score = density_score + random_tie_breaker

            # Select N least crowded
            _, top_indices = torch.topk(final_score, k=self.pop_size, largest=False)
            self.pop = rank1_pop[top_indices]
            self.fit = rank1_fit[top_indices]


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = PESA2(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
