import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, randint

from evomo.operators.selection import non_dominate_rank


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
        fmin = torch.min(fit, dim=0)[0]
        fmax = torch.max(fit, dim=0)[0]
        step = (fmax - fmin) / self.div.float()
        safe_step = torch.where(step > 0, step, torch.ones_like(step))
        gloc = torch.floor((fit - fmin) / safe_step)
        gloc = torch.where(step > 0, gloc, torch.zeros_like(gloc))
        gloc = torch.clamp(gloc, 0, self.div - 1)

        same_grid = (gloc.unsqueeze(1) == gloc.unsqueeze(0)).all(dim=-1)
        density = same_grid.sum(dim=1)

        size = fit.shape[0]
        lower_triangle = torch.tril(torch.ones((size, size), dtype=torch.bool, device=fit.device), diagonal=-1)
        is_representative = ~(same_grid & lower_triangle).any(dim=1)
        return same_grid, density, is_representative

    def _mating_selection(self, fit: torch.Tensor):
        same_grid, density, is_representative = self._get_grid_info(fit)
        representatives = torch.where(is_representative)[0]
        grid_count = representatives.shape[0]

        grid_pair = randint(0, grid_count, (self.pop_size, 2), device=fit.device)
        grid_pair = representatives[grid_pair]
        pair_density = density[grid_pair]
        random_tie = torch.rand(self.pop_size, device=fit.device) < 0.5
        choose_first = (pair_density[:, 0] < pair_density[:, 1]) | (
            (pair_density[:, 0] == pair_density[:, 1]) & random_tie
        )
        chosen_grid = torch.where(choose_first, grid_pair[:, 0], grid_pair[:, 1])

        members = same_grid[chosen_grid]
        member_priority = torch.rand(members.shape, device=fit.device).masked_fill(~members, torch.inf)
        return torch.argmin(member_priority, dim=1)

    def _truncate(self, pop: torch.Tensor, fit: torch.Tensor):
        same_grid, _, _ = self._get_grid_info(fit)
        size = fit.shape[0]
        active = torch.ones(size, dtype=torch.bool, device=fit.device)
        lower_triangle = torch.tril(torch.ones((size, size), dtype=torch.bool, device=fit.device), diagonal=-1)

        for _ in range(size - self.pop_size):
            density = (same_grid & active.unsqueeze(0)).sum(dim=1)
            has_earlier_active = (same_grid & lower_triangle & active.unsqueeze(0)).any(dim=1)
            representatives = active & ~has_earlier_active
            max_density = torch.max(density[representatives])
            crowded_grids = representatives & (density == max_density)

            grid_priority = torch.rand(size, device=fit.device).masked_fill(~crowded_grids, torch.inf)
            selected_grid = torch.argmin(grid_priority)
            members = active & same_grid[selected_grid]
            member_priority = torch.rand(size, device=fit.device).masked_fill(~members, torch.inf)
            active[torch.argmin(member_priority)] = False

        return pop[active], fit[active]

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

    def step(self) -> None:
        # 1. Select two occupied grids uniformly, prefer the less crowded
        # grid, then sample one individual uniformly from that grid.
        mating_pool = self._mating_selection(self.fit)

        # 2. Variation
        parents = self.pop[mating_pool]
        if parents.shape[0] % 2 == 1:
            parents = torch.cat([parents, parents[:1]], dim=0)
        crossovered = simulated_binary(parents)[: self.pop_size]
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 3. Evaluation
        off_fit = self.evaluate(offspring)

        # 4. Environmental Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        # Pareto Filter (Bug #24: Dominance Logic)
        rank = non_dominate_rank(combined_fit)
        is_rank1 = rank == 0
        rank1_pop = combined_pop[is_rank1]
        rank1_fit = combined_fit[is_rank1]

        if rank1_pop.shape[0] > self.pop_size:
            rank1_pop, rank1_fit = self._truncate(rank1_pop, rank1_fit)

        self.pop = rank1_pop
        self.fit = rank1_fit


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
