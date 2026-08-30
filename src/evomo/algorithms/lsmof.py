import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class LSMOF(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.D = lb.numel()

        # LSMOF Specific Parameters
        self.wD = kwargs.get("wD", 5)  # Number of reference solutions
        self.SubN = kwargs.get("SubN", 20)  # Sub-population for weight optimization
        self.wmax = Parameter(torch.tensor(kwargs.get("wmax", 0.1), device=device))
        self.max_fe = kwargs.get("max_fe", 10000)
        self.switch_fe = self.max_fe // 2

        # Initialize State
        self.pop = Mutable(torch.rand(pop_size, self.D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        self.archive_pop = Mutable(self.pop.clone())
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        self.fe_counter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        self.rank = Mutable(torch.full((pop_size,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device))
        self.dis = Mutable(torch.full((pop_size,), -torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.fe_counter = self.fe_counter + self.pop_size
        self.archive_fit = self.fit.clone()
        self.archive_pop = self.pop.clone()
        _, _, self.rank, self.dis = self._environmental_selection(self.pop, self.fit, self.pop_size)

    def _reconstruct_decisions(
        self, weights: torch.Tensor, directions: torch.Tensor, base: torch.Tensor, wmax: torch.Tensor
    ) -> torch.Tensor:
        # weights: (SubN, 2*wD), directions: (2*wD, D), base: (2*wD, D)
        # Broadcasting: (SubN, 2*wD, 1) * (1, 2*wD, D) -> (SubN, 2*wD, D)
        offset = torch.einsum("sw,wd->swd", weights, directions) * wmax
        dec = base.unsqueeze(0) + offset
        return dec.reshape(-1, self.D)

    def _environmental_selection(self, pop, fit, N):
        combined_pop, combined_idx = unique_rows_sorted(pop)
        combined_fit = fit[combined_idx]

        rank = non_dominate_rank(combined_fit)
        N_total = combined_fit.shape[0]
        mask = torch.zeros(N_total, dtype=torch.bool, device=pop.device)
        distances = torch.full((N_total,), -1.0, device=pop.device)

        num_selected = 0

        # Peeling Loop
        for i in range(N_total):
            curr_front_mask = rank == i
            count = torch.sum(curr_front_mask.int())

            is_empty = count == 0
            is_done = num_selected >= N

            if not is_done and not is_empty:
                if num_selected + count <= N:
                    mask |= curr_front_mask
                    distances[curr_front_mask] = crowding_distance(combined_fit, curr_front_mask)[curr_front_mask]
                    num_selected += count
                else:
                    dist = crowding_distance(combined_fit, curr_front_mask)
                    idx_in_front = torch.where(curr_front_mask)[0]
                    sorted_sub_idx = torch.argsort(dist[curr_front_mask], descending=True)
                    selected_indices = idx_in_front[sorted_sub_idx[: N - num_selected]]
                    mask[selected_indices] = True
                    distances[selected_indices] = dist[selected_indices]
                    num_selected = N

        survivor_pop = combined_pop[mask]
        survivor_fit = combined_fit[mask]
        survivor_rank = rank[mask]
        survivor_dis = distances[mask]
        return survivor_pop, survivor_fit, survivor_rank, survivor_dis

    def step(self) -> None:
        device = self.pop.device

        if self.fe_counter < self.switch_fe:
            # Phase A: Bi-directional Weight Optimization
            # 1. Reference Selection
            ref_pop, _, _, _ = self._environmental_selection(self.pop, self.fit, self.wD)

            # 2. Direction Matrix
            Direct_L = (ref_pop - self.lb) / (torch.norm(ref_pop - self.lb, dim=1, keepdim=True) + 1e-6)
            Direct_U = (ref_pop - self.ub) / (torch.norm(ref_pop - self.ub, dim=1, keepdim=True) + 1e-6)
            Direct = torch.cat([Direct_L, Direct_U], dim=0)  # (2*wD, D)
            Base = torch.cat([self.lb.repeat(self.wD, 1), self.ub.repeat(self.wD, 1)], dim=0)  # (2*wD, D)

            # 3. Internal DE for Weights
            weights = torch.rand(self.SubN, 2 * self.wD, device=device)
            decisions = self._reconstruct_decisions(weights, Direct, Base, self.wmax)
            decisions = clamp(decisions, self.lb, self.ub)
            off_fit = self.evaluate(decisions)
            self.fe_counter = self.fe_counter + decisions.shape[0]

            # Update Archive and Pop
            merged_pop = torch.cat([self.pop, decisions], dim=0)
            merged_fit = torch.cat([self.fit, off_fit], dim=0)
            self.pop, self.fit, self.rank, self.dis = self._environmental_selection(merged_pop, merged_fit, self.pop_size)

        else:
            # Phase B: Standard NSGA-II Refinement
            mating_pool = tournament_selection_multifit(self.pop_size, [-self.dis, self.rank.float()], tournament_size=2)
            parents = self.pop[mating_pool]
            offspring = simulated_binary(parents)
            offspring = polynomial_mutation(offspring, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)

            off_fit = self.evaluate(offspring)
            self.fe_counter = self.fe_counter + self.pop_size

            merged_pop = torch.cat([self.pop, offspring], dim=0)
            merged_fit = torch.cat([self.fit, off_fit], dim=0)
            self.pop, self.fit, self.rank, self.dis = self._environmental_selection(merged_pop, merged_fit, self.pop_size)


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = LSMOF(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
