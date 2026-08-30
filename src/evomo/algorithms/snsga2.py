import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class SNSGA2(Algorithm):
    def __init__(
        self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, sLower: float = 0.1, sUpper: float = 0.9, **kwargs
    ):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.sLower = sLower
        self.sUpper = sUpper

        # Initialize State (Mutables)
        # Initialization (VSSPS Logic)
        rho = torch.linspace(1 - sLower, 1 - sUpper, pop_size, device=device)
        W = (rho * self.dim).ceil().to(torch.int32)
        idx = torch.arange(self.dim, device=device).unsqueeze(0).expand(pop_size, self.dim)
        shift = torch.arange(pop_size, device=device).unsqueeze(1)
        rotated_idx = (idx + shift) % self.dim
        mask = rotated_idx < W.unsqueeze(1)

        initial_pop = (self.lb + torch.rand((pop_size, self.dim), device=device) * (self.ub - self.lb)) * mask
        self.pop = Mutable(initial_pop)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        sentinel = torch.iinfo(torch.int32).max
        self.front_no = Mutable(torch.full((pop_size,), sentinel, dtype=torch.int32, device=device))
        self.crowd_dis = Mutable(torch.full((pop_size,), -1.0, dtype=torch.float32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        # Initial Environmental Selection
        self.pop, self.fit, self.front_no, self.crowd_dis = self._environmental_selection(self.pop, self.fit)

    def _sparse_sbx(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        # SSBX Logic
        mask1, mask2 = (p1 != 0), (p2 != 0)
        matching = mask1 == mask2

        # Standard SBX on matching
        off = simulated_binary(torch.cat([p1, p2], dim=0), pro_c=1.0, dis_c=20.0)
        off1, off2 = off[: self.pop_size // 2], off[self.pop_size // 2 :]

        # Swap on non-matching with 0.5 probability
        swap_mask = (torch.rand(p1.shape, device=p1.device) < 0.5) & (~matching)
        res1 = torch.where(swap_mask, p2, p1)
        res2 = torch.where(swap_mask, p1, p2)

        # Combine matching SBX and non-matching swaps
        final1 = torch.where(matching, off1, res1)
        final2 = torch.where(matching, off2, res2)
        return torch.cat([final1, final2], dim=0)

    def _sparse_mutate_mask(self, pop: torch.Tensor, target_W: torch.Tensor) -> torch.Tensor:
        N, D = pop.shape
        curr_W = (pop != 0).sum(1)
        diff = target_W - curr_W

        to_add = diff.clamp(min=0)
        to_rem = (-diff).clamp(min=0)

        add_probs = (pop == 0).float() + 1e-6
        rem_probs = (pop != 0).float() + 1e-6

        add_noise = torch.rand((N, D), device=pop.device) * add_probs
        rem_noise = torch.rand((N, D), device=pop.device) * rem_probs

        add_rank = torch.argsort(add_noise, dim=1, descending=True)
        rem_rank = torch.argsort(rem_noise, dim=1, descending=True)

        col_idx = torch.arange(D, device=pop.device).unsqueeze(0).expand(N, D)
        add_mask = col_idx < to_add.unsqueeze(1)
        rem_mask = col_idx < to_rem.unsqueeze(1)

        final_add = torch.zeros_like(pop, dtype=torch.bool).scatter_(1, add_rank, add_mask)
        final_rem = torch.zeros_like(pop, dtype=torch.bool).scatter_(1, rem_rank, rem_mask)

        new_pop = pop.clone()
        vals = self.lb + torch.rand((N, D), device=pop.device) * (self.ub - self.lb)
        new_pop = torch.where(final_add, vals, new_pop)
        new_pop = torch.where(final_rem, torch.zeros_like(new_pop), new_pop)

        return new_pop

    def _environmental_selection(self, merge_pop: torch.Tensor, merge_fit: torch.Tensor):
        N = self.pop_size
        X, u_idx = unique_rows_sorted(merge_pop)
        F = merge_fit[u_idx]

        rank = non_dominate_rank(F)
        num_selected = 0
        selected_indices = torch.full((N,), -1, dtype=torch.long, device=X.device)
        all_crowd_dis = torch.zeros(F.shape[0], device=X.device)

        for i in range(2 * N):
            if num_selected >= N:
                break

            curr_mask = rank == i
            count = curr_mask.sum()

            if count == 0:
                continue

            dist = crowding_distance(F, curr_mask)
            all_crowd_dis = torch.where(curr_mask, dist, all_crowd_dis)

            if num_selected + count <= N:
                indices = torch.where(curr_mask)[0]
                selected_indices[num_selected : num_selected + count] = indices
                num_selected += count
            else:
                needed = N - num_selected
                front_indices = torch.where(curr_mask)[0]
                front_dist = dist[front_indices]
                top_indices = front_indices[torch.argsort(front_dist, descending=True)[:needed]]
                selected_indices[num_selected:N] = top_indices
                num_selected = N

        survivor_pop = X[selected_indices]
        survivor_fit = F[selected_indices]
        survivor_rank = rank[selected_indices]
        survivor_dist = all_crowd_dis[selected_indices]

        return survivor_pop, survivor_fit, survivor_rank.to(torch.int32), survivor_dist

    def step(self) -> None:
        mating_pool_idx = tournament_selection_multifit(
            self.pop_size, [self.front_no.float(), -self.crowd_dis], tournament_size=2
        )
        parents = self.pop[mating_pool_idx]

        off_pop = self._sparse_sbx(parents[: self.pop_size // 2], parents[self.pop_size // 2 :])
        if off_pop.shape[0] < self.pop_size:
            off_pop = torch.cat([off_pop, off_pop[:1]], dim=0)

        off_pop = torch.where(off_pop != 0, polynomial_mutation(off_pop, self.lb, self.ub), off_pop)

        curr_rho = (off_pop != 0).sum(dim=1).float() / self.dim
        target_rho = clamp(curr_rho + torch.randn_like(curr_rho) * 0.1, 1.0 / self.dim, 1.0)
        target_W = (target_rho * self.dim).round().clamp(1, self.dim).to(torch.int32)
        off_pop = self._sparse_mutate_mask(off_pop, target_W)

        off_pop = clamp(off_pop, self.lb, self.ub)
        off_fit = self.evaluate(off_pop)

        merge_pop = torch.cat([self.pop, off_pop], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit, self.front_no, self.crowd_dis = self._environmental_selection(merge_pop, merge_fit)


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SNSGA2(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
