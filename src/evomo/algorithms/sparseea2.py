import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection import nd_environmental_selection, non_dominate_rank
from evomo.utils import unique_rows_sorted


class SparseEA2(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.mask = Mutable(torch.zeros((pop_size, self.dim), dtype=torch.bool, device=device))
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.var_fit = Mutable(torch.zeros(self.dim, device=device))
        self.rank = Mutable(torch.zeros(pop_size, device=device))
        self.dis = Mutable(torch.zeros(pop_size, device=device))

    def init_step(self) -> None:
        device = self.lb.device
        D = self.dim
        N = self.pop_size

        # 1. Variable Fitness Estimation (Identity Sampling)
        mask_eye = torch.eye(D, device=device)
        # Evaluate identity-masked population to estimate variable importance
        # We use the first D individuals or repeat if N < D
        init_pop_for_var = self.pop[0].repeat(D, 1) * mask_eye
        fits_var = self.evaluate(init_pop_for_var)

        # Ranking (NDSort) - Lower rank is better
        ranks = non_dominate_rank(fits_var)
        self.var_fit = ranks.float()

        # 2. Initial Mask Generation via Tournament Selection
        # For each individual, pick a random number of bits k, then pick k bits via tournament
        k_values = torch.randint(1, D + 1, (N,), device=device)
        init_mask = torch.zeros((N, D), dtype=torch.bool, device=device)

        # Vectorized tournament for initial mask
        indices_pool = torch.arange(D, device=device).repeat(N, 1)
        for i in range(2):  # Small loop for tournament rounds is fine if not data-dependent
            rand_idx = torch.randint(0, D, (N, D), device=device)
            winner_mask = self.var_fit[indices_pool] < self.var_fit[rand_idx]
            indices_pool = torch.where(winner_mask, indices_pool, rand_idx)

        # Use top-k selection based on tournament results per row
        # (Simplified for JIT: use the tournament winners and mask by k)
        row_idx = torch.arange(N, device=device).unsqueeze(1)
        col_idx = torch.argsort(torch.rand(N, D, device=device), dim=1)
        mask_threshold = torch.arange(D, device=device).expand(N, D) < k_values.unsqueeze(1)
        init_mask[row_idx, col_idx] = mask_threshold

        self.mask = init_mask
        self.fit = self.evaluate(self.pop * self.mask)

        # Initial Environmental Selection
        self.pop, self.fit, self.rank, self.dis, _ = nd_environmental_selection(self.pop, self.fit, N)

    def _sparse_tournament(self, indices: torch.Tensor, var_fit: torch.Tensor) -> torch.Tensor:
        # Performs tournament selection on the var_fit values
        N, D = indices.shape
        rand_indices = torch.randint(0, self.dim, (N, D), device=indices.device)
        fit_a = var_fit[indices]
        fit_b = var_fit[rand_indices]
        return torch.where(fit_a < fit_b, indices, rand_indices)

    def _grouped_mutation(self, pop: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        N, D = pop.shape
        device = pop.device

        # Dec Mutation (Grouped)
        ranks = torch.argsort(pop, dim=1)
        group_size = D // 4
        group_id = torch.randint(0, 4, (N, 1), device=device)
        mutation_mask = ranks // (group_size + 1e-6) == group_id

        off_pop = polynomial_mutation(pop, self.lb, self.ub)
        off_pop = torch.where(mutation_mask, off_pop, pop)

        # Mask Mutation
        flip_prob = 1.0 / D
        flip_mask = (torch.rand((N, D), device=device) < flip_prob) & mutation_mask

        # Bug #36: Knowledge-guided bit flip
        # If flipping 0 -> 1, bias by var_fit. If 1 -> 0, flip.
        off_mask = mask.clone()
        to_flip = flip_mask

        # Apply flips
        off_mask = torch.where(to_flip, ~off_mask, off_mask)

        return off_pop, off_mask

    def step(self) -> None:
        N = self.pop_size
        D = self.dim
        device = self.lb.device

        # 1. Mating (Selection & Crossover)
        mating_pool = tournament_selection_multifit(N, [-self.dis, self.rank])
        p1_idx = mating_pool[: N // 2]
        p2_idx = mating_pool[N // 2 :]
        # Ensure N is even for pairing or handle remainder
        p2_idx = torch.cat([p2_idx, mating_pool[: N - (N // 2) * 2]])

        # Dec Crossover
        off_pop = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)

        # Mask Crossover
        mask_p1 = self.mask[p1_idx]
        mask_p2 = self.mask[p2_idx]
        diff = mask_p1 != mask_p2

        # Tournament for bit inheritance
        rand_neighbor_fit = self.var_fit[torch.randint(0, D, (N // 2, D), device=device)]
        inherit_p1 = self.var_fit.expand(N // 2, D) < rand_neighbor_fit

        off_mask_half = torch.where(diff, torch.where(inherit_p1, mask_p1, mask_p2), mask_p1)
        off_mask = torch.cat([off_mask_half, off_mask_half], dim=0)
        off_mask = off_mask[:N]  # Match N

        # 2. Grouped Mutation
        off_pop, off_mask = self._grouped_mutation(off_pop, off_mask)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # 3. Evaluation
        off_fit = self.evaluate(off_pop * off_mask)

        # 4. Environmental Selection
        all_pop = torch.cat([self.pop, off_pop], dim=0)
        all_fit = torch.cat([self.fit, off_fit], dim=0)
        all_mask = torch.cat([self.mask, off_mask], dim=0)

        # Unique Check (Bug #3)
        merged_pop, u_idx = unique_rows_sorted(all_pop)
        merged_fit = all_fit[u_idx]
        merged_mask = all_mask[u_idx]

        # Non-Dominated Sort
        rank = non_dominate_rank(merged_fit)

        # Integrated Peeling (Bug #9)
        num_merged = merged_pop.shape[0]
        selected_indices = torch.zeros(N, dtype=torch.long, device=device)
        count = 0
        current_rank = 0

        # Sentinel for loop (JIT requires fixed or bounded iterations)
        # Max fronts is num_merged
        for _ in range(num_merged):
            if count >= N:
                pass
            else:
                curr_front_mask = rank == current_rank
                num_in_front = curr_front_mask.sum()

                # Deadlock Breaker
                if num_in_front == 0:
                    # Fill with whatever is left that isn't selected
                    # (In practice, rank will cover all)
                    count = N
                else:
                    dist = crowding_distance(merged_fit, curr_front_mask)

                    if count + num_in_front <= N:
                        # Select all in front
                        front_indices = torch.where(curr_front_mask)[0]
                        selected_indices[count : count + num_in_front] = front_indices
                        count += num_in_front
                    else:
                        # Select top needed by crowding distance
                        needed = N - count
                        front_indices = torch.where(curr_front_mask)[0]
                        front_dist = dist[front_indices]
                        # Lexsort: primary key (rank) last, but here rank is constant
                        # so we just sort by -dist
                        sub_idx = torch.argsort(-front_dist)[:needed]
                        selected_indices[count : count + needed] = front_indices[sub_idx]
                        count = N
                current_rank += 1

        self.pop = merged_pop[selected_indices]
        self.fit = merged_fit[selected_indices]
        self.mask = merged_mask[selected_indices]
        self.rank = rank[selected_indices]
        # Re-calculate distance for the new population for next mating
        # Note: This is done per front in selection, we store it here
        final_dist = torch.zeros(N, device=device)
        for r in range(current_rank):
            m = self.rank == r
            if m.any():
                final_dist[m] = crowding_distance(self.fit, m)[m]
        self.dis = final_dist


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SparseEA2(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
