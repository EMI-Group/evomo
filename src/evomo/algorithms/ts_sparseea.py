import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp, lexsort, randint

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class TSSparseEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()
        self.dim = D

        # Stage 1 Grouping Parameters
        self.nGroup = 50
        group_size = (D + self.nGroup - 1) // self.nGroup
        sentinel = torch.iinfo(torch.int32).max

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.mask = Mutable(torch.ones((pop_size, D), dtype=torch.bool, device=device))
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.rank = Mutable(torch.zeros(pop_size, dtype=torch.int32, device=device))
        self.dis = Mutable(torch.zeros(pop_size, device=device))
        self.is_initialized = Mutable(torch.tensor(False, device=device))
        self.subcomponents = Mutable(torch.full((self.nGroup, group_size), sentinel, dtype=torch.int64, device=device))

    def _match_operator(self, off_mask: torch.Tensor, archive_pop: torch.Tensor) -> torch.Tensor:
        # Vectorized Cosine Similarity (Bug #12, #29 Compliance)
        mask_f = off_mask.float()
        dot = mask_f @ archive_pop.t()
        norm_m = torch.norm(mask_f, dim=1, keepdim=True)
        norm_p = torch.norm(archive_pop, dim=1, keepdim=True)
        sim = dot / (norm_m @ norm_p.t() + 1e-6)
        best_match_idx = torch.argmax(sim, dim=1)
        return archive_pop[best_match_idx]

    def init_step(self) -> None:
        device = self.lb.device
        D = self.dim

        # 1. Variable Ranking via Stage 1 Warm-up
        # Evaluate D individuals (one-hot masks)
        warmup_pop = self.lb.repeat(D, 1)
        # For ranking, we use the upper bound for the active variable
        warmup_pop[torch.arange(D), torch.arange(D)] = self.ub[torch.arange(D)]
        warmup_fit = self.evaluate(warmup_pop)

        ranks = non_dominate_rank(warmup_fit)
        rank_indices = torch.argsort(ranks)

        # 2. Grouping (Bug #1 Compliance)
        group_size = self.subcomponents.shape[1]
        sentinel = torch.iinfo(torch.int32).max

        # Fill subcomponents using vectorized scatter if possible, or simple slice fill
        # Since JIT requires fixed shapes, we pad the rank_indices
        padded_indices = torch.full((self.nGroup * group_size,), sentinel, dtype=torch.int64, device=device)
        padded_indices[:D] = rank_indices
        self.subcomponents = padded_indices.view(self.nGroup, group_size)

        # Initial Evaluation
        self.fit = self.evaluate(self.pop * self.mask.float())

        # Initial Environmental Selection
        self.pop, self.fit, self.rank, self.dis = self._environmental_selection(self.pop, self.mask, self.fit, self.pop_size)
        self.is_initialized = torch.tensor(True, device=device)

    def _environmental_selection(self, pop, mask, fit, N):
        # Unique Filter (Bug #3)
        _, uni_idx = unique_rows_sorted(fit)
        fit, pop, mask = fit[uni_idx], pop[uni_idx], mask[uni_idx]

        ranks = non_dominate_rank(fit)

        # JIT-Compliant Peeling (Bug #9, #28)
        selected_indices = torch.full((N,), -1, dtype=torch.int64, device=pop.device)
        final_rank = torch.zeros(N, dtype=torch.int32, device=pop.device)
        final_dis = torch.zeros(N, device=pop.device)

        num_selected = 0
        current_front = 0

        # Max fronts is N, but we use a fixed loop for JIT
        for _ in range(N):
            if num_selected >= N:
                pass
            else:
                mask_front = ranks == current_front
                num_in_front = torch.sum(mask_front.int())

                # Crowding distance within front (Bug #21)
                dist = crowding_distance(fit, mask_front)

                if num_selected + num_in_front <= N:
                    # Select all
                    indices = torch.where(
                        mask_front, torch.arange(len(fit), device=pop.device), torch.tensor(-1, device=pop.device)
                    )
                    # Filter out -1s using topk to get valid indices
                    _, idx = torch.topk((indices != -1).long(), num_in_front.int())
                    valid_indices = indices[idx]

                    selected_indices[num_selected : num_selected + num_in_front] = valid_indices
                    final_rank[num_selected : num_selected + num_in_front] = current_front
                    final_dis[num_selected : num_selected + num_in_front] = dist[valid_indices]
                    num_selected += num_in_front
                else:
                    # Select topk by distance
                    needed = N - num_selected
                    front_indices = torch.where(mask_front)[0]
                    front_dists = dist[front_indices]
                    _, top_k_idx = torch.topk(front_dists, needed)

                    selected_indices[num_selected:N] = front_indices[top_k_idx]
                    final_rank[num_selected:N] = current_front
                    final_dis[num_selected:N] = front_dists[top_k_idx]
                    num_selected = N

                current_front += 1

        # Final Sort (Bug #25)
        # Primary key (Rank) last
        sort_idx = lexsort(torch.stack([-final_dis, final_rank.float()]))
        sel_idx = selected_indices[sort_idx]

        return pop[sel_idx], fit[sel_idx], final_rank[sort_idx], final_dis[sort_idx]

    def step(self) -> None:
        N = self.pop_size
        D = self.dim
        device = self.lb.device

        # 1. Selection (Bug #27, #31)
        mating_pool = tournament_selection_multifit(N, [self.rank, -self.dis], tournament_size=2)

        # 2. Mask Variation (Stage 2 Logic)
        parent_mask = self.mask[mating_pool]
        # One-point crossover
        off_mask = parent_mask.clone()
        cp = randint(0, D, (N // 2,), device=device)
        for i in range(N // 2):
            idx1, idx2 = i, i + N // 2
            mask_range = torch.arange(D, device=device) > cp[i]
            off_mask[idx1] = torch.where(mask_range, parent_mask[idx2], parent_mask[idx1])
            off_mask[idx2] = torch.where(mask_range, parent_mask[idx1], parent_mask[idx2])

        # Bit-flip mutation (Bug #20)
        prob = 1.0 / D
        flip_mask = torch.rand(N, D, device=device) < prob
        off_mask = torch.where(flip_mask, ~off_mask, off_mask)

        # 3. Dec Variation (SBX + PM)
        parent_pop = self.pop[mating_pool]
        off_pop_dec = simulated_binary(parent_pop, pro_c=1.0, dis_c=20.0)
        off_pop_dec = polynomial_mutation(off_pop_dec, self.lb, self.ub, pro_m=1.0 / D, dis_m=20.0)
        off_pop_dec = clamp(off_pop_dec, self.lb, self.ub)

        # 4. Match Operator (The Core Logic - Bug #29)
        matched_pop = self._match_operator(off_mask, self.pop)
        # Combine matched decision variables with new variations
        off_pop = torch.where(torch.rand(N, D, device=device) < 0.5, off_pop_dec, matched_pop)

        # 5. Evaluation
        off_fit = self.evaluate(off_pop * off_mask.float())

        # 6. Merge & Environmental Selection
        combined_pop = torch.cat([self.pop, off_pop], dim=0)
        combined_mask = torch.cat([self.mask, off_mask], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit, self.rank, self.dis = self._environmental_selection(combined_pop, combined_mask, combined_fit, N)


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = TSSparseEA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
