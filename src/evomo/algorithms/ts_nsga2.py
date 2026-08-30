import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class TSNSGAII(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, max_fe: int = 10000):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.max_fe = max_fe
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Reference Weights
        w, _ = uniform_sampling(pop_size, n_objs)
        self.w = Mutable(w.to(device))

        # Selection Keys
        self.front_no = Mutable(torch.zeros(pop_size, dtype=torch.int32, device=device))
        self.d2 = Mutable(torch.zeros(pop_size, device=device))
        self.fe = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.fe = self.fe + self.pop_size
        # Initial Selection to populate keys
        self.pop, self.fit, self.front_no, self.d2 = self._environmental_selection(self.pop, self.fit)

    def step(self) -> None:
        # 1. Mating
        # Lexsort keys: Primary (front_no) must be last. Secondary (d2) first.
        mating_pool = tournament_selection_multifit(self.pop_size, [self.d2, self.front_no.float()], tournament_size=2)
        offspring = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub, pro_m=1.0 / self.lb.numel(), dis_m=20.0)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(offspring)
        self.fe = self.fe + self.pop_size

        # 3. Selection
        merged_pop = torch.cat([self.pop, offspring], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit, self.front_no, self.d2 = self._environmental_selection(merged_pop, merged_fit)

    def _environmental_selection(self, pop, fit):
        N = self.pop_size
        M = self.n_objs

        # Normalization (Bug #12)
        f_min = torch.min(fit, dim=0)[0]
        f_max = torch.max(fit, dim=0)[0]
        norm_fit = (fit - f_min) / (f_max - f_min + 1e-6)

        # Stage Transition Logic
        if self.fe < (self.max_fe // 2):
            # Stage 1: SPD-Sort
            # Association
            norm_val = torch.norm(norm_fit, dim=1, keepdim=True)
            w_norm = torch.norm(self.w, dim=1, keepdim=True)
            cosine = (norm_fit @ self.w.T) / (norm_val @ w_norm.T + 1e-6)

            # d1 and d2
            d1_all = norm_val * cosine
            d2_all = norm_val * torch.sqrt(clamp(1 - cosine**2, 0.0, 1.0) + 1e-6)

            # Find closest weight for each individual
            closest_w_idx = torch.argmin(d2_all, dim=1)
            idx_range = torch.arange(pop.shape[0], device=pop.device)

            d1 = d1_all[idx_range, closest_w_idx]
            d2 = d2_all[idx_range, closest_w_idx]
            spd_score = d1 + 5 * d2

            front_no = self._spd_sort(fit, spd_score, closest_w_idx)

            # Selection via Lexsort
            rank_indices = lexsort(torch.stack([spd_score, front_no.float()]))
            survivor_idx = rank_indices[:N]

            return pop[survivor_idx], fit[survivor_idx], front_no[survivor_idx], d2[survivor_idx]

        else:
            # Stage 2: Level-Sort & Decomposition
            # Interval-based levels
            interval = (f_max - f_min) / (2 * M + 1e-6)
            torch.max((fit - f_min) / (interval + 1e-6), dim=1)[0].floor()

            # Greedy Selection (Vectorized)
            survivor_idx = self._level_sort_selection(norm_fit, self.w, N)

            # Re-calculate front_no and d2 for the next mating step
            new_fit = fit[survivor_idx]
            new_norm = norm_fit[survivor_idx]
            new_front = non_dominate_rank(new_fit)

            # Update d2 for mating
            new_norm_val = torch.norm(new_norm, dim=1, keepdim=True)
            new_cosine = (new_norm @ self.w.T) / (new_norm_val @ torch.norm(self.w, dim=1, keepdim=True).T + 1e-6)
            new_d2 = torch.min(new_norm_val * torch.sqrt(clamp(1 - new_cosine**2, 0.0, 1.0) + 1e-6), dim=1)[0]

            return pop[survivor_idx], new_fit, new_front, new_d2

    def _spd_sort(self, fit, spd_score, w_idx):
        # Bug #24: Pareto Dominance
        pareto_dom = (fit.unsqueeze(1) <= fit.unsqueeze(0)).all(-1) & (fit.unsqueeze(1) < fit.unsqueeze(0)).any(-1)

        # SPD Dominance: better score on the same weight
        same_w = w_idx.unsqueeze(1) == w_idx.unsqueeze(0)
        dist_dom = (spd_score.unsqueeze(1) < spd_score.unsqueeze(0)) & same_w

        total_dom = pareto_dom | dist_dom

        # Peeling logic (Vectorized Rank Assignment)
        num_dominated = torch.sum(total_dom, dim=0)
        return num_dominated.int()

    def _level_sort_selection(self, norm_fit, w, n_required):
        # Vectorized Greedy Selection
        # Calculate cosine similarity between all individuals and all weights
        norm_val = torch.norm(norm_fit, dim=1, keepdim=True)
        w_norm = torch.norm(w, dim=1, keepdim=True)
        cosine = (norm_fit @ w.T) / (norm_val @ w_norm.T + 1e-6)  # [2N, N]

        # For each weight, find the individual with max cosine similarity
        best_match_for_w = torch.argmax(cosine, dim=0)  # [N]

        # Handle potential duplicates if one individual is best for multiple weights
        # But for TSNSGAII Stage 2, we typically take the best for each weight
        # If N_weights == N_required, this naturally selects N individuals
        u_idx, _ = unique_rows_sorted(best_match_for_w.unsqueeze(1))
        u_idx = u_idx.squeeze(1)

        # If not enough unique individuals, pad with remaining best by cosine
        # This is a JIT-safe way to ensure we return exactly N
        mask = torch.zeros(norm_fit.shape[0], dtype=torch.bool, device=norm_fit.device)
        mask[u_idx] = True

        remaining_indices = torch.where(~mask)[0]
        # Sort remaining by max similarity to any weight
        max_sim_to_any = torch.max(cosine, dim=1)[0]
        rem_sorted = remaining_indices[torch.argsort(max_sim_to_any[remaining_indices], descending=True)]

        combined = torch.cat([u_idx, rem_sorted], dim=0)
        return combined[:n_required]


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = TSNSGAII(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
