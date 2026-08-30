import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class BiGE(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)  # [N,D]
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))  # [N,M]

    def _calculate_bi_goal(self, fit: torch.Tensor) -> torch.Tensor:
        N, M = fit.shape
        # 1. Normalization (Bug #12)
        f_min = fit.min(dim=0, keepdim=True)[0]
        f_max = fit.max(dim=0, keepdim=True)[0]
        norm_fit = (fit - f_min) / (f_max - f_min + 1e-6)

        # 2. Proximity (fpr)
        fpr = norm_fit.sum(dim=1)  # [N]

        # 3. Crowding Degree (fcd)
        dist_mat = torch.cdist(norm_fit, norm_fit, p=2)  # [N, N]

        # Niche radius (Bug #12)
        r = 1.0 / (N ** (1.0 / M) + 1e-6)
        is_neighbor = dist_mat < r

        # Asymmetric Sharing (Bug #29: No loops)
        better_prox = fpr.unsqueeze(1) <= fpr.unsqueeze(0)  # [N, N]

        # sh calculation
        dist_ratio = dist_mat / (r + 1e-6)
        sh_better = (0.5 * (1 - dist_ratio)) ** 2
        sh_worse = (1.5 * (1 - dist_ratio)) ** 2

        sh = torch.where(better_prox, sh_better, sh_worse)
        sh = torch.where(is_neighbor, sh, torch.zeros_like(sh))

        fcd = torch.sqrt(sh.sum(dim=1))  # [N]

        return torch.stack([fpr, fcd], dim=1)

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size

        # 1. Mating Selection
        bi_fit = self._calculate_bi_goal(self.fit)
        # bi_fit contains [fpr, fcd], both to be minimized
        mating_idx = tournament_selection_multifit(N, [bi_fit[:, 0], bi_fit[:, 1]], tournament_size=2)

        # 2. Variation
        offspring = simulated_binary(self.pop[mating_idx], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub, pro_m=1.0 / self.lb.numel(), dis_m=20.0)
        offspring = clamp(offspring, self.lb, self.ub)

        # 3. Evaluation
        off_fit = self.evaluate(offspring)

        # 4. Environmental Selection
        total_pop = torch.cat([self.pop, offspring], dim=0)
        total_fit = torch.cat([self.fit, off_fit], dim=0)

        # Tier 1: Objective Space NDSort
        ranks = non_dominate_rank(total_fit)

        # Identify fronts
        # We need to find the threshold rank that includes N individuals
        sorted_ranks, _ = torch.sort(ranks)
        threshold_rank = sorted_ranks[N - 1]

        mask_before = ranks < threshold_rank
        mask_last = ranks == threshold_rank

        num_before = torch.sum(mask_before.int())
        num_needed = N - num_before

        # Tier 2: Bi-Goal Space for the Last Front
        # Extract individuals in the last front
        last_front_indices = torch.where(mask_last)[0]
        fit_last = total_fit[last_front_indices]
        bi_fit_last = self._calculate_bi_goal(fit_last)

        # NDSort on bi-goal indicators
        ranks_bi = non_dominate_rank(bi_fit_last)

        # Final selection indices
        # For the last front, we sort by rank_bi and then random for tie-breaking (Bug #14, #25)
        rand_vals = torch.rand(bi_fit_last.shape[0], device=device)
        # lexsort: primary key last
        last_front_sort_idx = lexsort(torch.stack([rand_vals, ranks_bi.float()]))
        selected_from_last = last_front_indices[last_front_sort_idx[:num_needed]]

        selected_indices = torch.cat([torch.where(mask_before)[0], selected_from_last])

        self.pop = total_pop[selected_indices]
        self.fit = total_fit[selected_indices]


# === FIXED DEMO BLOCK ===
# This block MUST be appended at the end of the file.
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    # BiGE must be replaced by your actual class name
    algo = BiGE(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
