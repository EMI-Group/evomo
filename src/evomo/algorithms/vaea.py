import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, nanmax, nanmin

from evomo.operators.selection import non_dominate_rank


class VaEA(Algorithm):
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

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)  # [N,M]

    def _cosine_distance_matrix(self, x: torch.Tensor) -> torch.Tensor:
        # [Bug #12] Safe Division & [Bug #29] Vectorization
        norm = torch.norm(x, dim=1, keepdim=True)
        similarity = (x @ x.t()) / (norm @ norm.t() + 1e-6)
        # Clamp to prevent NaNs in acos
        return torch.acos(similarity.clamp(-1 + 1e-7, 1 - 1e-7))

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size
        M = self.n_objs

        # 1. Mating (Binary Tournament Selection on sum of objectives)
        scalar_fit = torch.sum(self.fit, dim=1)
        mating_pool = tournament_selection_multifit(N, [scalar_fit], tournament_size=2)
        parents = self.pop[mating_pool]

        offspring = simulated_binary(parents, pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(offspring)

        # 3. Environmental Selection
        C_pop = torch.cat([self.pop, offspring], dim=0)
        C_fit = torch.cat([self.fit, off_fit], dim=0)

        # NDSort
        rank = non_dominate_rank(C_fit)

        # Front Peeling
        select_mask = torch.zeros(2 * N, dtype=torch.bool, device=device)
        current_rank = torch.zeros((), dtype=torch.int32, device=device)

        # Identify fronts until we exceed N
        num_selected = 0
        while num_selected < N:
            mask_rank = rank == current_rank
            count_rank = torch.sum(mask_rank.int())
            # If adding this front exceeds N, this is our F_k
            if num_selected + count_rank >= N:
                break
            select_mask = select_mask | mask_rank
            num_selected += count_rank
            current_rank += 1

        # Normalization
        z_min = nanmin(C_fit, dim=0)[0]
        z_max = nanmax(C_fit, dim=0)[0]
        norm_fit = (C_fit - z_min) / (z_max - z_min + 1e-6)
        conv = torch.sum(norm_fit, dim=1)

        # Angle Matrix
        angle_mat = self._cosine_distance_matrix(norm_fit)

        # Initial Selection (Extreme solutions and best convergence)
        select_idx = torch.where(select_mask)[0]

        if select_idx.shape[0] == 0:
            # Extreme solutions for each objective
            extremes = torch.argmin(norm_fit, dim=0)
            # Best convergence solutions
            _, best_conv_idx = torch.topk(conv, k=M, largest=False)
            select_idx = torch.unique(torch.cat([extremes, best_conv_idx]))

        # F_k candidates
        f_k_mask = rank == current_rank
        in_select = torch.zeros(2 * N, dtype=torch.bool, device=device)
        in_select[select_idx] = True
        remain_mask = f_k_mask & (~in_select)

        # Niching Loop
        pi = 3.141592653589793
        threshold = pi / (2 * (N + 1))

        for _ in range(2 * N):
            if select_idx.shape[0] >= N:
                break

            remain_idx = torch.where(remain_mask)[0]
            if remain_idx.shape[0] == 0:
                not_selected_mask = ~in_select
                not_selected_idx = torch.where(not_selected_mask)[0]
                needed = N - select_idx.shape[0]
                if not_selected_idx.shape[0] > 0:
                    _, add_idx = torch.topk(conv[not_selected_idx], k=min(needed, not_selected_idx.shape[0]), largest=False)
                    select_idx = torch.cat([select_idx, not_selected_idx[add_idx]])
                break

            # Max-Min Angle Selection
            sub_angles = angle_mat[remain_idx][:, select_idx]
            min_angles, _ = torch.min(sub_angles, dim=1)
            best_in_remain = torch.argmax(min_angles)
            best_candidate = remain_idx[best_in_remain]

            # Worse Elimination
            angles_to_selected = angle_mat[best_candidate, select_idx]
            nn_local_idx = torch.argmin(angles_to_selected)
            nearest_neighbor = select_idx[nn_local_idx]

            if angles_to_selected[nn_local_idx] < threshold:
                if conv[best_candidate] < conv[nearest_neighbor]:
                    # Replace nearest neighbor
                    select_idx[nn_local_idx] = best_candidate
                    in_select[nearest_neighbor] = False
                    in_select[best_candidate] = True
                remain_mask[best_candidate] = False
            else:
                # Add candidate
                select_idx = torch.cat([select_idx, best_candidate.view(-1)])
                in_select[best_candidate] = True
                remain_mask[best_candidate] = False

        final_idx = select_idx[:N]
        self.pop = C_pop[final_idx]
        self.fit = C_fit[final_idx]


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = VaEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
