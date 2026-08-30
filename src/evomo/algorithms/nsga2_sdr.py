import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp

from evomo.utils import unique_rows_sorted


def _calculate_sdr_dominance_matrix(norm_sum: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # Bug #7: Implement SDR Dominance using broadcasting
    return (norm_sum.view(-1, 1) * theta) < norm_sum.view(1, -1)


class NSGAII_SDR(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        self.zmin = Mutable(torch.full((n_objs,), torch.inf, device=device))
        self.zmax = Mutable(torch.full((n_objs,), -torch.inf, device=device))

        sentinel = torch.iinfo(torch.int32).max
        self.front_no = Mutable(torch.full((pop_size,), sentinel, dtype=torch.int32, device=device))
        self.crowd_dis = Mutable(torch.full((pop_size,), -1.0, dtype=torch.float32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.zmin = torch.min(self.fit, dim=0)[0]
        self.zmax = torch.max(self.fit, dim=0)[0]
        self._environmental_selection(self.pop, self.fit)

    def _environmental_selection(self, combined_pop: torch.Tensor, combined_fit: torch.Tensor):
        device = combined_fit.device
        N = self.pop_size

        # 1. Unique Filtering (Bug #3, #17)
        rounded_fit = torch.round(combined_fit * 1e6) / 1e6
        _, unique_indices = unique_rows_sorted(rounded_fit)
        fit = combined_fit[unique_indices]
        pop = combined_pop[unique_indices]

        # 2. Z-Update
        self.zmin = torch.min(self.zmin, torch.min(fit, dim=0)[0])

        # 3. Normalization Trigger (Bug #18)
        f_min = self.zmin
        f_max = torch.max(fit, dim=0)[0]
        f_range = f_max - f_min

        if 0.05 * torch.max(f_range) < torch.min(f_range):
            norm_fit = (fit - f_min) / (f_range + 1e-6)
        else:
            norm_fit = fit - f_min

        # 4. Core Metric Calculation (SDR Matrix)
        norm_p = torch.norm(norm_fit, p=2, dim=1)
        cosine = (norm_fit @ norm_fit.T) / (norm_p.unsqueeze(1) * norm_p.unsqueeze(0) + 1e-6)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        acos_matrix = torch.acos(cosine)

        # Bug #17: Angle(eye)=Pi/2. Avoid fill_diagonal_ for JIT compliance.
        eye = torch.eye(acos_matrix.shape[0], device=device)
        acos_matrix = acos_matrix * (1 - eye) + 1.570796 * eye

        # Adaptive Theta (Bug #2, #17)
        min_angles = torch.min(acos_matrix, dim=1)[0]
        unique_min_angles, _ = unique_rows_sorted(min_angles.unsqueeze(1))
        unique_min_angles = unique_min_angles.squeeze(1)

        # Bug #2: Median index. Fix TypeError by ensuring clamp input is a Tensor.
        num_unique = unique_min_angles.shape[0]
        mid_idx_val = (num_unique + 1) // 2
        mid_idx = torch.tensor(mid_idx_val, device=device).clamp(0, num_unique - 1)
        minA = unique_min_angles[mid_idx]
        theta = torch.clamp(acos_matrix / (minA + 1e-6), min=1.0)

        # SDR Dominance (Bug #7)
        norm_sum = torch.sum(norm_fit, dim=1)
        sdr_dom = _calculate_sdr_dominance_matrix(norm_sum, theta)

        # 5. Integrated Peeling (Bug #9, #41)
        num_total = fit.shape[0]
        remaining_mask = torch.ones(num_total, dtype=torch.bool, device=device)
        final_indices = torch.zeros(N, dtype=torch.long, device=device)
        final_front_no = torch.zeros(N, dtype=torch.int32, device=device)
        final_crowd_dis = torch.zeros(N, dtype=torch.float32, device=device)

        count = 0
        front_idx = 1

        # Use a fixed loop for JIT. Max iterations is num_total.
        for _ in range(num_total):
            active = count < N
            if active:
                # SDR Peeling logic
                is_dominated = (sdr_dom & remaining_mask.unsqueeze(1)).any(dim=0)
                current_front_mask = remaining_mask & (~is_dominated)

                num_in_front = current_front_mask.sum()

                # Deadlock Breaker (Bug #9)
                if num_in_front == 0 and remaining_mask.any():
                    current_front_mask = remaining_mask
                    num_in_front = current_front_mask.sum()

                if num_in_front > 0:
                    if count + num_in_front <= N:
                        # Add whole front
                        indices = torch.where(current_front_mask)[0]
                        cd = crowding_distance(norm_fit, current_front_mask)

                        # Vectorized assignment to pre-allocated tensors
                        num_to_add = indices.shape[0]
                        fill_slice = torch.arange(count, count + num_to_add, device=device)
                        final_indices[fill_slice] = indices
                        final_front_no[fill_slice] = front_idx
                        final_crowd_dis[fill_slice] = cd[indices]

                        if front_idx == 1:
                            self.zmax = torch.max(fit[current_front_mask], dim=0)[0]

                        count += num_to_add
                        remaining_mask = remaining_mask & (~current_front_mask)
                        front_idx += 1
                    else:
                        # Last front slicing
                        indices = torch.where(current_front_mask)[0]
                        cd = crowding_distance(norm_fit, current_front_mask)
                        front_cd = cd[indices]

                        needed = N - count
                        _, sort_idx = torch.sort(front_cd, descending=True)
                        sel_indices = indices[sort_idx[:needed]]

                        fill_slice = torch.arange(count, N, device=device)
                        final_indices[fill_slice] = sel_indices
                        final_front_no[fill_slice] = front_idx
                        final_crowd_dis[fill_slice] = front_cd[sort_idx[:needed]]

                        count = N
                        remaining_mask.fill_(False)

        self.pop = pop[final_indices]
        self.fit = fit[final_indices]
        self.front_no = final_front_no
        self.crowd_dis = final_crowd_dis

    def step(self) -> None:
        # 1. Selection (Bug #25, #27)
        mating_pool = tournament_selection_multifit(
            self.pop_size, fitnesses=[-self.crowd_dis, self.front_no.float()], tournament_size=2
        )

        # 2. Variation
        crossovered = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(crossovered, self.lb, self.ub, pro_m=1.0 / self.lb.numel(), dis_m=20.0)
        offspring = clamp(offspring, self.lb, self.ub)

        # 3. Evaluation
        off_fit = self.evaluate(offspring)

        # 4. Environmental Selection
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)
        self._environmental_selection(merge_pop, merge_fit)


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = NSGAII_SDR(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
