import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class NSBiDiCo(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Archive State
        self.archive_pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.archive_fit = self.evaluate(self.archive_pop)

    def _calculate_cosine_rank(self, objs: torch.Tensor) -> torch.Tensor:
        # Normalization (Bug #12)
        f_min = torch.min(objs, dim=0, keepdim=True)[0]
        f_max = torch.max(objs, dim=0, keepdim=True)[0]
        norm_obj = (objs - f_min) / (f_max - f_min + 1e-10) + 1e-10

        # Cosine Similarity Matrix (Vectorized)
        norms = torch.norm(norm_obj, dim=1, keepdim=True)
        sim_matrix = (norm_obj @ norm_obj.T) / (norms @ norms.T + 1e-6)
        angle_matrix = torch.acos(torch.clamp(sim_matrix, -1.0, 1.0))

        # Angle Rank: Rank based on proximity in objective space
        sorted_angle, _ = torch.sort(angle_matrix, dim=1)
        # We use the sum of proximity ranks or the sorted values to determine density
        # Following Blueprint: Rank based on proximity
        angle_rank = torch.argsort(torch.argsort(sorted_angle, dim=0), dim=0).float().sum(dim=1)
        return angle_rank

    def _update_archive(self, pop, fit, arc_pop, arc_fit, off_pop, off_fit):
        combined_pop = torch.cat([pop, arc_pop, off_pop], dim=0)
        combined_fit = torch.cat([fit, arc_fit, off_fit], dim=0)

        # Unique rows to keep archive clean (Bug #3)
        u_pop, u_idx = unique_rows_sorted(combined_pop)
        u_fit = combined_fit[u_idx]

        ranks = non_dominate_rank(u_fit)
        mask_f1 = ranks == 0
        front1_pop = u_pop[mask_f1]
        front1_fit = u_fit[mask_f1]

        num_f1 = front1_pop.shape[0]

        if num_f1 <= self.pop_size:
            # Pad with remaining if necessary (though usually archive is just Front 1)
            return front1_pop, front1_fit
        else:
            # Greedy Peeling Vectorized (Bug #9, #41)
            # Calculate density via Mean Cosine Similarity
            f_min = torch.min(front1_fit, dim=0, keepdim=True)[0]
            f_max = torch.max(front1_fit, dim=0, keepdim=True)[0]
            norm_f1 = (front1_fit - f_min) / (f_max - f_min + 1e-10) + 1e-10
            norms = torch.norm(norm_f1, dim=1, keepdim=True)
            sim_matrix = (norm_f1 @ norm_f1.T) / (norms @ norms.T + 1e-6)

            mean_sim = sim_matrix.sum(dim=1)
            _, drop_idx = torch.topk(mean_sim, k=num_f1 - self.pop_size)

            keep_mask = torch.ones(num_f1, dtype=torch.bool, device=self.lb.device)
            keep_mask[drop_idx] = False
            return front1_pop[keep_mask], front1_fit[keep_mask]

    def step(self) -> None:
        # 1. Bidirectional Mating Selection
        # Combine pop and archive for normalization context
        combined_fit = torch.cat([self.fit, self.archive_fit], dim=0)
        angle_ranks = self._calculate_cosine_rank(combined_fit)

        pop_ranks = angle_ranks[: self.pop_size]
        arc_ranks = angle_ranks[self.pop_size :]

        # Tournament Selection (Bug #27, #31)
        idx1 = tournament_selection_multifit(self.pop_size, [pop_ranks], tournament_size=2)
        idx2 = tournament_selection_multifit(self.pop_size, [arc_ranks], tournament_size=2)

        parents = torch.cat([self.pop[idx1], self.archive_pop[idx2]], dim=0)

        # 2. Variation
        offspring = simulated_binary(parents, pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        off_fit = self.evaluate(offspring)

        # 3. Archive Update
        new_arc_pop, new_arc_fit = self._update_archive(
            self.pop, self.fit, self.archive_pop, self.archive_fit, offspring, off_fit
        )
        # Ensure fixed size for Mutable
        self.archive_pop[: new_arc_pop.shape[0]] = new_arc_pop
        self.archive_fit[: new_arc_fit.shape[0]] = new_arc_fit

        # 4. Environmental Selection (NSGA-II Style)
        merged_pop = torch.cat([self.pop, offspring], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)

        ranks = non_dominate_rank(merged_fit)

        new_pop = torch.empty_like(self.pop)
        new_fit = torch.empty_like(self.fit)
        count = 0

        # JIT-friendly loop for environmental selection
        for r in range(merged_pop.shape[0]):
            mask = ranks == r
            num_in_front = torch.sum(mask).int()

            if count + num_in_front <= self.pop_size:
                new_pop[count : count + num_in_front] = merged_pop[mask]
                new_fit[count : count + num_in_front] = merged_fit[mask]
                count += num_in_front
            else:
                remaining = self.pop_size - count
                if remaining > 0:
                    # Crowding distance ONLY for this front (Bug #21, #28)
                    dist = crowding_distance(merged_fit, mask)
                    # Lexsort (Bug #25): Primary key (rank) is implicit here as we are in one front
                    # We sort by distance descending
                    front_indices = torch.where(mask)[0]
                    sub_idx = torch.argsort(dist[mask], descending=True)
                    sel_idx = front_indices[sub_idx[:remaining]]

                    new_pop[count : self.pop_size] = merged_pop[sel_idx]
                    new_fit[count : self.pop_size] = merged_fit[sel_idx]
                    count += remaining

            if count >= self.pop_size:
                break

        self.pop = new_pop
        self.fit = new_fit


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = NSBiDiCo(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
