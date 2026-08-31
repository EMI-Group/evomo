import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class AGEMOEA(Algorithm):
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

        sentinel = torch.iinfo(torch.int32).max
        self.front_no = Mutable(torch.full((pop_size,), sentinel, dtype=torch.int32, device=device))
        self.crowd_dis = Mutable(torch.zeros(pop_size, device=device))
        self.ideal_point = Mutable(torch.full((n_objs,), torch.inf, device=device))
        self.p = Mutable(torch.ones(1, device=device))
        self.normalization = Mutable(torch.ones(n_objs, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self._environmental_selection(self.pop, self.fit)

    def step(self) -> None:
        # 1. Mating
        # Primary key: front_no (min), Secondary key: crowd_dis (max)
        mating_pool = tournament_selection_multifit(self.pop_size, [self.front_no.float(), -self.crowd_dis], tournament_size=2)

        crossovered = simulated_binary(self.pop[mating_pool])
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(offspring)

        # 3. Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)
        self._environmental_selection(combined_pop, combined_fit)

    def _environmental_selection(self, pop, fit):
        N, M = fit.shape
        device = fit.device

        # Pre-processing
        objs = torch.round(fit * 1e6) / 1e6
        self.ideal_point = torch.min(self.ideal_point, torch.min(objs, dim=0)[0])
        f = objs - self.ideal_point

        # NDSort
        front_no = non_dominate_rank(f)

        # Corner Detection & Normalization (Front 1)
        mask_f1 = front_no == 0
        if mask_f1.any():
            f_f1 = f[mask_f1]
        else:
            f_f1 = f  # Fallback

        W = torch.eye(M, device=device) + 1e-6
        dist_to_axes = self._point_to_line_dist(f_f1, W)
        extreme_indices_in_f1 = torch.argmin(dist_to_axes, dim=0)
        extreme_f = f_f1[extreme_indices_in_f1]

        # Solve for intercepts
        try:
            # Solve extreme_f * x = 1
            x, _, _, _ = torch.linalg.lstsq(extreme_f, torch.ones((M, 1), device=device))
            a = 1.0 / (x.view(-1) + 1e-12)
        except RuntimeError:
            a = torch.full((M,), -1.0, device=device)

        # Fallback for intercepts
        invalid_a = (a <= 1e-6) | torch.isnan(a) | torch.isinf(a)
        a = torch.where(invalid_a, torch.max(f_f1, dim=0)[0] + 1e-6, a)
        self.normalization = a

        # Adaptive p Estimation
        f_norm = f / (a + 1e-6)
        f_norm_f1 = f_norm[mask_f1]

        # p estimation logic from source
        d_to_diag = self._point_to_line_dist(f_norm_f1, torch.ones((1, M), device=device))
        # Set extreme points to inf to find the "middle" point
        d_to_diag_mod = d_to_diag.clone()
        d_to_diag_mod[extreme_indices_in_f1] = torch.inf
        mid_idx = torch.argmin(d_to_diag_mod)

        avg_val = torch.mean(f_norm_f1[mid_idx])
        p = torch.log(torch.tensor(float(M), device=device)) / torch.log(1.0 / (avg_val + 1e-6))
        p = torch.where(torch.isnan(p) | (p <= 0.1), torch.tensor(1.0, device=device), p)
        self.p = torch.clamp(p, min=0.1, max=20.0)

        # Selection Loop
        crowd_dis = torch.zeros(N, device=device)
        max_front = torch.max(front_no)

        # Iterate fronts to calculate crowding distance
        for i in range(int(max_front) + 1):
            in_front = front_no == i
            if not in_front.any():
                continue

            if i == 0:
                crowd_dis[in_front] = self._calculate_front1_diversity(f_norm[in_front], self.p, extreme_indices_in_f1)
            else:
                # Minkowski distance to ideal point (origin)
                crowd_dis[in_front] = 1.0 / (torch.norm(f_norm[in_front], p=self.p, dim=1) + 1e-6)

        # Finalize State using Lexsort
        # Primary: front_no (asc), Secondary: crowd_dis (desc)
        indices = lexsort(torch.stack([-crowd_dis, front_no.float()]))[: self.pop_size]

        self.pop = pop[indices]
        self.fit = fit[indices]
        self.front_no = front_no[indices]
        self.crowd_dis = crowd_dis[indices]

    def _point_to_line_dist(self, points, direction_vecs):
        # points: (N, M), direction_vecs: (K, M)
        norm_w_sq = torch.sum(direction_vecs**2, dim=1)  # (K,)
        t = (points @ direction_vecs.T) / (norm_w_sq + 1e-6)  # (N, K)
        # projection: t.unsqueeze(-1) * direction_vecs.unsqueeze(0) -> (N, K, M)
        proj = t.unsqueeze(-1) * direction_vecs.unsqueeze(0)
        dist = torch.norm(points.unsqueeze(1) - proj, dim=2)  # (N, K)
        return dist

    def _calculate_front1_diversity(self, f_norm, p, extreme_indices):
        N_f = f_norm.shape[0]
        device = f_norm.device
        scores = torch.zeros(N_f, device=device)

        # Extreme points get infinite distance
        scores[extreme_indices] = torch.inf

        if N_f <= extreme_indices.numel():
            return scores

        dist_mat = self._minkowski_dist_matrix(f_norm, f_norm, p)
        # Normalize distances by point norms as per source
        nn = torch.norm(f_norm, p=p, dim=1, keepdim=True)
        dist_mat = dist_mat / (nn + 1e-6)

        selected = torch.zeros(N_f, dtype=torch.bool, device=device)
        selected[extreme_indices] = True

        # Greedy selection loop - fixed iterations for JIT
        # We need to select N_f - len(extreme) points
        remaining_indices = torch.arange(N_f, device=device)

        for _ in range(self.pop_size):  # Upper bound on iterations
            if selected.all():
                break

            mask_unselected = ~selected
            # Distances from unselected to selected
            dists_to_selected = dist_mat[mask_unselected][:, selected]

            # Survival score: sum of distances to 2 nearest neighbors
            # If only 1 selected, use that
            k = min(2, dists_to_selected.shape[1])
            vals, _ = torch.topk(dists_to_selected, k, dim=1, largest=False)
            current_scores = torch.sum(vals, dim=1)

            # Find best among unselected
            best_local_idx = torch.argmax(current_scores)
            best_global_idx = remaining_indices[mask_unselected][best_local_idx]

            scores[best_global_idx] = current_scores[best_local_idx]
            selected[best_global_idx] = True

        return scores

    def _minkowski_dist_matrix(self, A, B, p):
        diff = torch.abs(A.unsqueeze(1) - B.unsqueeze(0))
        return torch.pow(torch.sum(torch.pow(diff, p), dim=-1) + 1e-6, 1.0 / (p + 1e-6))


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = AGEMOEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
