import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class tDEA_CPBI(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize Weights and Theta (Blueprint 3.A)
        W, actual_n = uniform_sampling(pop_size, n_objs)
        self.pop_size = actual_n
        self.W = Mutable(W.to(device))

        is_boundary = (self.W > 1e-4).sum(dim=1) == 1
        theta = torch.full((self.pop_size,), 5.0, device=device)
        theta[is_boundary] = 1e6
        self.theta = Mutable(theta)

        # Initialize State
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        self.z = Mutable(torch.full((n_objs,), torch.inf, device=device))
        self.znad = Mutable(torch.full((n_objs,), -torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0)[0]
        self.znad = torch.max(self.fit, dim=0)[0]

    def _calculate_hyperplane_intercepts(self, combined_fit, z):
        # Blueprint 4: ASF-based extreme points
        f_shifted = combined_fit - z
        M = self.n_objs
        # Axis-aligned weights for ASF
        eye = torch.eye(M, device=combined_fit.device) + 1e-6

        # ASF(f, w) = max (f_i / w_i)
        # Shape: [N_combined, M, M] -> [N_combined, M]
        asf = torch.max(f_shifted.unsqueeze(1) / eye.unsqueeze(0), dim=2)[0]
        extreme_idx = torch.argmin(asf, dim=0)
        E = f_shifted[extreme_idx]

        # Solve Ex = 1
        try:
            if torch.abs(torch.linalg.det(E)) > 1e-6:
                intercepts = torch.linalg.solve(E, torch.ones(M, device=combined_fit.device))
                znad = z + 1.0 / (intercepts + 1e-6)
                # Validate intercepts
                invalid = (intercepts <= 1e-6).any() | torch.isnan(znad).any()
                znad = torch.where(invalid, torch.max(combined_fit, dim=0)[0], znad)
            else:
                znad = torch.max(combined_fit, dim=0)[0]
        except RuntimeError:
            znad = torch.max(combined_fit, dim=0)[0]

        return znad

    def step(self) -> None:
        device = self.pop.device
        N = self.W.shape[0]

        # 1. Mating
        mating_idx = tournament_selection_multifit(N, [self.fit.sum(dim=1)], tournament_size=2)
        off_pop = simulated_binary(self.pop[mating_idx], pro_c=1.0, dis_c=20.0)
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(off_pop)

        # 3. Selection
        combined_pop = torch.cat([self.pop, off_pop], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)
        N_total = combined_fit.shape[0]

        # Update Ideal and Nadir
        self.z = torch.min(torch.stack([self.z, torch.min(combined_fit, dim=0)[0]]), dim=0)[0]
        self.znad = self._calculate_hyperplane_intercepts(combined_fit, self.z)

        # Normalization
        f_norm = (combined_fit - self.z) / (self.znad - self.z + 1e-6)

        # PBI-based Clustering
        norm_f = torch.norm(f_norm, dim=1, keepdim=True)
        norm_W = torch.norm(self.W, dim=1, keepdim=True)
        cosine = (f_norm @ self.W.t()) / (norm_f @ norm_W.t() + 1e-6)
        cosine = clamp(cosine, -1.0, 1.0)

        d1 = norm_f * cosine
        d2 = norm_f * torch.sqrt(clamp(1 - cosine**2, 0.0, 1.0) + 1e-6)

        cluster_idx = torch.argmin(d2, dim=1)
        row_idx = torch.arange(N_total, device=device)
        d1_assigned = d1[row_idx, cluster_idx]
        d2_assigned = d2[row_idx, cluster_idx]

        # Selection Strategy
        g = d1_assigned + self.theta[cluster_idx] * d2_assigned
        pareto_rank = non_dominate_rank(combined_fit)

        # Calculate sub-rank within clusters (Vectorized)
        # Sort by cluster index first, then by pareto rank, then by g
        sort_idx = lexsort(torch.stack([g, pareto_rank.float(), cluster_idx.float()]))

        sorted_clusters = cluster_idx[sort_idx]
        # diff is 1 where a new cluster starts
        diff = torch.zeros(N_total, device=device, dtype=torch.int32)
        diff[0] = 1
        diff[1:] = (sorted_clusters[1:] != sorted_clusters[:-1]).int()

        # Use cumsum on diff to identify segments, then subtract the start of each segment
        # This creates a 0, 1, 2... counter that resets at each new cluster
        cluster_sub_rank = (
            torch.arange(N_total, device=device)
            - torch.where(diff == 1, torch.arange(N_total, device=device), torch.tensor(0, device=device)).cummax(dim=0)[0]
        )

        # Map sub_ranks back to original order
        original_sub_ranks = torch.zeros(N_total, device=device, dtype=torch.int32)
        original_sub_ranks[sort_idx] = cluster_sub_rank.int()

        # Final Selection: Primary: sub_rank, Secondary: pareto_rank, Tertiary: g
        final_sort_idx = lexsort(torch.stack([g, pareto_rank.float(), original_sub_ranks.float()]))

        survivor_idx = final_sort_idx[:N]
        self.pop = combined_pop[survivor_idx]
        self.fit = combined_fit[survivor_idx]


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = tDEA_CPBI(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
