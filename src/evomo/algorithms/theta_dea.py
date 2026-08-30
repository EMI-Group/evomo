import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, lexsort, randint

from evomo.operators.selection import non_dominate_rank


class ThetaDEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # 1. Initialize Reference Weights
        W, n_samples = uniform_sampling(pop_size, n_objs)
        self.W = Mutable(W.to(device))
        self.pop_size = n_samples

        # 2. Initialize Theta (Penalty Parameters)
        # Boundary weights: only one non-zero element (Bug #13)
        is_boundary = (self.W > 1e-4).sum(dim=1) == 1
        theta_vals = torch.where(is_boundary, torch.tensor(1e6, device=device), torch.tensor(5.0, device=device))
        self.theta = Mutable(theta_vals)

        # 3. Initialize State
        self.pop = Mutable(torch.rand(self.pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        self.z = Mutable(torch.full((n_objs,), torch.inf, device=device))
        self.znad = Mutable(torch.full((n_objs,), -torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0)[0]
        self.znad = torch.max(self.fit, dim=0)[0]

    def _get_hyperplane_intercepts(self, merged_fit, z):
        # merged_fit: [2N, M], z: [M]
        f_shifted = merged_fit - z

        # ASF calculation: max(f_i / w_i)
        # Use eye matrix for extreme point search as per MATLAB Normalization
        W_extreme = torch.eye(self.n_objs, device=merged_fit.device) + 1e-6

        # Scaling factor for ASF (Bug #32)
        scale = self.znad - self.z
        scale = torch.where(scale < 1e-6, torch.tensor(1e-6, device=scale.device), scale)

        # ASF: [2N, M]
        asf = torch.max(f_shifted.unsqueeze(1) / (scale.unsqueeze(0).unsqueeze(1) * W_extreme.unsqueeze(0)), dim=-1)[0]
        extreme_idx = torch.argmin(asf, dim=0)  # [M]

        # A: [M, M]
        A = f_shifted[extreme_idx]
        b = torch.ones((self.n_objs, 1), device=merged_fit.device)

        # Solve Aa = 1 using pseudo-inverse for stability and JIT compliance (Bug #41)
        # torch.linalg.solve is risky if A is singular
        A_inv = torch.linalg.pinv(A)
        hyperplane = torch.matmul(A_inv, b).squeeze(-1)  # [M]

        # Intercepts a = 1 / hyperplane
        a = 1.0 / (hyperplane + 1e-6) + z

        # Fallback logic (Bug #5)
        invalid = (hyperplane <= 1e-6).any() | torch.isnan(a).any() | (a <= z).any()
        znad = torch.where(invalid, torch.max(merged_fit, dim=0)[0], a)

        return znad

    def _theta_rank(self, norm_fit):
        # norm_fit: [S_t, M], W: [N, M]
        norm_W = self.W / (torch.norm(self.W, dim=1, keepdim=True) + 1e-6)

        # Projection d1: [S_t, N]
        d1 = torch.matmul(norm_fit, norm_W.t())

        # Perpendicular distance d2: [S_t, N]
        # norm_fit_sq: [S_t], d1_sq: [S_t, N]
        norm_fit_sq = torch.sum(norm_fit**2, dim=1, keepdim=True)
        d2_sq = norm_fit_sq - d1**2
        d2 = torch.sqrt(torch.clamp(d2_sq, min=0.0))

        # Clustering
        cluster_id = torch.argmin(d2, dim=1)
        row_idx = torch.arange(norm_fit.shape[0], device=norm_fit.device)

        # PBI values for assigned clusters
        d1_assigned = d1[row_idx, cluster_id]
        d2_assigned = d2[row_idx, cluster_id]
        pbi = d1_assigned + self.theta[cluster_id] * d2_assigned

        # Vectorized Grouped Sort (tFrontNo)
        # Sort by cluster_id (primary) then by PBI (secondary)
        # lexsort: primary key goes last
        sort_keys = torch.stack([pbi, cluster_id.float()])
        sort_idx = lexsort(sort_keys)

        # Calculate ranks within each cluster
        sorted_clusters = cluster_id[sort_idx]
        # Identify where a new cluster starts
        is_diff = torch.cat([torch.tensor([1], device=norm_fit.device), (sorted_clusters[1:] != sorted_clusters[:-1]).int()])
        # Cumulative sum of is_diff gives a unique ID for each group,
        # but we need the rank within the group.
        # Use a trick: intra_rank = global_rank - rank_at_start_of_group
        global_rank = torch.arange(len(sort_idx), device=norm_fit.device)

        # To find rank_at_start_of_group without loops:
        group_start_indices = torch.where(is_diff == 1, global_rank, torch.tensor(0, device=norm_fit.device))
        # Forward fill the group_start_indices
        # In PyTorch, we can use cummax for forward fill on a non-decreasing sequence
        group_offsets = torch.cummax(group_start_indices, dim=0)[0]

        intra_rank = global_rank - group_offsets + 1

        tFrontNo = torch.zeros(norm_fit.shape[0], device=norm_fit.device)
        tFrontNo[sort_idx] = intra_rank.float()

        return tFrontNo

    def step(self) -> None:
        # 1. Mating
        mating_pool = randint(0, self.pop_size, (self.pop_size,), device=self.pop.device)
        offspring = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(offspring)
        merged_pop = torch.cat([self.pop, offspring], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)

        # 3. Normalization
        self.z = torch.min(torch.cat([self.z.unsqueeze(0), merged_fit], dim=0), dim=0)[0]
        self.znad = self._get_hyperplane_intercepts(merged_fit, self.z)
        norm_fit = (merged_fit - self.z) / (self.znad - self.z + 1e-6)

        # 4. Non-dominated Sort
        front_no = non_dominate_rank(merged_fit)

        # 5. Theta-Dominance Ranking
        t_front_no = self._theta_rank(norm_fit)

        # 6. Selection
        # Primary key: t_front_no, Secondary key: front_no
        # Bug #25: Primary key (t_front_no) goes last in lexsort
        keys = torch.stack([front_no.float(), t_front_no])
        idx = lexsort(keys)

        selected_idx = idx[: self.pop_size]
        self.pop = merged_pop[selected_idx]
        self.fit = merged_fit[selected_idx]


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = ThetaDEA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
