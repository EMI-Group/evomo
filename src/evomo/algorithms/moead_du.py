import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp

from evomo.utils import unique_rows_sorted


class MOEADDU(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        T: int = 20,
        delta: float = 0.9,
        K: int = 5,
        **kwargs,
    ):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.T = T
        self.delta = delta
        self.K = K
        D = lb.numel()

        # 1. Weight Generation
        W, n_samples = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_samples
        self.T = min(T, self.pop_size)
        self.W = Mutable(W.to(device))

        # 2. Neighborhood (self.B)
        dist = torch.cdist(self.W, self.W, p=2)
        self.B = Mutable(torch.topk(dist, k=self.T, largest=False).indices.to(torch.int32))

        # 3. Initialize State
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        self.z = Mutable(torch.zeros((1, n_objs), device=device))
        self.znad = Mutable(torch.ones((1, n_objs), device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0, keepdim=True).values
        self.znad = torch.max(self.fit, dim=0, keepdim=True).values

    def _update_nadir_point(self, curr_fit, z_ideal) -> torch.Tensor:
        # ASF-based extreme point identification
        f_norm = (curr_fit - z_ideal) / (torch.max(curr_fit, dim=0).values - z_ideal + 1e-6)
        # asf = max(f_norm / weight_axes)
        eye = torch.eye(self.n_objs, device=curr_fit.device) + 1e-6
        asf = torch.max(f_norm.unsqueeze(1) / eye, dim=2).values
        extreme_idx = torch.argmin(asf, dim=0)

        A = curr_fit[extreme_idx] - z_ideal
        try:
            # Solve Ax = 1
            b = torch.ones((self.n_objs, 1), device=curr_fit.device)
            x = torch.linalg.solve(A, b)
            intercepts = 1 / (x.view(-1) + 1e-6)

            # Fallback if intercepts are invalid
            invalid = torch.any(intercepts <= 1e-6)
            znad = torch.where(invalid, torch.max(curr_fit, dim=0).values, z_ideal.view(-1) + intercepts)
        except RuntimeError:
            znad = torch.max(curr_fit, dim=0).values

        return znad.view(1, -1)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size

        # 1. Mating / Parent Selection
        rdm_mask = torch.rand(N, device=device) < self.delta

        # P1 selection
        neigh_idx1 = torch.gather(self.B, 1, torch.randint(0, self.T, (N, 1), device=device).to(torch.int32)).view(-1)
        global_idx1 = torch.randint(0, N, (N,), device=device)
        P1 = torch.where(rdm_mask, neigh_idx1, global_idx1)

        # P2 selection
        neigh_idx2 = torch.gather(self.B, 1, torch.randint(0, self.T, (N, 1), device=device).to(torch.int32)).view(-1)
        global_idx2 = torch.randint(0, N, (N,), device=device)
        P2 = torch.where(rdm_mask, neigh_idx2, global_idx2)

        # 2. Variation
        offspring = simulated_binary(torch.cat([self.pop[P1], self.pop[P2]], dim=0))
        offspring = offspring[:N]  # Take half as per standard MOEA/D batch
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        off_fit = self.evaluate(offspring)

        # 3. Update Ideal and Nadir Points
        self.z = torch.min(torch.cat([self.z, off_fit], dim=0), dim=0, keepdim=True).values
        self.znad = self._update_nadir_point(torch.cat([self.fit, off_fit], dim=0), self.z)

        # 4. Distance-based Updating (DU)
        # Cosine Similarity
        off_n = off_fit / (torch.norm(off_fit, dim=1, keepdim=True) + 1e-6)
        W_n = self.W / (torch.norm(self.W, dim=1, keepdim=True) + 1e-6)
        cos_sim = off_n @ W_n.T  # [N, N]

        # Top-K Subproblems for each offspring
        P_indices = torch.topk(cos_sim, k=self.K, dim=1, largest=True).indices  # [N, K]

        # Modified Tchebycheff
        scale = self.znad - self.z + 1e-6

        # g_old: [N, K] - Tchebycheff of current population at P_indices
        # We need to gather fit and W based on P_indices
        flat_P = P_indices.view(-1)
        fit_P = self.fit[flat_P].view(N, self.K, self.n_objs)
        W_P = self.W[flat_P].view(N, self.K, self.n_objs)

        g_old = torch.max(torch.abs(fit_P - self.z) / scale / (W_P + 1e-6), dim=2).values

        # g_new: [N, K] - Tchebycheff of offspring at P_indices
        g_new = torch.max(torch.abs(off_fit.unsqueeze(1) - self.z) / scale / (W_P + 1e-6), dim=2).values

        better_mask = g_new < g_old  # [N, K]

        # Sequential Update Emulation:
        # For each subproblem (column in cos_sim), find the best offspring that improves it.
        # We transpose the problem: for each weight vector j, which offspring i is best?
        # To stay JIT-friendly and vectorized:
        inf_val = 1e18

        # Scatter g_new into the positions defined by P_indices where better_mask is true
        rows = torch.arange(N, device=device).unsqueeze(1).expand(N, self.K)
        valid_rows = rows[better_mask]
        valid_cols = P_indices[better_mask]
        valid_vals = g_new[better_mask]

        # We want the offspring that results in the minimum g_new for each subproblem
        # If multiple offspring want to update the same subproblem, we pick the one with min g_new
        # Sort by subproblem index (valid_cols) then by g_new value
        if valid_cols.numel() > 0:
            sort_idx = torch.argsort(valid_cols * inf_val + valid_vals)
            u_cols, first_idx = unique_rows_sorted(valid_cols[sort_idx].unsqueeze(1))

            best_offspring_idx = valid_rows[sort_idx][first_idx]
            target_subproblems = u_cols.view(-1)

            # Update population
            self.pop[target_subproblems] = offspring[best_offspring_idx]
            self.fit[target_subproblems] = off_fit[best_offspring_idx]


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADDU(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
