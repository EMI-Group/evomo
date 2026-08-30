import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class OSP_NSDE(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, t_max: int = 100, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.t_max = t_max

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Trajectory Tracking
        self.history_pop = Mutable(torch.zeros((pop_size, self.dim, t_max), device=device))
        self.history_fit = Mutable(torch.zeros((pop_size, n_objs, t_max), device=device))
        self.history_hv = Mutable(torch.zeros((t_max,), device=device))

        # Counters and Parameters
        self.t = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        self.t_init = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        self.p = Mutable(torch.tensor(10, dtype=torch.int32, device=device))  # Forecast horizon
        self.nadir = Mutable(torch.ones(n_objs, device=device) * 1.5)

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        # Initial trajectory entry
        self.history_pop[:, :, 0] = self.pop
        self.history_fit[:, :, 0] = self.fit

    def step(self) -> None:
        device = self.lb.device
        N, D = self.pop_size, self.dim

        # 1. Trajectory Alignment (Step 3.A)
        # Find nearest neighbors in previous generation to maintain trajectory consistency
        prev_pop = self.history_pop[:, :, self.t]
        dist_matrix = torch.cdist(self.pop, prev_pop)  # [N, N]
        indices = torch.argmin(dist_matrix, dim=1)  # [N]

        # Update history with aligned data
        self.t = self.t + 1
        # Use boolean masking or indexing to update history (JIT compliant)
        self.history_pop[:, :, : self.t] = self.history_pop[indices, :, : self.t]
        self.history_fit[:, :, : self.t] = self.history_fit[indices, :, : self.t]
        self.history_pop[:, :, self.t] = self.pop
        self.history_fit[:, :, self.t] = self.fit

        # 2. OSP Logic (ARX + GMM-VI)
        # Trigger condition: check if we have enough history (e.g., > 5 gens)
        # Using torch.where to handle logic without graph breaks
        do_osp = self.t > 5

        # 2.1 ARX Forecast
        phi_f = self._arx_forecast(self.history_fit[:, :, : self.t + 1], self.p)

        # 2.2 GMM-VI Sampling
        # Find individuals closest to forecasted fitness
        dist_to_phi = torch.cdist(self.fit, phi_f)
        best_idx = torch.argmin(dist_to_phi, dim=0)
        X_f = self.pop[best_idx]

        mu, sigma = self._gmm_vi_fit(X_f, K=3)
        # Sample offspring from GMM
        eps = torch.randn(N, D, device=device)
        L = torch.linalg.cholesky(sigma + torch.eye(D, device=device) * 1e-6)
        off_osp = mu + torch.einsum("nd,bd->nb", eps, L)

        # 2.3 Standard Variation (DE/Poly)
        # Fallback/Hybrid: Use DE if not OSP, or combine
        off_de = simulated_binary(self.pop, pro_c=1.0, dis_c=20.0)
        off_de = polynomial_mutation(off_de, self.lb, self.ub)

        offspring = torch.where(do_osp.unsqueeze(-1), off_osp, off_de)
        offspring = clamp(offspring, self.lb, self.ub)

        # 3. Evaluation
        off_fit = self.evaluate(offspring)

        # 4. Environmental Selection (Brutal Static Truncation - Bug #30)
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        rank = non_dominate_rank(combined_fit)

        # Calculate density for all
        dist_all = torch.cdist(combined_fit, combined_fit)
        dist_all.fill_diagonal_(float("inf"))
        min_dist = torch.min(dist_all, dim=1).values

        # Lexsort: Rank is primary (last), Distance is secondary
        # We want small rank and large distance
        idx = lexsort(torch.stack([-min_dist, rank.float()]))

        selected_idx = idx[:N]
        self.pop = combined_pop[selected_idx]
        self.fit = combined_fit[selected_idx]

    def _arx_forecast(self, history: torch.Tensor, p: int) -> torch.Tensor:
        # history: [N, M, T]
        N, M, T = history.shape
        lags = 3
        if T <= lags:
            return history[:, :, -1]

        # Construct Design Matrix [T-lags, lags*M]
        # For simplicity in pure tensor form, we use the last available window
        Y = history[:, :, -lags:].transpose(1, 2).reshape(N, -1)  # [N, lags*M]
        # Target is the current fit
        target = history[:, :, -1]  # [N, M]

        # Solve W via Least Squares: Y @ W = target
        # Add epsilon for stability
        W = torch.linalg.lstsq(Y, target).solution  # [lags*M, M]

        # Forecast
        phi_f = Y @ W
        return phi_f

    def _gmm_vi_fit(self, X: torch.Tensor, K: int) -> (torch.Tensor, torch.Tensor):
        # Simplified VI for GMM (Single Component for JIT stability)
        # X: [N, D]
        mu = torch.mean(X, dim=0)
        diff = X - mu
        sigma = (diff.t() @ diff) / (X.shape[0] + 1e-6)
        return mu, sigma


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    # OSP_NSDE must be replaced by your actual class name
    algo = OSP_NSDE(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
