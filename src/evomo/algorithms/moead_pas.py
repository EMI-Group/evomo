import torch
from evox.core import Algorithm, Mutable
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp

from evomo.operators.selection import non_dominate_rank


class MOEAD_PaS(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, T: int = 20, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.T = T
        D = lb.numel()

        # Sentinel for L-infinity norm (Bug #1)
        self.sentinel_inf = torch.iinfo(torch.int32).max

        # Initialize State (Mutables)
        # Ensure initial population is exactly pop_size
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Weight Vectors and Neighborhood
        W, n_actual = uniform_sampling(pop_size, n_objs)
        # Adjust pop_size to match uniform_sampling output
        self.pop_size = n_actual
        self.T = min(T, self.pop_size)
        self.W = Mutable(W.to(device))

        # Re-init pop and fit if size changed
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

        dist = torch.cdist(self.W, self.W)
        self.B = Mutable(torch.topk(dist, self.T, largest=False, dim=1).indices)

        # PaS Specifics
        self.p = Mutable(torch.ones((self.pop_size,), device=device))
        self.z = Mutable(torch.full((1, n_objs), torch.inf, device=device))
        self.znad = Mutable(torch.full((1, n_objs), -torch.inf, device=device))
        self.gen = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def _update_nadir(self, fit: torch.Tensor) -> torch.Tensor:
        rank = non_dominate_rank(fit)
        rank1_mask = rank == 1
        # Deadlock Breaker: If no rank 1 found, use all individuals to calculate nadir
        safe_mask = torch.where(rank1_mask.any(), rank1_mask, torch.ones_like(rank1_mask, dtype=torch.bool))
        return torch.max(fit[safe_mask], dim=0, keepdim=True).values

    def _calc_scalar_func(self, nObj: torch.Tensor, W: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # Bug #12: Safe division
        v = nObj / (W + 1e-6)

        # Handle L-infinity (Tchebycheff) and L-p norm
        g_inf = torch.max(v, dim=-1).values

        # p is expanded to match v's leading dims
        p_val = p.unsqueeze(-1)
        # Bug #12: Safe power and sum. Clamp v to avoid negative bases for pow.
        v_safe = torch.clamp(v, min=0.0) + 1e-6
        g_p = torch.pow(torch.sum(torch.pow(v_safe, p_val), dim=-1), 1.0 / (p + 1e-6))

        # Bug #41: Use torch.where instead of python if
        return torch.where(p >= self.sentinel_inf, g_inf, g_p)

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0, keepdim=True).values
        self.znad = self._update_nadir(self.fit)

    def step(self) -> None:
        self.gen = self.gen + 1
        device = self.lb.device
        N = self.pop_size
        T = self.T

        # 1. Mating / Parent Selection
        P_indices = torch.where(
            torch.rand(N, 1, device=device) < 0.9, self.B, torch.arange(N, device=device).unsqueeze(1).repeat(1, T)
        )

        # Pick 2 parents for each subproblem
        rand_idx1 = torch.randint(0, T, (N,), device=device)
        rand_idx2 = torch.randint(0, T, (N,), device=device)
        idx1 = P_indices[torch.arange(N), rand_idx1]
        idx2 = P_indices[torch.arange(N), rand_idx2]

        # Generate offspring using DE
        F = 0.5
        CR = 0.5
        # Ensure self.pop is treated as a tensor for the operation
        curr_pop = self.pop
        off_pop = curr_pop + F * (curr_pop[idx1] - curr_pop[idx2])

        # Binomial Crossover
        cross_mask = torch.rand(off_pop.shape, device=device) < CR
        off_pop = torch.where(cross_mask, off_pop, curr_pop)

        off_pop = clamp(off_pop, self.lb, self.ub)
        off_fit = self.evaluate(off_pop)

        # 2. Update Ideal and Nadir Points
        self.z = torch.min(self.z, torch.min(off_fit, dim=0, keepdim=True).values)
        self.znad = self._update_nadir(torch.cat([self.fit, off_fit], dim=0))

        # 3. Environmental Selection (Replacement)
        denom = self.znad - self.z + 1e-6
        nObj_off = (off_fit - self.z) / denom
        nObj_pop = (self.fit - self.z) / denom

        W_neighbors = self.W[self.B]
        p_neighbors = self.p[self.B]

        g_neighbors = self._calc_scalar_func(nObj_pop[self.B], W_neighbors, p_neighbors)
        g_off_on_neighbors = self._calc_scalar_func(nObj_off.unsqueeze(1).expand(-1, T, -1), W_neighbors, p_neighbors)

        better_mask = g_off_on_neighbors < g_neighbors

        # Bug #2: Limit replacement
        limit = (T + 9) // 10
        replace_counts = torch.cumsum(better_mask.to(torch.int32), dim=1)
        final_mask = better_mask & (replace_counts <= limit)

        # Scatter update - using a loop-free vectorized approach to update pop and fit
        # To handle duplicates in flat_indices (where multiple neighbors of different subproblems point to same index),
        # we use the fact that MOEA/D typically updates the population sequentially or via a mask.
        # Here we use index_put_ which is JIT-friendly.
        rows, cols = torch.where(final_mask)
        flat_indices = self.B[rows, cols]

        # We update the population and fitness tensors
        new_pop = self.pop.clone()
        new_fit = self.fit.clone()
        new_pop[flat_indices] = off_pop[rows]
        new_fit[flat_indices] = off_fit[rows]
        self.pop = new_pop
        self.fit = new_fit

        # 4. Norm Adaptation (Every 10 generations)
        if self.gen % 10 == 0:
            p_set = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, self.sentinel_inf], device=device, dtype=torch.float32)

            # Recalculate normalized objectives for current population
            nObj_curr = (self.fit - self.z) / (self.znad - self.z + 1e-6)

            # Vectorized calculation for all p in p_set
            # nObj_curr: (N, M), W: (N, M)
            # We want g for every combination of weight i and individual j for every p
            # g shape: (11, N, N) -> (p_idx, weight_idx, individual_idx)

            # To keep memory usage sane, we follow the MATLAB logic: for each weight i, find best individual j
            # nObj_all: (1, 1, N, M), W_all: (1, N, 1, M)
            nObj_all = nObj_curr.view(1, 1, N, self.n_objs)
            W_all = self.W.view(1, N, 1, self.n_objs)
            p_all = p_set.view(-1, 1, 1)

            v_all = nObj_all / (W_all + 1e-6)
            g_all_inf = torch.max(v_all, dim=-1).values

            v_safe_all = torch.clamp(v_all, min=0.0) + 1e-6
            g_all_p = torch.pow(torch.sum(torch.pow(v_safe_all, p_all.unsqueeze(-1)), dim=-1), 1.0 / (p_all + 1e-6))
            g_all = torch.where(p_all >= self.sentinel_inf, g_all_inf, g_all_p)  # (11, N, N)

            best_ind_indices = torch.argmin(g_all, dim=2)  # (11, N)

            # fit_best: (11, N, M)
            fit_best = nObj_curr[best_ind_indices]

            # Cosine similarity
            norm_fit = torch.norm(fit_best, dim=-1, keepdim=True) + 1e-6
            norm_W = self.W.norm(dim=-1, keepdim=True).unsqueeze(0) + 1e-6
            cos_sim = torch.sum((fit_best / norm_fit) * (self.W.unsqueeze(0) / norm_W), dim=-1)  # (11, N)

            # Manifold approximation metric from MATLAB: sqrt(1 - cos^2) * norm
            # This is equivalent to sine * distance
            dist_val = torch.sqrt(torch.clamp(1 - cos_sim**2, min=0.0, max=1.0)) * torch.norm(fit_best, dim=-1)

            # Update p to the one with minimum distance (best manifold approximation)
            best_p_idx = torch.argmin(dist_val, dim=0)  # (N,)
            self.p = p_set[best_p_idx]


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAD_PaS(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
