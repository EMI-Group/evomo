import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class TELSO(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, max_fe: int = 10000, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.maxFE = Parameter(max_fe)
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.mask = Mutable(torch.zeros(pop_size, D, device=device))
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.vel = Mutable(torch.zeros(pop_size, D, device=device))

        # Reference Vectors (Bug #13)
        v, _ = uniform_sampling(pop_size, n_objs)
        self.v = Mutable(v.to(device))

        # Gamma calculation (Min angle between reference vectors)
        cosine = self.v @ self.v.t()
        cosine = torch.clamp(cosine, -1, 1)
        acos_val = torch.acos(cosine)
        # Set diagonal to inf to find min off-diagonal
        acos_val.fill_diagonal_(torch.inf)
        self.gamma = Mutable(torch.min(acos_val, dim=1)[0].min())

        self.fe = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def _cal_fitness(self, fit: torch.Tensor) -> torch.Tensor:
        # Bug #29: Vectorized Shift-based density
        N = fit.shape[0]
        # Pairwise max objectives: (N, N, M)
        f_max = torch.max(fit.unsqueeze(1), fit.unsqueeze(0))
        # Distance matrix: (N, N)
        dist = torch.norm(fit.unsqueeze(1) - f_max, dim=-1)
        # Shift-based density fitness
        fitness = torch.sum(torch.exp(-dist / (0.05 * N + 1e-6)), dim=1)
        return fitness

    def _mask_learn(self, m1: torch.Tensor, m2: torch.Tensor, m_curr: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        # Bug #20: Sparse Mask Operation
        N, D = m_curr.shape
        identical_mask = (m1 == m2).float()

        # Calculate how many bits to keep from identical parts
        num_to_keep = (identical_mask.sum(dim=1) * ratio).floor().long()

        # Use topk on random values masked by identical_mask to pick bits
        rand_val = torch.rand(N, D, device=m_curr.device)
        # Only consider positions where m1 == m2
        scores = rand_val * identical_mask

        # Create a mask for selected bits using topk
        # We use a loop-free approach with scatter
        _, indices = torch.topk(scores, D, dim=1)
        # Create range tensor for broadcasting
        range_tensor = torch.arange(D, device=m_curr.device).expand(N, D)
        selected_bits = range_tensor < num_to_keep.unsqueeze(1)

        # Final mask construction
        off_mask = torch.where(selected_bits.gather(1, torch.argsort(indices, dim=1)), m1, m_curr)
        return off_mask

    def init_step(self) -> None:
        device = self.pop.device
        D = self.lb.numel()

        # 1. DF Calculation (Decision Variable Importance)
        identity = torch.eye(D, device=device)
        # Evaluate Identity as masks with all-ones pop
        ones_pop = torch.ones(D, D, device=device)
        df_fit = self.evaluate(ones_pop * identity)
        self.fe = self.fe + D

        # Pareto Ranks for DF
        ranks = non_dominate_rank(df_fit)
        df_scores = ranks.float()  # Lower rank is better

        # 2. Mask Init
        # For each individual, pick K bits based on DF + noise
        K = torch.randint(1, D + 1, (self.pop_size,), device=device)
        noise = torch.rand(self.pop_size, D, device=device)
        # We want to pick indices with LOW rank (high importance)
        # So we use topk on negative rank + noise
        init_scores = -df_scores.expand(self.pop_size, D) + noise
        _, top_indices = torch.topk(init_scores, D, dim=1)

        range_tensor = torch.arange(D, device=device).expand(self.pop_size, D)
        mask_bits = range_tensor < K.unsqueeze(1)

        new_mask = torch.zeros(self.pop_size, D, device=device)
        new_mask.scatter_(1, top_indices, mask_bits.float())

        self.mask = new_mask
        self.fit = self.evaluate(self.pop * self.mask)
        self.fe = self.fe + self.pop_size

    def step(self) -> None:
        device = self.pop.device
        N, D = self.pop.shape
        M = self.n_objs

        # 1. Learning Swarm Operator (Variation)
        # Informant Selection
        idx_r1 = torch.randint(0, N, (N,), device=device)
        idx_r2 = torch.randint(0, N, (N,), device=device)

        v_new = (
            0.5 * torch.rand(N, D, device=device) * self.vel
            + torch.rand(N, D, device=device) * (self.pop[idx_r1] - self.pop)
            + torch.rand(N, D, device=device) * (self.pop[idx_r2] - self.pop)
        )

        off_pop = self.pop + v_new
        off_pop = clamp(off_pop, self.lb, self.ub)

        # Mask Evolution
        ratio = self.fe.float() / self.maxFE
        off_mask = self._mask_learn(self.mask[idx_r1], self.mask[idx_r2], self.mask, ratio)

        # Evaluation
        off_fit = self.evaluate(off_pop * off_mask)
        self.fe = self.fe + N

        # 2. Environmental Selection (APD-based)
        merged_pop = torch.cat([self.pop, off_pop], dim=0)
        merged_mask = torch.cat([self.mask, off_mask], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)
        merged_vel = torch.cat([self.vel, v_new], dim=0)

        # Translation
        f_min = torch.min(merged_fit, dim=0)[0]
        obj = merged_fit - f_min

        # APD Calculation
        norm_obj = torch.norm(obj, dim=1, keepdim=True)
        norm_v = torch.norm(self.v, dim=1, keepdim=True)
        cosine = (obj @ self.v.t()) / (norm_obj @ norm_v.t() + 1e-6)
        angles = torch.acos(torch.clamp(cosine, -1, 1))

        angles_min, assoc_idx = torch.min(angles, dim=1)

        theta = (self.fe.float() / self.maxFE) ** 2
        # APD: (2N,)
        apd = (1 + M * theta * (angles_min / (self.gamma + 1e-6))) * norm_obj.squeeze()

        # Final Selection: For each reference vector, find min APD
        # Vectorized selection of best individual per reference vector
        # Sort by association index (primary) and APD (secondary)
        sort_idx = lexsort(torch.stack([apd, assoc_idx.float()]))
        sorted_assoc = assoc_idx[sort_idx]

        # Find first occurrence of each association index in sorted list
        diff = torch.cat([torch.tensor([1], device=device), sorted_assoc[1:] - sorted_assoc[:-1]])
        first_indices = torch.where(diff > 0)[0]

        # Map back to original merged indices
        final_idx = sort_idx[first_indices]

        # If we have fewer than pop_size unique associations, fill with others
        if final_idx.shape[0] < self.pop_size:
            # This is a fallback for JIT compliance - use a mask to find remaining
            mask_rem = torch.ones(2 * N, dtype=torch.bool, device=device)
            mask_rem[final_idx] = False
            rem_idx = torch.where(mask_rem)[0][: self.pop_size - final_idx.shape[0]]
            final_idx = torch.cat([final_idx, rem_idx])

        self.pop = merged_pop[final_idx]
        self.mask = merged_mask[final_idx]
        self.fit = merged_fit[final_idx]
        self.vel = merged_vel[final_idx]


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = TELSO(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
