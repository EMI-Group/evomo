import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp


class EFRRR(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, K: int = 2, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.K = K
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Weight Vectors
        w, _ = uniform_sampling(pop_size, n_objs)
        self.w = Mutable(w.to(device=device))

        # Ideal and Nadir points
        self.z = Mutable(torch.zeros((1, n_objs), device=device))
        self.znad = Mutable(torch.ones((1, n_objs), device=device))

        # Ranking for selection
        self.rank_no = Mutable(torch.zeros(pop_size, device=device, dtype=torch.int32))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0)[0].unsqueeze(0)
        self.znad = torch.max(self.fit, dim=0)[0].unsqueeze(0)

        norm_fit, new_znad = self._update_normalization(self.fit)
        self.znad = new_znad
        self.rank_no = self._maximum_ranking(norm_fit, self.w, self.K)

    def step(self) -> None:
        # 1. Mating / Variation
        # Selection: tournament_selection_multifit (lower rank is better)
        mating_pool = tournament_selection_multifit(self.pop_size, [self.rank_no.float()], tournament_size=2)

        # Variation
        crossovered = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(crossovered, self.lb, self.ub, pro_m=1.0 / self.lb.numel(), dis_m=20.0)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(offspring)

        # 3. Environmental Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        # Update Ideal Point
        self.z = torch.min(torch.cat([self.z, combined_fit], dim=0), dim=0)[0].unsqueeze(0)

        # Normalization and Nadir Update
        norm_fit, new_znad = self._update_normalization(combined_fit)
        self.znad = new_znad

        # Maximum Ranking Logic
        Rg = self._maximum_ranking(norm_fit, self.w, self.K)

        # Truncation
        # In MATLAB: RgFrontNo(LastFront(1:sum(RgFrontNo<=MaxFNo)-N)) = inf;
        # This is equivalent to sorting by Rg and taking top N.
        idx = torch.argsort(Rg, stable=True)[: self.pop_size]
        self.pop = combined_pop[idx]
        self.fit = combined_fit[idx]
        self.rank_no = Rg[idx]

    def _update_normalization(self, fit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, M = fit.shape
        device = fit.device

        # Step 1: Find extreme points using ASF
        # W = zeros(M) + 1e-6; W(logical(eye(M))) = 1;
        eye_w = torch.eye(M, device=device) + 1e-6
        f_shifted = fit - self.z

        # ASF calculation: max( (f-z) / (znad-z) ./ w )
        # Here we use the simplified version from the blueprint
        scale = self.znad - self.z + 1e-6
        asf_val = (f_shifted.unsqueeze(1) / scale.unsqueeze(0) / eye_w.unsqueeze(0)).max(dim=-1)[0]
        extreme_idx = torch.argmin(asf_val, dim=0)
        S = f_shifted[extreme_idx]  # (M, M)

        # Step 2: Solve S a = 1 for intercepts
        try:
            # Hyperplane = S \ ones(M,1)
            hyperplane = torch.linalg.solve(S, torch.ones((M, 1), device=device))
            a = (1.0 / (hyperplane.view(1, M) + 1e-6)) + self.z
        except RuntimeError:
            a = torch.max(fit, dim=0)[0].unsqueeze(0)

        # Fallback if intercepts are invalid (a <= z)
        invalid_mask = torch.any(a <= self.z)
        znad = torch.where(invalid_mask, torch.max(fit, dim=0)[0].unsqueeze(0), a)

        # Step 3: Normalize
        norm_fit = (fit - self.z) / (znad - self.z + 1e-6)
        return norm_fit, znad

    def _maximum_ranking(self, norm_fit: torch.Tensor, w: torch.Tensor, K: int) -> torch.Tensor:
        N_total = norm_fit.shape[0]
        N_w = w.shape[0]
        device = norm_fit.device
        sentinel = 1e9  # Use a large float instead of int sentinel

        # Cosine Similarity (Distance)
        norm_f = torch.norm(norm_fit, dim=1, keepdim=True) + 1e-6
        norm_w_vec = torch.norm(w, dim=1, keepdim=True) + 1e-6

        cos_sim = (norm_fit @ w.T) / (norm_f @ norm_w_vec.T)
        dist = 1.0 - cos_sim  # (N_total, N_w)

        # Restriction Mask: K closest weights for each solution
        _, indices = torch.topk(dist, k=K, dim=1, largest=False)
        mask = torch.zeros((N_total, N_w), dtype=torch.bool, device=device)
        mask.scatter_(1, indices, True)

        # Tchebycheff Matrix: g = max(norm_fit_i / w_j)
        # Broadcasting: (N_total, 1, M) / (1, N_w, M) -> (N_total, N_w, M)
        g = (norm_fit.unsqueeze(1) / (w.unsqueeze(0) + 1e-6)).max(dim=-1)[0]

        # Apply Restriction
        g_restricted = torch.where(mask, g, torch.tensor(sentinel, device=device))

        # Ranking: Rank of each solution for each weight vector
        # sort twice to get the rank (0 to N_total-1)
        ranks = torch.argsort(torch.argsort(g_restricted, dim=0), dim=0)

        # Rg = min(r.*g,[],2) where g is 1 if not inf
        # In our case, g_restricted is sentinel if not in K-nearest
        # We want the rank only for the allowed weights
        valid_ranks = torch.where(mask, ranks.float(), torch.tensor(float("inf"), device=device))
        Rg = torch.min(valid_ranks, dim=1)[0]

        # Get the front of each solution: [~,~,RgFrontNO] = unique(Rg)
        # return_inverse gives the indices to reconstruct the original tensor from unique values
        _, RgFrontNO = torch.unique(Rg, return_inverse=True)

        return RgFrontNO.to(torch.int32)


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = EFRRR(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
