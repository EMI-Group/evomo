import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import nd_environmental_selection, non_dominate_rank
from evomo.utils import unique_rows_sorted


class DMMOEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.P = 5  # History size
        self.H = 10  # Hidden size for MLP

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.mask = Mutable(torch.rand(pop_size, self.dim, device=device) < 0.5)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.var_fit = Mutable(torch.zeros(self.dim, device=device))

        # History Buffers
        self.dec_history = Mutable(torch.zeros(self.P + 1, pop_size, self.dim, device=device))
        self.mask_history = Mutable(torch.zeros(self.P + 1, pop_size, self.dim, device=device))
        self.iter_count = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        self.dis = Mutable(torch.zeros(pop_size, device=device))
        self.rank = Mutable(torch.full((pop_size,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        device = self.lb.device
        # A. Initialization (Variable Fitness Estimation)
        test_mask = torch.eye(self.dim, device=device).bool()
        test_dec = torch.full((self.dim, self.dim), 0.5, device=device) * (self.ub - self.lb) + self.lb

        # Evaluate D solutions where each solution i has only variable i active
        test_fit = self.evaluate(test_dec * test_mask)
        ranks = non_dominate_rank(test_fit)
        self.var_fit = ranks.float()

        # Initial Population Evaluation
        self.fit = self.evaluate(self.pop * self.mask)

        # Initial Selection to set rank/dis
        _, _, self.rank, self.dis, _ = nd_environmental_selection(self.pop, self.fit, self.pop_size)

    def _batched_mlp_predict(self, history: torch.Tensor) -> torch.Tensor:
        # history: (P, N, D) -> (N*D, P)
        P, N, D = history.shape
        x = history.permute(1, 2, 0).reshape(-1, P)

        # Simple deterministic weights for JIT-friendly "prediction"
        W1 = torch.ones(P, self.H, device=history.device) / P
        W2 = torch.ones(self.H, 1, device=history.device) / self.H

        out = torch.relu(x @ W1) @ W2
        return out.view(N, D)

    def _batched_krr_predict(self, history: torch.Tensor) -> torch.Tensor:
        # history: (P, N, D) -> (N*D, P)
        P, N, D = history.shape
        X = torch.arange(P, device=history.device).float().view(-1, 1)
        Y = history.reshape(P, -1)  # (P, N*D)

        # K = exp(-gamma * dist^2)
        K = torch.exp(-0.1 * torch.cdist(X, X) ** 2)
        # alpha = (K + lambda I)^-1 Y
        identity = torch.eye(P, device=history.device)
        alpha = torch.linalg.solve(K + 1e-4 * identity, Y)

        # Predict next step (P+1)
        K_test = torch.exp(-0.1 * torch.cdist(torch.tensor([[float(P)]], device=history.device), X) ** 2)
        pred = (K_test @ alpha).view(N, D)
        return pred > 0.5

    def step(self) -> None:
        device = self.lb.device
        self.iter_count += 1

        # 1. Dynamic Prediction (Triggered every 20 gens as proxy for "Change")
        # Update history buffers
        new_dec_hist = torch.cat([self.dec_history[1:], self.pop.unsqueeze(0)], dim=0)
        new_mask_hist = torch.cat([self.mask_history[1:], self.mask.float().unsqueeze(0)], dim=0)
        self.dec_history = new_dec_hist
        self.mask_history = new_mask_hist

        # Apply Dual-Model Prediction if buffer is full
        is_change = self.iter_count % 20 == 0
        pred_dec = torch.where(is_change, self._batched_mlp_predict(self.dec_history[: self.P]), self.pop)
        pred_mask = torch.where(is_change, self._batched_krr_predict(self.mask_history[: self.P]), self.mask)

        # 2. Mating / Variation
        # Tournament Selection based on Rank and Crowding Distance
        mating_pool = tournament_selection_multifit(self.pop_size, [self.rank.float(), -self.dis], tournament_size=2)

        # Real-valued variation (SBX)
        off_dec = simulated_binary(pred_dec[mating_pool], pro_c=1.0, dis_c=20.0)
        off_dec = clamp(off_dec, self.lb, self.ub)

        # Mask variation
        p1_idx = mating_pool[: self.pop_size // 2]
        p2_idx = mating_pool[self.pop_size // 2 :]
        mask_p1 = pred_mask[p1_idx]
        mask_p2 = pred_mask[p2_idx]

        # Crossover masks
        off_mask = torch.where(torch.rand(self.pop_size // 2, self.dim, device=device) < 0.5, mask_p1, mask_p2)
        off_mask = torch.cat([off_mask, off_mask], dim=0)  # Match pop_size

        # Mutation masks (Bug #20 Compliance)
        mut_prob = 1.0 / self.dim
        mut_mask = torch.rand(self.pop_size, self.dim, device=device) < mut_prob

        # Apply mutation: flip bits where mut_mask is true
        off_mask = torch.where(mut_mask, ~off_mask, off_mask)

        # 3. Evaluation
        off_fit = self.evaluate(off_dec * off_mask)

        # 4. Environmental Selection (Bug #9 & #21 Compliance)
        merged_pop = torch.cat([self.pop, off_dec], dim=0)
        merged_mask = torch.cat([self.mask, off_mask], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)

        # Unique Filtering (Bug #3)
        u_pop_combined = torch.cat([merged_pop, merged_mask.float()], dim=1)
        _, u_idx = unique_rows_sorted(u_pop_combined)
        merged_pop, merged_mask, merged_fit = merged_pop[u_idx], merged_mask[u_idx], merged_fit[u_idx]

        # Non-dominated sorting
        ranks = non_dominate_rank(merged_fit)

        # Front Peeling Loop (Bug #9)
        num_selected = 0
        next_indices = torch.zeros(self.pop_size, dtype=torch.long, device=device)
        current_front = 0

        total_n = merged_fit.shape[0]
        remaining_mask = torch.ones(total_n, dtype=torch.bool, device=device)

        # Pre-allocate results
        final_ranks = torch.zeros(self.pop_size, dtype=torch.int32, device=device)
        final_dis = torch.zeros(self.pop_size, device=device)

        while num_selected < self.pop_size:
            mask = (ranks == current_front) & remaining_mask
            num_in_front = mask.sum()

            # Deadlock Breaker
            mask = torch.where(num_in_front == 0, remaining_mask, mask)
            num_in_front = mask.sum()

            # Crowding Distance (Bug #6, #21)
            dist = crowding_distance(merged_fit, mask)

            # Selection within front
            if num_selected + num_in_front <= self.pop_size:
                # Take whole front
                idx = torch.where(mask)[0]
                next_indices[num_selected : num_selected + num_in_front] = idx
                final_ranks[num_selected : num_selected + num_in_front] = current_front
                final_dis[num_selected : num_selected + num_in_front] = dist[mask]
                num_selected += num_in_front
            else:
                # Partial front selection using lexsort (Bug #25)
                needed = self.pop_size - num_selected
                front_indices = torch.where(mask)[0]
                front_dist = dist[mask]
                # Primary key (Rank) is last, but here rank is constant, so sort by -dist
                sort_idx = lexsort(torch.stack([-front_dist]))
                sel_idx = front_indices[sort_idx[:needed]]
                next_indices[num_selected:] = sel_idx
                final_ranks[num_selected:] = current_front
                final_dis[num_selected:] = front_dist[sort_idx[:needed]]
                num_selected = self.pop_size

            remaining_mask = remaining_mask & (~mask)
            current_front += 1

        self.pop = merged_pop[next_indices]
        self.mask = merged_mask[next_indices]
        self.fit = merged_fit[next_indices]
        self.rank = final_ranks
        self.dis = final_dis


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = DMMOEA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
