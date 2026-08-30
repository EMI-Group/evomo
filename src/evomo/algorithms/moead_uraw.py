import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp

from evomo.utils import unique_rows_sorted


class MOEAURAW(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        delta: float = 0.9,
        nr: int = 2,
        nEP: int = 200,
        **kwargs,
    ):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.delta = delta
        self.nr = nr
        self.nEP = nEP
        D = lb.numel()
        T = (pop_size + 9) // 10  # Bug #2 Compliance

        # 1. Initialization (Uniformly Randomly Adaptive Weights)
        P = torch.abs(torch.randn(5000, n_objs, device=device))
        P = P / (P.sum(dim=1, keepdim=True) + 1e-6)

        W_set = torch.eye(n_objs, device=device)
        # Greedy spreading for initial weights
        for _ in range(pop_size - n_objs):
            dist_mat = torch.cdist(P, W_set)
            min_dist = dist_mat.min(dim=1).values
            idx = min_dist.argmax()
            W_set = torch.cat([W_set, P[idx : idx + 1]], dim=0)

        # Transformation: W = 1/W / sum(1/W)
        inv_W = 1.0 / (W_set + 1e-6)
        w = inv_W / (inv_W.sum(dim=1, keepdim=True) + 1e-6)

        dist_w = torch.cdist(w, w)
        b = torch.topk(dist_w, k=T, largest=False).indices.to(torch.int32)

        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), 1e18, device=device))
        self.w = Mutable(w)
        self.z = Mutable(torch.full((1, n_objs), 1e18, device=device))
        self.b = Mutable(b)
        self.archive_pop = Mutable(torch.empty((0, D), device=device))
        self.archive_fit = Mutable(torch.empty((0, n_objs), device=device))
        self.gen = Mutable(torch.zeros(1, dtype=torch.int32, device=device))

    def _calc_product_dist_score(self, fit: torch.Tensor, M: int) -> torch.Tensor:
        # Bug #26 & Helper Contract Compliance
        if fit.shape[0] == 0:
            return torch.empty(0, device=fit.device)
        D = torch.cdist(fit, fit)
        # Use a large sentinel for diagonal to ignore self-distance
        sentinel = 1e18
        D = D + torch.eye(D.shape[0], device=D.device) * sentinel
        sorted_D, _ = torch.sort(D, dim=1)
        # Product of M nearest neighbors
        k = min(M, sorted_D.shape[1])
        return torch.prod(sorted_D[:, :k], dim=1)

    def _update_ep(self, off_pop: torch.Tensor, off_fit: torch.Tensor):
        combined_pop = torch.cat([self.archive_pop, off_pop], dim=0)
        combined_fit = torch.cat([self.archive_fit, off_fit], dim=0)

        u_pop, u_idx = unique_rows_sorted(combined_pop)
        u_fit = combined_fit[u_idx]

        # Bug #24: Pareto Dominance
        # (X <= Y).all(dim=-1) & (X < Y).any(dim=-1)
        # Check if any individual in u_fit dominates others
        # Shape: (N, N, M)
        diff_le = (u_fit.unsqueeze(1) <= u_fit.unsqueeze(0)).all(dim=-1)
        diff_lt = (u_fit.unsqueeze(1) < u_fit.unsqueeze(0)).any(dim=-1)
        dom_matrix = diff_le & diff_lt
        # If any individual j dominates i, i is not non-dominated
        is_dominated = dom_matrix.any(dim=0)

        ep_pop = u_pop[~is_dominated]
        ep_fit = u_fit[~is_dominated]

        curr_size = ep_pop.shape[0]
        if curr_size > self.nEP:
            score = self._calc_product_dist_score(ep_fit, self.n_objs)
            _, keep_idx = torch.topk(score, k=self.nEP, largest=True)
            ep_pop = ep_pop[keep_idx]
            ep_fit = ep_fit[keep_idx]

        self.archive_pop = ep_pop
        self.archive_fit = ep_fit

    def _update_weight(self):
        nus = self.pop_size // 20  # 0.05 * N
        if nus == 0:
            nus = 1

        if self.archive_fit.shape[0] == 0:
            return

        # 1. Delete overcrowded subproblems
        score = self._calc_product_dist_score(self.fit, self.n_objs)
        num_to_del = min(nus, self.archive_fit.shape[0])
        _, del_idx = torch.topk(score, k=num_to_del, largest=False)

        # 2. Add new subproblems from EP greedily
        C_pop = self.archive_pop
        C_fit = self.archive_fit

        keep_mask = torch.ones(self.pop_size, dtype=torch.bool, device=self.lb.device)
        keep_mask[del_idx] = False
        current_fit = self.fit[keep_mask]

        selected_c_indices = torch.full((num_to_del,), -1, dtype=torch.long, device=self.lb.device)
        temp_fit = current_fit

        # Greedy selection loop
        for i in range(num_to_del):
            dist_to_pop = torch.cdist(C_fit, temp_fit)
            min_dist_to_pop = dist_to_pop.min(dim=1).values

            # Mask already selected candidates
            if i > 0:
                mask_indices = selected_c_indices[:i]
                min_dist_to_pop[mask_indices] = -1.0

            best_cand_idx = min_dist_to_pop.argmax()
            selected_c_indices[i] = best_cand_idx
            temp_fit = torch.cat([temp_fit, C_fit[best_cand_idx : best_cand_idx + 1]], dim=0)

        # 3. Update weights and population
        added_pop = C_pop[selected_c_indices]
        added_fit = C_fit[selected_c_indices]

        diff = torch.abs(added_fit - self.z)
        diff = torch.where(diff == 0, torch.tensor(0.999999, device=diff.device), diff)
        new_w = (1.0 / diff) / (1.0 / (diff + 1e-6)).sum(dim=1, keepdim=True)

        self.pop[del_idx] = added_pop
        self.fit[del_idx] = added_fit
        self.w[del_idx] = new_w

        # Recompute neighborhoods
        dist_w = torch.cdist(self.w, self.w)
        T = (self.pop_size + 9) // 10
        self.b = torch.topk(dist_w, k=T, largest=False).indices.to(torch.int32)

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0, keepdim=True).values
        self._update_ep(self.pop, self.fit)

    def step(self) -> None:
        self.gen += 1
        N = self.pop_size
        T = self.b.shape[1]
        device = self.lb.device

        # Parent Selection
        r1 = torch.rand(N, device=device)
        mask = r1 < self.delta

        # Local vs Global selection
        local_idx = self.b[torch.arange(N, device=device), torch.randint(T, (N,), device=device)].long()
        global_idx = torch.randint(N, (N,), device=device)
        P1_idx = torch.where(mask, local_idx, global_idx)

        local_idx2 = self.b[torch.arange(N, device=device), torch.randint(T, (N,), device=device)].long()
        global_idx2 = torch.randint(N, (N,), device=device)
        P2_idx = torch.where(mask, local_idx2, global_idx2)

        # Variation
        offspring = simulated_binary(torch.cat([self.pop[P1_idx], self.pop[P2_idx]], dim=0))
        offspring = offspring[:N]
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        off_fit = self.evaluate(offspring)
        self.z = torch.min(self.z, off_fit.min(dim=0, keepdim=True).values)

        # Tchebycheff Update (Iterate over subproblems to apply 'nr' limit)
        for i in range(N):
            P = self.b[i].long()
            # g(x|w) = max w_j * |f_j(x) - z_j|
            g_old = torch.max(torch.abs(self.fit[P] - self.z) * self.w[P], dim=1).values
            g_new = torch.max(torch.abs(off_fit[i] - self.z) * self.w[P], dim=1).values

            better_mask = g_new <= g_old
            if better_mask.any():
                better_indices = P[better_mask]
                # Limit replacement to nr
                update_idx = better_indices[: self.nr]
                self.pop[update_idx] = offspring[i]
                self.fit[update_idx] = off_fit[i]

        self._update_ep(offspring, off_fit)

        # Adaptation moment: 0.05 of max generations
        # In Evox workflow, we don't have maxFE directly, so we use a fixed interval
        if self.gen % 20 == 0:
            self._update_weight()


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAURAW(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
