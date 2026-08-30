import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp


class MOEADAWA(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        T: int = None,
        nr: int = None,
        nEP: int = None,
        **kwargs,
    ):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # 1. Weight Init & Transformation
        W, actual_n = uniform_sampling(pop_size, n_objs)
        self.pop_size = int(actual_n)
        W = W.to(device)

        W_inv = 1.0 / (W + 1e-6)
        self.W = Mutable(W_inv / (W_inv.sum(dim=1, keepdim=True) + 1e-6))

        self.T = T if T is not None else (self.pop_size + 9) // 10
        self.nr = nr if nr is not None else (self.pop_size + 99) // 100
        self.nEP = nEP if nEP is not None else (self.pop_size * 3 + 1) // 2

        dist = torch.cdist(self.W, self.W)
        self.B = Mutable(torch.topk(dist, self.T, largest=False, dim=1).indices.to(torch.int32))

        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), 1e10, device=device))
        self.z = Mutable(torch.full((1, n_objs), 1e10, device=device))
        self.pi = Mutable(torch.ones((self.pop_size, 1), device=device))
        self.old_obj = Mutable(torch.full((self.pop_size, 1), 1e10, device=device))

        self.archive_pop = Mutable(torch.zeros((self.nEP, D), device=device))
        self.archive_fit = Mutable(torch.full((self.nEP, n_objs), 1e10, device=device))
        self.archive_size = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0, keepdim=True).values
        self.old_obj = torch.max(self.W * torch.abs(self.fit - self.z), dim=1, keepdim=True).values
        self.archive_pop, self.archive_fit, self.archive_size = _update_ep(
            self.archive_pop, self.archive_fit, self.archive_size, self.pop, self.fit, self.nEP
        )

    def step(self) -> None:
        device = self.lb.device
        N = self.pop_size
        M = self.n_objs

        # 2.1 Resource Allocation
        g = torch.max(self.W * torch.abs(self.fit - self.z), dim=1, keepdim=True).values
        delta = (self.old_obj - g) / (self.old_obj + 1e-6)
        decay = 0.95 + 0.05 * delta / 0.001
        self.pi = torch.where(delta > 0.001, torch.ones_like(self.pi), self.pi * decay)
        self.old_obj = g

        # 2.2 Subproblem Selection
        is_boundary = (self.W < 1e-3).sum(dim=1) == (M - 1)
        boundary_indices = torch.where(is_boundary)[0]
        num_boundary = boundary_indices.numel()

        num_candidates = max(0, (N // 5) - num_boundary)
        mating_indices = tournament_selection_multifit(num_candidates, [-self.pi.squeeze()], tournament_size=10)
        chosen_subproblems = torch.cat([boundary_indices, mating_indices])
        num_chosen = chosen_subproblems.numel()

        # 2.3 Variation & Neighborhood Update (Vectorized over chosen subproblems)
        for _ in range(5):
            # Parent selection for all chosen subproblems
            # rand_mask: 1 for neighborhood, 0 for whole population
            rand_mask = torch.rand(num_chosen, device=device) < 0.9

            # Generate parent indices
            p1_idx = torch.zeros(num_chosen, dtype=torch.long, device=device)
            p2_idx = torch.zeros(num_chosen, dtype=torch.long, device=device)

            # Neighborhood parents
            nb_indices = self.B[chosen_subproblems.long()]
            p1_nb = nb_indices[torch.arange(num_chosen), torch.randint(0, self.T, (num_chosen,), device=device)]
            p2_nb = nb_indices[torch.arange(num_chosen), torch.randint(0, self.T, (num_chosen,), device=device)]

            # Global parents
            p1_gl = torch.randint(0, N, (num_chosen,), device=device)
            p2_gl = torch.randint(0, N, (num_chosen,), device=device)

            p1_idx = torch.where(rand_mask, p1_nb.long(), p1_gl)
            p2_idx = torch.where(rand_mask, p2_nb.long(), p2_gl)

            # Variation
            parents = torch.stack([self.pop[p1_idx], self.pop[p2_idx]], dim=1)  # (num_chosen, 2, D)
            offspring = simulated_binary(parents.reshape(-1, parents.shape[-1])).reshape(num_chosen, 2, -1)[:, 0, :]
            offspring = polynomial_mutation(offspring, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)

            off_fit = self.evaluate(offspring)
            self.z = torch.min(self.z, torch.min(off_fit, dim=0, keepdim=True).values)

            # Update neighborhood for each chosen subproblem
            # This part is semi-vectorized to maintain MOEA/D logic
            for idx, sub_i in enumerate(chosen_subproblems):
                P_nb = self.B[sub_i.long()].long()
                g_old = torch.max(self.W[P_nb] * torch.abs(self.fit[P_nb] - self.z), dim=1).values
                g_new = torch.max(self.W[P_nb] * torch.abs(off_fit[idx] - self.z), dim=1).values

                mask = g_new <= g_old
                update_indices = torch.where(mask)[0]
                num_to_update = min(update_indices.numel(), self.nr)

                if num_to_update > 0:
                    # Use the first nr indices that improved
                    actual_idx = P_nb[update_indices[:num_to_update]]
                    self.pop[actual_idx] = offspring[idx].expand(actual_idx.numel(), -1)
                    self.fit[actual_idx] = off_fit[idx].expand(actual_idx.numel(), -1)

            # Update EP
            self.archive_pop, self.archive_fit, self.archive_size = _update_ep(
                self.archive_pop, self.archive_fit, self.archive_size, offspring, off_fit, self.nEP
            )

        # 2.4 Adaptive Weight Adjustment (AWA)
        # Using a fixed interval logic (wag=100) or probability
        if torch.rand(1, device=device) < 0.05:
            self.W, self.B, self.pop, self.fit = _awa_logic(
                self.W, self.pop, self.fit, self.archive_pop, self.archive_fit, self.archive_size, self.z, self.T
            )


def _update_ep(archive_pop, archive_fit, archive_size, new_pop, new_fit, max_size):
    device = archive_pop.device
    # Merge
    combined_pop = torch.cat([archive_pop[:archive_size], new_pop], dim=0)
    combined_fit = torch.cat([archive_fit[:archive_size], new_fit], dim=0)

    # Non-dominated filtering
    # f1 dominates f2 if f1 <= f2 and f1 < f2
    dom_mat = (combined_fit.unsqueeze(1) <= combined_fit.unsqueeze(0)).all(dim=-1) & (
        combined_fit.unsqueeze(1) < combined_fit.unsqueeze(0)
    ).any(dim=-1)
    is_nondominated = ~dom_mat.any(dim=0)

    curr_pop = combined_pop[is_nondominated]
    curr_fit = combined_fit[is_nondominated]

    # Pruning using product of K-nearest neighbors
    if curr_fit.shape[0] > max_size:
        M = curr_fit.shape[1]
        dist_mat = torch.cdist(curr_fit, curr_fit) + torch.eye(curr_fit.shape[0], device=device) * 1e10
        knn_dist = torch.topk(dist_mat, min(M, curr_fit.shape[0]), largest=False, dim=1).values
        density = torch.prod(knn_dist + 1e-6, dim=1)
        keep_idx = torch.topk(density, max_size, largest=True).indices
        curr_pop = curr_pop[keep_idx]
        curr_fit = curr_fit[keep_idx]

    new_size_val = curr_fit.shape[0]
    res_pop = torch.zeros_like(archive_pop)
    res_fit = torch.full_like(archive_fit, 1e10)
    res_pop[:new_size_val] = curr_pop
    res_fit[:new_size_val] = curr_fit
    return res_pop, res_fit, torch.tensor(new_size_val, dtype=torch.int32, device=device)


def _awa_logic(W, pop, fit, ep_pop, ep_fit, ep_size, z, T):
    device = W.device
    N, M = W.shape
    valid_ep_fit = ep_fit[:ep_size]
    valid_ep_pop = ep_pop[:ep_size]

    # 1. Re-assign
    all_fit = torch.cat([fit, valid_ep_fit], dim=0)
    all_pop = torch.cat([pop, valid_ep_pop], dim=0)

    # Tchebycheff for all solutions against all weights
    g_all = torch.max(W.unsqueeze(1) * torch.abs(all_fit.unsqueeze(0) - z), dim=2).values
    best_idx = torch.argmin(g_all, dim=1)

    new_pop = all_pop[best_idx]
    new_fit = all_fit[best_idx]

    # 2. Delete Overcrowded & Add New
    sub_dist = torch.cdist(new_fit, new_fit) + torch.eye(N, device=device) * 1e10
    knn_sub = torch.topk(sub_dist, min(M, N), largest=False, dim=1).values
    density_pop = torch.prod(knn_sub + 1e-6, dim=1)

    num_change = min(int(N * 0.05) + 1, ep_size)
    _, remove_candidates = torch.topk(density_pop, num_change, largest=False)

    dist_to_pop = torch.cdist(valid_ep_fit, new_fit).min(dim=1).values
    _, add_candidates = torch.topk(dist_to_pop, min(num_change, ep_size), largest=True)

    # Replace
    new_pop[remove_candidates[: add_candidates.numel()]] = valid_ep_pop[add_candidates]
    new_fit[remove_candidates[: add_candidates.numel()]] = valid_ep_fit[add_candidates]

    # Update Weights
    W_new = 1.0 / (torch.abs(new_fit - z) + 1e-6)
    W_new = W_new / (W_new.sum(dim=1, keepdim=True) + 1e-6)

    dist = torch.cdist(W_new, W_new)
    B_new = torch.topk(dist, T, largest=False, dim=1).indices.to(torch.int32)

    return W_new, B_new, new_pop, new_fit


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADAWA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
            fit = fit[~torch.any(torch.isinf(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")
