import torch
from evox.core import Algorithm, Mutable
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, randint

from evomo.operators.selection import non_dominate_rank


class BCEMOEAD(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        T: int | None = None,
        nr: int | None = None,
        **kwargs,
    ):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        # 1. Weights & neighbours
        W, n_actual = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_actual
        default_T = (self.pop_size + 9) // 10
        default_nr = (self.pop_size + 99) // 100
        self.T = min(max(2, default_T if T is None else T), self.pop_size)
        self.nr = min(max(1, default_nr if nr is None else nr), self.pop_size)
        W = W.to(device)
        self.W = Mutable(W)

        dist = torch.cdist(W, W)
        self.B = Mutable(torch.topk(dist, self.T, largest=False).indices)

        # 2. NPC is initialized first; PC is selected from NPC in init_step.
        initial_pop = torch.rand(self.pop_size, self.dim, device=device) * (ub - lb) + lb
        self.pop = Mutable(initial_pop.clone())
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        self.npc_pop = Mutable(initial_pop)
        self.npc_fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

        self.z = Mutable(torch.zeros((1, n_objs), device=device))
        self.nND = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.npc_fit = self.evaluate(self.npc_pop)
        self.z = torch.min(self.npc_fit, dim=0, keepdim=True).values
        self.pop, self.fit, self.nND = self._pc_selection(self.npc_pop, self.npc_fit)

    def _cal_tchebycheff(self, fit: torch.Tensor, z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # Bug #12: Safe division
        return torch.max(torch.abs(fit - z) / (W + 1e-6), dim=-1).values

    def _operator_ga_half(self, parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
        """Generate one offspring for each pair of parents."""
        mu = torch.rand(parent1.shape, device=parent1.device)
        beta = torch.where(mu <= 0.5, torch.pow(2 * mu, 1 / 21.0), torch.pow(2 - 2 * mu, -1 / 21.0))
        beta = beta * (1 - torch.randint(0, 2, beta.shape, device=parent1.device) * 2)
        beta = torch.where(torch.rand(beta.shape, device=parent1.device) < 0.5, 1, beta)
        offspring = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        return clamp(offspring, self.lb, self.ub)

    def _niche_deletion(self, pop: torch.Tensor, fit: torch.Tensor, topk: int):
        f_min = torch.min(fit, dim=0).values
        f_max = torch.max(fit, dim=0).values
        norm_fit = (fit - f_min) / (f_max - f_min + 1e-6)

        dist_matrix = torch.cdist(norm_fit, norm_fit)
        dist_matrix.fill_diagonal_(torch.inf)
        neighbor_rank = min(3, fit.shape[0])
        radius = torch.topk(dist_matrix, neighbor_rank, largest=False).values[:, neighbor_rank - 1].mean()
        R = torch.clamp(dist_matrix / (radius + 1e-6), max=1.0)

        active_mask = torch.ones(fit.shape[0], dtype=torch.bool, device=fit.device)
        for _ in range(fit.shape[0] - topk):
            R_curr = R[active_mask][:, active_mask]
            density = 1 - torch.prod(R_curr, dim=1)
            local_idx = torch.argmax(density)
            global_indices = torch.where(active_mask)[0]
            active_mask[global_indices[local_idx]] = False

        return pop[active_mask], fit[active_mask]

    def _pc_selection(self, pop: torch.Tensor, fit: torch.Tensor):
        rank = non_dominate_rank(fit)
        is_nd = rank == 0
        pc_pop = pop[is_nd]
        pc_fit = fit[is_nd]
        n_nd = torch.sum(is_nd).to(torch.int32)

        permutation = torch.randperm(pc_pop.shape[0], device=pop.device)
        pc_pop = pc_pop[permutation]
        pc_fit = pc_fit[permutation]
        if pc_pop.shape[0] > self.pop_size:
            pc_pop, pc_fit = self._niche_deletion(pc_pop, pc_fit, self.pop_size)

        return pc_pop, pc_fit, n_nd

    def _exploration(self):
        pc_size = self.pop.shape[0]
        f_min = torch.min(self.fit, dim=0).values
        f_max = torch.max(self.fit, dim=0).values
        norm_pc = (self.fit - f_min) / (f_max - f_min + 1e-6)
        norm_npc = (self.npc_fit - f_min) / (f_max - f_min + 1e-6)

        dist_pc = torch.cdist(norm_pc, norm_pc)
        dist_pc.fill_diagonal_(torch.inf)
        neighbor_rank = min(3, pc_size)
        r0 = torch.topk(dist_pc, neighbor_rank, largest=False).values[:, neighbor_rank - 1].mean()
        r = (self.nND.float() / self.pop_size) * r0

        exploration_mask = (torch.cdist(norm_pc, norm_npc) <= r).sum(dim=1) <= 1
        random_mates = randint(0, pc_size, (pc_size,), device=self.pop.device)
        parent1 = self.pop[exploration_mask]
        parent2 = self.pop[random_mates[exploration_mask]]
        return self._operator_ga_half(parent1, parent2)

    def _update_npc_with_new_pc(self, new_pc: torch.Tensor, new_pc_fit: torch.Tensor) -> None:
        if new_pc.shape[0] == 0:
            return

        self.z = torch.min(torch.cat([self.z, new_pc_fit], dim=0), dim=0, keepdim=True).values
        g_old = self._cal_tchebycheff(self.npc_fit, self.z, self.W)
        diff = torch.abs(new_pc_fit.unsqueeze(1) - self.z.unsqueeze(0))
        g_new = torch.max(diff / (self.W.unsqueeze(0) + 1e-6), dim=-1).values
        eligible = g_new <= g_old.unsqueeze(0)

        # A random permutation followed by the first eligible item is
        # equivalent to selecting one eligible NPC uniformly at random.
        priorities = torch.rand(eligible.shape, device=self.pop.device).masked_fill(~eligible, torch.inf)
        chosen = torch.argmin(priorities, dim=1)
        has_candidate = eligible.any(dim=1)
        proposals = torch.zeros_like(eligible)
        proposals.scatter_(1, chosen.unsqueeze(1), has_candidate.unsqueeze(1))

        proposal_values = torch.where(proposals, g_new, torch.inf)
        best_new_pc = torch.argmin(proposal_values, dim=0)
        replace = proposals.any(dim=0)
        self.npc_pop[replace] = new_pc[best_new_pc[replace]]
        self.npc_fit[replace] = new_pc_fit[best_new_pc[replace]]

    def _evolve_npc(self):
        N = self.pop_size
        device = self.pop.device

        neighbor_order = torch.argsort(torch.rand((N, self.T), device=device), dim=1)
        neighbor_parents = torch.gather(self.B, 1, neighbor_order[:, :2])
        global_parents = torch.topk(torch.rand((N, N), device=device), 2, largest=False).indices
        use_neighbor = torch.rand(N, device=device) < 0.9
        parents = torch.where(use_neighbor.unsqueeze(1), neighbor_parents, global_parents)

        new_npc = self._operator_ga_half(self.npc_pop[parents[:, 0]], self.npc_pop[parents[:, 1]])
        new_npc_fit = self.evaluate(new_npc)
        self.z = torch.min(torch.cat([self.z, new_npc_fit], dim=0), dim=0, keepdim=True).values

        diff = torch.abs(new_npc_fit.unsqueeze(1) - self.z.unsqueeze(0))
        g_new = torch.max(diff / (self.W.unsqueeze(0) + 1e-6), dim=-1).values
        g_old = self._cal_tchebycheff(self.npc_fit, self.z, self.W)

        neighbor_candidates = torch.zeros((N, N), dtype=torch.bool, device=device)
        neighbor_candidates.scatter_(1, self.B, True)
        candidates = torch.where(use_neighbor.unsqueeze(1), neighbor_candidates, torch.ones_like(neighbor_candidates))
        eligible = candidates & (g_new <= g_old.unsqueeze(0))

        priorities = torch.rand((N, N), device=device).masked_fill(~eligible, torch.inf)
        selected = torch.topk(priorities, self.nr, dim=1, largest=False).indices
        selected_valid = torch.gather(eligible, 1, selected)
        proposals = torch.zeros_like(eligible)
        proposals.scatter_(1, selected, selected_valid)

        proposal_values = torch.where(proposals, g_new, torch.inf)
        best_offspring = torch.argmin(proposal_values, dim=0)
        replace = proposals.any(dim=0)
        self.npc_pop[replace] = new_npc[best_offspring[replace]]
        self.npc_fit[replace] = new_npc_fit[best_offspring[replace]]
        return new_npc, new_npc_fit

    def step(self) -> None:
        new_pc = self._exploration()
        new_pc_fit = self.evaluate(new_pc)
        self._update_npc_with_new_pc(new_pc, new_pc_fit)

        new_npc, new_npc_fit = self._evolve_npc()
        combined_pop = torch.cat([self.pop, new_npc, new_pc], dim=0)
        combined_fit = torch.cat([self.fit, new_npc_fit, new_pc_fit], dim=0)
        self.pop, self.fit, self.nND = self._pc_selection(combined_pop, combined_fit)


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = BCEMOEAD(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
