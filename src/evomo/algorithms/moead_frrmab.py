import torch
from evox.core import Algorithm, Mutable
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp


class MOEADFRRMAB(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        T: int = 20,
        delta: float = 0.9,
        nr: int = 2,
        window_size: int = None,
        **kwargs,
    ):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # Parameters
        self.T = T
        self.delta = delta
        self.nr = nr

        # Weights and Neighborhood - uniform_sampling may change pop_size
        weights, actual_pop_size = uniform_sampling(pop_size, n_objs)
        self.pop_size = int(actual_pop_size)
        self.T = min(T, self.pop_size)
        self.weights = Mutable(weights.to(device))

        # Initialize State (Mutables) with actual_pop_size
        self.pop = Mutable(torch.rand(self.pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

        dist = torch.cdist(self.weights, self.weights)
        self.neighbors = Mutable(torch.topk(dist, self.T, largest=False, dim=1).indices.to(torch.int32))

        # Ideal Point
        self.z = Mutable(torch.full((1, n_objs), torch.inf, device=device))

        # MAB and Utility
        self.W = window_size if window_size is not None else (self.pop_size + 1) // 2
        self.D_decay = 1.0  # Decay factor for credit assignment

        self.pi = Mutable(torch.ones(self.pop_size, device=device))
        self.old_obj = Mutable(torch.zeros(self.pop_size, device=device))
        self.sw = Mutable(torch.zeros((2, self.W), device=device))  # Row 0: Op, Row 1: Reward
        self.frr = Mutable(torch.zeros(4, device=device))
        self.gen_counter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0, keepdim=True).values

        # Initial Tchebycheff
        diff = torch.abs(self.fit - self.z)
        self.old_obj = torch.max(diff * self.weights, dim=1).values

    def _apply_four_de(self, parents: torch.Tensor, op_indices: torch.Tensor, current_x: torch.Tensor) -> torch.Tensor:
        # parents shape: [I_size, 5, D]
        F = 0.5
        K = 0.5  # Scaling factor for current-to-rand

        r1, r2, r3, r4, r5 = parents[:, 0], parents[:, 1], parents[:, 2], parents[:, 3], parents[:, 4]

        # DE/rand/1
        v0 = r1 + F * (r2 - r3)
        # DE/rand/2
        v1 = r1 + F * (r2 - r3) + F * (r4 - r5)
        # DE/current-to-rand/2 (MATLAB op 3)
        v2 = current_x + K * (current_x - r1) + F * (r2 - r3) + F * (r4 - r5)
        # DE/current-to-rand/1 (MATLAB op 4)
        v3 = current_x + K * (current_x - r1) + F * (r2 - r3)

        # Stack and Gather
        candidates = torch.stack([v0, v1, v2, v3], dim=1)  # [I_size, 4, D]
        idx = op_indices.view(-1, 1, 1).expand(-1, 1, self.dim)
        offspring_v = torch.gather(candidates, 1, idx).squeeze(1)

        # Crossover logic: op 1,2 use CR=1.0 (from MATLAB code), op 3,4 use CR=1.0 + (op>2)
        # Actually MATLAB code says: Site = rand(1,D) < (CR+(op>2)); where CR=1.
        # This means for all ops, Site is always true (1.0 < 1.0 is false, but 1.0 <= 1.0 is true).
        # In PlatEMO FourDE, CR is 1, so Site is always true.
        return offspring_v

    def _update_sliding_window(self, op: torch.Tensor, reward: torch.Tensor):
        new_entry = torch.stack([op.to(torch.float32), reward])
        self.sw = torch.roll(self.sw, shifts=-1, dims=1)
        self.sw[:, -1] = new_entry

    def step(self) -> None:
        self.gen_counter += 1
        device = self.pop.device

        # Sub-problem Selection
        boundary = torch.sum(self.weights < 1e-3, dim=1) == (self.n_objs - 1)
        boundary_indices = torch.where(boundary)[0]
        num_boundary = boundary_indices.numel()
        num_tour = max(0, (self.pop_size // 5) - num_boundary)

        tour_idx = tournament_selection_multifit(num_tour, [-self.pi], tournament_size=10)
        subproblem_indices = torch.cat([boundary_indices, tour_idx])
        I_size = subproblem_indices.numel()

        # Operator Selection (MAB)
        ops_in_sw = self.sw[0]
        n_j = torch.stack([(ops_in_sw == (j + 1)).sum() for j in range(4)])  # Ops are 1-indexed in SW
        sw_len = self.W

        # Handle initial case where FRR or SW is empty
        use_random = torch.logical_or(torch.any(self.frr == 0), torch.any(n_j == 0))

        ucb = self.frr + 5.0 * torch.sqrt(2 * torch.log(torch.tensor(float(sw_len), device=device)) / (n_j + 1e-6))
        selected_op_idx = torch.where(use_random, torch.randint(0, 4, (), device=device), torch.argmax(ucb))

        # Mating and Update (Loop 5 times as per MATLAB subgeneration)
        for _ in range(5):
            use_neighbor = torch.rand(I_size, device=device) < self.delta

            rand_neighbor = torch.randint(0, self.T, (I_size, 5), device=device)
            neighbor_pool = self.neighbors[subproblem_indices]
            p_neighbor = torch.gather(neighbor_pool, 1, rand_neighbor)
            p_all = torch.randint(0, self.pop_size, (I_size, 5), device=device)
            P_idx = torch.where(use_neighbor.view(-1, 1), p_neighbor, p_all)

            parents = self.pop[P_idx]
            op_indices = torch.full((I_size,), selected_op_idx, device=device)
            offspring = self._apply_four_de(parents, op_indices, self.pop[subproblem_indices])
            offspring = polynomial_mutation(offspring, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)

            off_fit = self.evaluate(offspring)
            self.z = torch.min(self.z, torch.min(off_fit, dim=0, keepdim=True).values)

            # Update neighbors
            # For each i in I, update its neighborhood P
            # To vectorize: we update the neighborhood of each i in I
            P_to_update = self.neighbors[subproblem_indices]  # [I_size, T]

            W_sub = self.weights[P_to_update]  # [I_size, T, M]
            g_old = torch.max(torch.abs(self.fit[P_to_update] - self.z) * W_sub, dim=2).values
            g_new = torch.max(torch.abs(off_fit.unsqueeze(1) - self.z) * W_sub, dim=2).values

            replace_mask = g_new <= g_old
            # Limit to nr replacements
            replace_counts = torch.cumsum(replace_mask.to(torch.int32), dim=1)
            replace_mask = torch.logical_and(replace_mask, replace_counts <= self.nr)

            # Scatter updates
            flat_P = P_to_update[replace_mask]
            flat_off = torch.arange(I_size, device=device).view(-1, 1).expand(-1, self.T)[replace_mask]

            self.pop[flat_P] = offspring[flat_off]
            self.fit[flat_P] = off_fit[flat_off]

            # FIR and Sliding Window
            fir_val = torch.sum((g_old[replace_mask] - g_new[replace_mask]) / (g_old[replace_mask] + 1e-6))
            self._update_sliding_window(selected_op_idx + 1, fir_val)

            # Credit Assignment
            sw_ops = self.sw[0]
            sw_rewards = self.sw[1]
            op_rewards = torch.stack([torch.sum(sw_rewards[sw_ops == (j + 1)]) for j in range(4)])

            # Rank-based
            sorted_rewards_idx = torch.argsort(op_rewards, descending=True)
            ranks = torch.argsort(sorted_rewards_idx).to(torch.float32)
            decay = (self.D_decay**ranks) * op_rewards
            self.frr = decay / (torch.sum(decay) + 1e-6)

        # Utility Update
        if self.gen_counter % 10 == 0:
            new_obj = torch.max(torch.abs(self.fit - self.z) * self.weights, dim=1).values
            delta_obj = (self.old_obj - new_obj) / (self.old_obj + 1e-6)
            pi_mask = delta_obj < 0.001
            self.pi = torch.where(pi_mask, (0.95 + 0.05 * delta_obj / 0.001) * self.pi, torch.ones_like(self.pi))
            self.old_obj = new_obj


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADFRRMAB(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
