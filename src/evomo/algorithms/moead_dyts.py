import torch
from evox.core import Algorithm, Mutable
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp


class MOEADDYTS(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, T: int = 20, nr: int = 2, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.T = T
        self.nr = nr
        D = lb.numel()

        # 1. Weights & Neighborhood
        weights, n_actual = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_actual
        self.T = min(T, self.pop_size)
        weights = weights.to(device)
        dist = torch.cdist(weights, weights)
        B = torch.topk(dist, self.T, largest=False).indices.to(torch.int32)

        # 2. Initialize State (Mutables)
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        self.weights = Mutable(weights)
        self.B = Mutable(B)
        self.z = Mutable(torch.full((n_objs,), torch.inf, device=device))
        self.pi = Mutable(torch.ones(self.pop_size, device=device))
        self.old_obj = Mutable(torch.zeros(self.pop_size, device=device))
        self.beta_a = Mutable(torch.ones(5, device=device))
        self.beta_b = Mutable(torch.ones(5, device=device))
        self.gen_counter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0)[0]
        # Initial Tchebycheff
        diff = torch.abs(self.fit - self.z.unsqueeze(0))
        self.old_obj = torch.max(diff * self.weights, dim=1)[0]

    def _vectorized_de_op(
        self, op_idx: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> torch.Tensor:
        # op_idx is a scalar tensor. JIT-safe Python branching.
        F = 0.5
        if op_idx == 0:
            return p1 + F * (p2 - p3)
        elif op_idx == 1:
            return p1 + F * (p2 - p3) + F * (p4 - p5)
        elif op_idx == 2:
            return p1 + F * (p2 - p1) + F * (p3 - p4)
        elif op_idx == 3:
            return p1 + F * (p2 - p3)
        else:
            return p1 + F * (p2 - p3)

    def step(self) -> None:
        device = self.weights.device
        N = self.pop_size
        D = self.lb.numel()

        # 1. Subproblem Selection
        boundary_mask = torch.any(self.weights < 1e-4, dim=1)
        boundary_indices = torch.where(boundary_mask)[0]
        num_to_select = (N + 4) // 5  # 20% selection
        other_indices = tournament_selection_multifit(num_to_select, [-self.pi], tournament_size=2)
        subproblem_indices = torch.cat([boundary_indices, other_indices])
        batch_size = subproblem_indices.shape[0]

        # 2. Bandit Operator Selection
        samples = torch.distributions.Beta(self.beta_a, self.beta_b).sample()
        op_idx = torch.argmax(samples)
        CR = torch.where(op_idx > 2, torch.tensor(1.0, device=device), torch.tensor(0.5, device=device))

        # 3. Parent Selection & Mating
        # Neighborhood (80%) or Population (20%)
        use_neighbor = torch.rand(batch_size, device=device) < 0.8

        # Generate random indices for parents
        # For each i in I, pick from B[i] or arange(N)
        rand_neighbor = torch.randint(0, self.T, (batch_size, 5), device=device)
        rand_pop = torch.randint(0, N, (batch_size, 5), device=device)

        # Gather parent indices
        neighbor_parents = torch.gather(self.B[subproblem_indices], 1, rand_neighbor)
        p_idx = torch.where(use_neighbor.unsqueeze(1), neighbor_parents, rand_pop)

        # Extract parent tensors
        p1, p2, p3, p4, p5 = [self.pop[p_idx[:, k]] for k in range(5)]

        # DE Crossover
        offspring = self._vectorized_de_op(op_idx, p1, p2, p3, p4, p5)

        # Binomial Crossover (Vectorized)
        j_rand = torch.randint(0, D, (batch_size,), device=device)
        mask = torch.rand(batch_size, D, device=device) < CR
        mask[torch.arange(batch_size), j_rand] = True
        offspring = torch.where(mask, offspring, p1)

        # Mutation & Repair
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 4. Evaluation
        off_fit = self.evaluate(offspring)

        # 5. Environmental Update
        self.z = torch.min(self.z, torch.min(off_fit, dim=0)[0])

        # Update neighbors for each offspring
        improved_any = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # We must loop over the selected subproblems I to update their specific neighborhoods
        for j in range(batch_size):
            idx_i = subproblem_indices[j]
            neighbors = self.B[idx_i]

            # Tchebycheff comparison
            g_old = torch.max(torch.abs(self.fit[neighbors] - self.z) * self.weights[neighbors], dim=1)[0]
            g_new = torch.max(torch.abs(off_fit[j] - self.z) * self.weights[neighbors], dim=1)[0]

            replace_mask = g_new < g_old

            # Limit to nr replacements
            if torch.any(replace_mask):
                improved_any[j] = True
                # Find indices to replace
                replace_indices = torch.where(replace_mask)[0]
                # Take at most nr
                num_replace = torch.minimum(
                    torch.tensor(self.nr, device=device), torch.tensor(replace_indices.shape[0], device=device)
                )
                actual_replace = neighbors[replace_indices[:num_replace]]

                self.pop[actual_replace] = offspring[j]
                self.fit[actual_replace] = off_fit[j]

        # 6. Bandit Feedback
        self.beta_a[op_idx] = torch.clamp(self.beta_a[op_idx] + torch.sum(improved_any.float()), max=100.0)
        self.beta_b[op_idx] = torch.clamp(self.beta_b[op_idx] + torch.sum((~improved_any).float()), max=100.0)

        # 7. Utility Update
        self.gen_counter += 1
        if self.gen_counter % 10 == 0:
            new_obj = torch.max(torch.abs(self.fit - self.z) * self.weights, dim=1)[0]
            delta = (self.old_obj - new_obj) / (self.old_obj + 1e-6)
            self.pi = torch.where(delta > 0.001, torch.ones_like(self.pi), (0.95 + 0.05 * delta / 0.001) * self.pi)
            self.old_obj = new_obj


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADDYTS(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
