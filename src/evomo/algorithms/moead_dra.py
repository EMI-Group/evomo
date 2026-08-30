import torch
from evox.core import Algorithm, Mutable
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp


class MOEAD_DRA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, T: int = 20, nr: int = 2, **kwargs):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.T = T
        self.nr = nr
        D = lb.numel()

        # 1. Weight Generation & Neighborhood
        w, n_actual = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_actual
        self.T = min(T, self.pop_size)
        w = w.to(device)

        # Neighborhood Calculation (Bug #19)
        dist = torch.cdist(w, w)
        B = torch.topk(dist, k=self.T, largest=False, dim=1).indices

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), 1e10, device=device))
        self.w = Mutable(w)
        self.z = Mutable(torch.full((n_objs,), 1e10, device=device))
        self.B = Mutable(B)
        self.pi = Mutable(torch.ones(self.pop_size, device=device))
        self.old_obj = Mutable(torch.full((self.pop_size,), 1e10, device=device))
        self.gen_counter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def _tchebycheff_scalarization(self, objs: torch.Tensor, weights: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Broadcasting Logic: objs (N, M), weights (N, M), z (M,)
        return torch.max(torch.abs(objs - z) * weights, dim=-1).values

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0).values
        self.old_obj = self._tchebycheff_scalarization(self.fit, self.w, self.z)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size

        # 1. Subproblem Selection (DRA Logic)
        # boundary is where M-1 components are 0.
        boundary = torch.sum(self.w < 1e-3, dim=1) == (self.n_objs - 1)
        boundary_indices = torch.where(boundary)[0]
        num_boundary = boundary_indices.shape[0]
        num_to_select = max(0, (N // 5) - num_boundary)

        # Tournament Selection (Bug #27, #31)
        winners_idx = tournament_selection_multifit(num_to_select, [-self.pi], tournament_size=10)
        subproblem_indices = torch.cat([boundary_indices, winners_idx])

        # 2. Variation & Replacement (Sub-gen Loop)
        for _ in range(5):
            num_I = subproblem_indices.shape[0]
            sel_mask = torch.rand(num_I, device=device) < 0.9

            # Vectorized Parent Picking
            # For each i in I, pick 2 parents from B[i] or full pop
            r1 = torch.randint(0, self.T, (num_I,), device=device)
            r2 = torch.randint(0, self.T, (num_I,), device=device)
            rp1 = torch.randint(0, N, (num_I,), device=device)
            rp2 = torch.randint(0, N, (num_I,), device=device)

            p1_idx = torch.where(sel_mask, self.B[subproblem_indices, r1], rp1)
            p2_idx = torch.where(sel_mask, self.B[subproblem_indices, r2], rp2)

            # Offspring Generation (DE/rand/1)
            # Offspring = Population(i) + F * (Population(P1) - Population(P2))
            F = 0.5
            offspring = self.pop[subproblem_indices] + F * (self.pop[p1_idx] - self.pop[p2_idx])

            # Simple Crossover (CR=1.0 means offspring is fully DE result)
            # Polynomial Mutation (Simplified vectorized version)
            eta = 20.0
            r = torch.rand(offspring.shape, device=device)
            mu = torch.where(r < 0.5, (2.0 * r) ** (1.0 / (eta + 1.0)) - 1.0, 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0)))
            offspring = offspring + mu * (self.ub - self.lb) * 0.05  # 0.05 is a mutation strength factor

            offspring = clamp(offspring, self.lb, self.ub)
            off_fit = self.evaluate(offspring)

            # Update Ideal Point
            self.z = torch.min(self.z, torch.min(off_fit, dim=0).values)

            # Replacement Logic
            for j in range(num_I):
                idx_i = subproblem_indices[j]
                if sel_mask[j]:
                    P_indices = self.B[idx_i]
                else:
                    P_indices = torch.randperm(N, device=device)

                # Scalar values for neighbors in P
                g_old = torch.max(torch.abs(self.fit[P_indices] - self.z) * self.w[P_indices], dim=1).values
                g_new = torch.max(torch.abs(off_fit[j] - self.z) * self.w[P_indices], dim=1).values

                better_mask = g_new <= g_old
                # To handle the 'nr' limit without JIT breaks:
                # We find all better indices, shuffle them, and take the first nr.
                better_indices = P_indices[better_mask]
                num_better = better_indices.shape[0]

                # Use a conditional update to avoid .item() or empty tensor issues
                # We can use a mask-based update for the first nr elements
                perm = (
                    torch.randperm(num_better, device=device)
                    if num_better > 0
                    else torch.empty(0, dtype=torch.long, device=device)
                )
                replace_count = min(num_better, self.nr)

                # Update only if we found better solutions
                if num_better > 0:
                    to_replace = better_indices[perm[:replace_count]]
                    self.pop[to_replace] = offspring[j]
                    self.fit[to_replace] = off_fit[j]

        # 3. Utility Update (Every 10 Generations)
        self.gen_counter += 1
        if self.gen_counter >= 10:
            self.gen_counter = torch.tensor(0, dtype=torch.int32, device=device)
            new_obj = self._tchebycheff_scalarization(self.fit, self.w, self.z)
            # Bug #12: Safe Division
            delta = (self.old_obj - new_obj) / (self.old_obj + 1e-6)

            # Piecewise Update
            self.pi = torch.where(delta > 0.001, torch.ones_like(delta), (0.95 + 0.05 * delta / 0.001) * self.pi)
            self.old_obj = new_obj


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAD_DRA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
