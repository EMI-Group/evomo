import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp


class MOEADDE(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        delta: float = 0.9,
        nr: int = 2,
        F: float = 0.5,
        CR: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # Hyperparameters
        self.delta = Parameter(delta)
        self.nr = Parameter(nr)
        self.F = Parameter(F)
        self.CR = Parameter(CR)

        # 1. Determine actual population size first (Bug #13, Bug #19)
        W, n_actual = uniform_sampling(pop_size, n_objs)
        W = W.to(device)
        # Refinement (Bug #12)
        W = torch.where(W < 1e-6, torch.full_like(W, 1e-6, device=device), W)
        self.W = Mutable(W)
        self.pop_size = n_actual

        # 2. Neighborhood size T = ceil(N/10) -> (N + 9) // 10 (Bug #2)
        self.T = (n_actual + 9) // 10

        # 3. Initialize State (Mutables) with correct n_actual
        self.pop = Mutable(torch.rand(n_actual, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((n_actual, n_objs), torch.inf, device=device))

        # 4. Neighborhood Calculation
        dist_mat = torch.cdist(W, W)
        self.B = Mutable(torch.topk(dist_mat, k=self.T, largest=False).indices)

        # 5. Ideal Point
        self.z = Mutable(torch.full((n_objs,), torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        # Bug #10: dim=0 for objective-wise min
        self.z = torch.min(self.fit, dim=0).values

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size
        T = self.T

        # 1. Parent Selection (Vectorized)
        rand_mask = torch.rand(N, device=device) < self.delta

        # neighbor_parents: (N, 2)
        neighbor_indices = torch.randint(0, T, (N, 2), device=device)
        neighbor_parents = torch.gather(self.B, 1, neighbor_indices)

        # global_parents: (N, 2)
        global_parents = torch.randint(0, N, (N, 2), device=device)

        # P: (N, 2)
        P = torch.where(rand_mask.unsqueeze(1), neighbor_parents, global_parents)

        # 2. DE Operation & Mutation
        # off_pop[i] is generated using pop[i] and two parents from P[i]
        off_pop = self.pop + self.F * (self.pop[P[:, 0]] - self.pop[P[:, 1]])
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # 3. Evaluation
        off_fit = self.evaluate(off_pop)

        # 4. Update Ideal Point (Bug #10)
        self.z = torch.min(self.z, torch.min(off_fit, dim=0).values)

        # 5. Environmental Selection (Neighbor Replacement)
        # Tchebycheff Scalarization (Broadcasting)
        # We compare offspring i against its specific neighborhood B[i]
        W_neighbors = self.W[self.B]  # (N, T, M)
        fit_neighbors = self.fit[self.B]  # (N, T, M)
        off_fit_expanded = off_fit.unsqueeze(1)  # (N, 1, M)

        g_old = _tchebycheff_scalarization(W_neighbors, fit_neighbors, self.z)  # (N, T)
        g_new = _tchebycheff_scalarization(W_neighbors, off_fit_expanded, self.z)  # (N, T)

        # Replacement Logic with nr constraint (Bug #29, Bug #41)
        better_mask = g_new <= g_old
        # Use cumsum to limit to nr replacements per offspring
        final_mask = better_mask & (torch.cumsum(better_mask.to(torch.int32), dim=1) <= self.nr)

        # In-place Update (Parallel)
        indices_to_replace = self.B[final_mask]
        offspring_to_use = torch.arange(N, device=device).unsqueeze(1).expand(N, T)[final_mask]

        # Update population and fitness
        self.pop[indices_to_replace] = off_pop[offspring_to_use]
        self.fit[indices_to_replace] = off_fit[offspring_to_use]


def _tchebycheff_scalarization(weight: torch.Tensor, fit: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Bug #19: Implementation using tensor operations
    return torch.max(weight * torch.abs(fit - z), dim=-1).values


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADDE(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
