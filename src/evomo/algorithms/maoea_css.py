import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, randint

from evomo.utils import unique_rows_sorted


class MaOEACSS(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, t: float = 0.0, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.t = Parameter(torch.tensor(t, device=device))

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.zmin = Mutable(torch.full((n_objs,), torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.zmin = torch.min(self.fit, dim=0).values

    def _get_cosine_angle_matrix(self, x: torch.Tensor) -> torch.Tensor:
        # Bug #29: Vectorized pairwise cosine similarity
        norm = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-6)
        cos_sim = torch.matmul(norm, norm.t())
        angle = torch.acos(torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7))
        return angle

    def _sequential_elimination(self, fit: torch.Tensor, zmin: torch.Tensor, n_required: int) -> torch.Tensor:
        # Bug #30: Brutal Static Truncation with Masking
        n_total = fit.shape[0]
        f_norm = fit - zmin
        con = torch.norm(f_norm, p=2, dim=1)
        angle_matrix = self._get_cosine_angle_matrix(f_norm)

        # Set diagonal to infinity to avoid self-selection (Bug #41: Avoid fill_diagonal_)
        sentinel_inf = 1e9
        eye_mask = torch.eye(n_total, device=fit.device, dtype=torch.bool)
        angle_matrix = torch.where(eye_mask, torch.full_like(angle_matrix, sentinel_inf), angle_matrix)

        mask = torch.ones(n_total, device=fit.device, dtype=torch.bool)
        n_to_remove = n_total - n_required

        # Sequential logic using masking to keep it JIT-friendly.
        for _ in range(n_to_remove):
            # Find global minimum angle among active individuals
            active_mask = mask.unsqueeze(1) & mask.unsqueeze(0)
            active_angles = torch.where(active_mask, angle_matrix, torch.full_like(angle_matrix, sentinel_inf))

            # Bug #41: Find indices of min angle without .item()
            flat_idx = torch.argmin(active_angles.view(-1))
            i = flat_idx // n_total
            j = flat_idx % n_total

            # Compare convergence
            # If Con[i] - Con[j] > self.t, mask i. Else if Con[j] - Con[i] > self.t, mask j. Else mask i.
            diff = con[i] - con[j]
            remove_i = torch.where(
                diff > self.t,
                torch.tensor(True, device=fit.device),
                torch.where(
                    con[j] - con[i] > self.t, torch.tensor(False, device=fit.device), torch.tensor(True, device=fit.device)
                ),
            )

            idx_to_remove = torch.where(remove_i, i, j)

            # Update mask and matrix
            mask[idx_to_remove] = False
            # Effectively remove the individual from future min calculations
            angle_matrix[idx_to_remove, :] = sentinel_inf
            angle_matrix[:, idx_to_remove] = sentinel_inf

        return mask

    def step(self) -> None:
        # 1. Mating Selection
        # ASF Calculation (Bug #12: Safe Division)
        w = torch.clamp(self.fit / (self.fit.sum(dim=1, keepdim=True) + 1e-6), min=1e-6)
        asf = torch.max((self.fit - self.zmin) / w, dim=1).values
        asf_rank = torch.argsort(torch.argsort(asf)).float()

        # Diversity (Amin)
        f_norm_current = self.fit - self.zmin
        angle_current = self._get_cosine_angle_matrix(f_norm_current)
        # Avoid fill_diagonal_ for JIT
        angle_current = torch.where(
            torch.eye(self.pop_size, device=self.pop.device, dtype=torch.bool),
            torch.tensor(1e9, device=self.pop.device),
            angle_current,
        )
        a_min = torch.min(angle_current, dim=1).values

        # Tournament (Bug #27: Selection Pressure)
        p1 = randint(0, self.pop_size, (self.pop_size,), device=self.pop.device)
        p2 = randint(0, self.pop_size, (self.pop_size,), device=self.pop.device)
        winner_mask = (asf[p1] < asf[p2]) & (a_min[p1] > a_min[p2])
        selected_idx = torch.where(winner_mask, p1, p2)

        # Probabilistic Refinement
        prob = 1.0002 - (asf_rank[selected_idx] / self.pop_size)
        final_idx = torch.where(
            torch.rand(self.pop_size, device=self.pop.device) < prob,
            selected_idx,
            randint(0, self.pop_size, (self.pop_size,), device=self.pop.device),
        )

        # 2. Variation
        off_pop = simulated_binary(self.pop[final_idx], pro_c=1.0, dis_c=20.0)
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub, pro_m=1.0 / self.dim, dis_m=20.0)
        off_pop = clamp(off_pop, self.lb, self.ub)
        off_fit = self.evaluate(off_pop)

        # Update zmin
        self.zmin = torch.min(torch.min(off_fit, dim=0).values, self.zmin)

        # 3. Environmental Selection
        total_pop = torch.cat([self.pop, off_pop], dim=0)
        total_fit = torch.cat([self.fit, off_fit], dim=0)

        # Unique rows (Bug #3)
        u_pop, u_idx = unique_rows_sorted(total_pop)
        u_fit = total_fit[u_idx]

        # Sequential Elimination
        survivor_mask = self._sequential_elimination(u_fit, self.zmin, self.pop_size)

        self.pop = u_pop[survivor_mask]
        self.fit = u_fit[survivor_mask]


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MaOEACSS(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
