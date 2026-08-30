import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp


class PREA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.zmin = Mutable(torch.full((1, n_objs), torch.inf, device=device))

    def _calc_indicator_matrix(self, fit: torch.Tensor) -> torch.Tensor:
        N = fit.shape[0]
        device = fit.device
        # Shift Objectives
        f = fit - self.zmin + 1e-6
        # 3D Broadcasting for Ratio: (N, N, M)
        # ratio_ij = f_j / f_i
        # MATLAB: Ir = PopObj./repmat(Fi,Problem.N,1) - 1;
        # In our broadcasting: ratio[i, j, k] = f[j, k] / f[i, k]
        ratio = f.unsqueeze(0) / (f.unsqueeze(1) + 1e-6)

        # Max Indicator: I_ij = max(ratio - 1)
        i_matrix = torch.max(ratio - 1.0, dim=-1).values

        # Dominance Check: (X <= Y).all & (X < Y).any
        # f_i dominates f_j
        f_i = f.unsqueeze(1)  # (N, 1, M)
        f_j = f.unsqueeze(0)  # (1, N, M)
        dom_mask = (f_i <= f_j).all(-1) & (f_i < f_j).any(-1)

        # If i dominates j, I_ij = -max(1 - f_rj/f_ri)
        # InvertIr = repmat(Fi,Problem.N,1)./PopObj - 1;
        # MinIr = max(InvertIr,[],2);
        invert_ratio = f.unsqueeze(1) / (f.unsqueeze(0) + 1e-6)
        min_ir = torch.max(invert_ratio - 1.0, dim=-1).values
        i_matrix = torch.where(dom_mask, -min_ir, i_matrix)

        # Self-Mask: Replace fill_diagonal_ with eye addition for JIT compatibility
        eye_inf = torch.eye(N, device=device) * 1e18
        i_matrix = i_matrix + eye_inf
        return i_matrix

    def _parallel_distance(self, norm_fit: torch.Tensor) -> torch.Tensor:
        M = self.n_objs
        # diff shape: (N, N, M)
        diff = norm_fit.unsqueeze(1) - norm_fit.unsqueeze(0)
        # Formula: sqrt( sum(d^2) - (sum(d)^2 / M) )
        sum_sq = torch.sum(diff**2, dim=-1)
        sum_val = torch.sum(diff, dim=-1)
        p_dist = torch.sqrt(torch.clamp(sum_sq - (sum_val**2 / (M + 1e-6)), min=0.0) + 1e-6)
        return p_dist

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.zmin = torch.min(self.fit, dim=0, keepdim=True).values

    def step(self) -> None:
        N = self.pop_size
        device = self.lb.device

        # 1. Mating Pool Identification
        i_matrix_pop = self._calc_indicator_matrix(self.fit)
        neighbors = torch.argmin(i_matrix_pop, dim=1)
        rand_idx = torch.randint(0, N, (N,), device=device)
        prob = torch.rand((N,), device=device)
        p2_idx = torch.where(prob < 0.7, neighbors, rand_idx)

        # Interleave for OperatorGAhalf (N parents -> N/2 offspring)
        mating_idx = torch.stack([torch.arange(N, device=device), p2_idx], dim=1).view(-1)
        offspring = simulated_binary(self.pop[mating_idx], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        off_fit = self.evaluate(offspring)

        # Update Global Ideal Point
        self.zmin = torch.min(torch.cat([self.zmin, off_fit], dim=0), dim=0, keepdim=True).values

        # 2. Environmental Selection (PREA_Update)
        comb_pop = torch.cat([self.pop, offspring], dim=0)
        comb_fit = torch.cat([self.fit, off_fit], dim=0)
        # A. Promising Region Boundary
        i_matrix_comb = self._calc_indicator_matrix(comb_fit)
        # Level 1 Mask: individuals not dominated by any in indicator sense (min indicator >= 0)
        ir_fitness = torch.min(i_matrix_comb, dim=1).values
        is_promising = ir_fitness >= 0

        # If not enough promising solutions, take top N based on indicator fitness
        num_promising = torch.sum(is_promising.to(torch.int32))

        # Case 1: Promising solutions <= N
        # We use a conditional to handle the two branches of PREA_Update
        if num_promising <= N:
            _, sort_idx = torch.sort(ir_fitness, descending=True)
            survivor_indices = sort_idx[:N]
            self.pop = comb_pop[survivor_indices]
            self.fit = comb_fit[survivor_indices]
        else:
            # Case 2: Promising solutions > N
            # Filter to only promising solutions
            promising_idx = torch.where(is_promising)[0]
            p_pop = comb_pop[promising_idx]
            p_fit = comb_fit[promising_idx]
            p_imatrix = i_matrix_comb[promising_idx][:, promising_idx]
            num_p = p_fit.shape[0]

            # Algorithm 1: Peeling to find Zmax
            # We need to find N best to calculate Zmax
            active_p = torch.ones(num_p, dtype=torch.bool, device=device)
            temp_imatrix = p_imatrix.clone()

            # Peeling loop
            for _ in range(num_p):
                # Check if we still need to peel
                if torch.sum(active_p.to(torch.int32)) > N:
                    # Find worst in current subset
                    # Mask out inactive rows/cols
                    row_min = torch.min(
                        torch.where(
                            active_p.unsqueeze(0) & active_p.unsqueeze(1),
                            temp_imatrix,
                            torch.tensor(float("inf"), device=device),
                        ),
                        dim=1,
                    ).values
                    worst_idx = torch.argmin(row_min)
                    active_p[worst_idx] = False
                    temp_imatrix[worst_idx, :] = float("inf")
                    temp_imatrix[:, worst_idx] = float("inf")

            best_idx = torch.where(active_p)[0]
            zmax = torch.max(p_fit[best_idx], dim=0).values

            # Remove individuals outside promising region (Algorithm 1, line 23)
            # min(zmax - p_fit, dim=1) < 0 means at least one objective > zmax
            outside_mask = torch.any(p_fit > zmax, dim=1)
            inside_idx = torch.where(~outside_mask)[0]

            # Valuable candidate set
            v_pop = p_pop[inside_idx]
            v_fit = p_fit[inside_idx]
            v_imatrix = p_imatrix[inside_idx][:, inside_idx]
            num_v = v_fit.shape[0]

            # Algorithm 2: Diversity maintenance
            norm_fit = (v_fit - self.zmin) / (zmax - self.zmin + 1e-6)
            p_dist = self._parallel_distance(norm_fit)
            # Self-mask diagonal
            p_dist = p_dist + torch.eye(num_v, device=device) * 1e18

            v_ir_fitness = torch.min(v_imatrix, dim=1).values
            active_v = torch.ones(num_v, dtype=torch.bool, device=device)

            # Pruning loop from num_v to N
            for _ in range(num_v):
                if torch.sum(active_v.to(torch.int32)) > N:
                    # Find pair with min distance
                    # Mask inactive
                    mask_2d = active_v.unsqueeze(0) & active_v.unsqueeze(1)
                    masked_dist = torch.where(mask_2d, p_dist, torch.tensor(float("inf"), device=device))

                    flat_idx = torch.argmin(masked_dist)
                    i_idx = flat_idx // num_v
                    j_idx = flat_idx % num_v

                    # Compare indicators of the pair
                    cond = v_ir_fitness[i_idx] < v_ir_fitness[j_idx]
                    remove_idx = torch.where(cond, i_idx, j_idx)

                    active_v[remove_idx] = False
                    p_dist[remove_idx, :] = float("inf")
                    p_dist[:, remove_idx] = float("inf")

            # Final Selection
            survivor_indices = torch.where(active_v)[0]
            # If for some reason (e.g. all outside) we have < N, pad with others
            if survivor_indices.shape[0] < N:
                # Deadlock breaker: if pruning logic fails to provide N, take top N by indicator
                _, fallback_idx = torch.sort(ir_fitness, descending=True)
                self.pop = comb_pop[fallback_idx[:N]]
                self.fit = comb_fit[fallback_idx[:N]]
            else:
                self.pop = v_pop[survivor_indices[:N]]
                self.fit = v_fit[survivor_indices[:N]]


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = PREA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
