import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class WOF(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, gamma: int = 10, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.gamma = gamma

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.G = Mutable(torch.zeros(self.dim, dtype=torch.int32, device=device))
        self.speed = Mutable(torch.zeros((pop_size, self.dim), device=device))

        # Reference weights for internal optimization (Method 3: 10 weight sets)
        self.w_pop_size = 10

        # Sentinel for rank initialization (Bug #1)
        self.sentinel = torch.iinfo(torch.int32).max

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        # Initial Grouping (Method 2: Ordered)
        x_prime = self.pop[0]
        idx = torch.argsort(x_prime)

        # Vectorized Grouping (Blueprint 3.A)
        group_ids = torch.repeat_interleave(torch.arange(self.gamma, device=self.lb.device), self.dim // self.gamma)
        # Handle remainder
        remainder = self.dim - group_ids.shape[0]
        if remainder > 0:
            group_ids = torch.cat([group_ids, torch.full((remainder,), self.gamma - 1, device=self.lb.device)])

        new_G = torch.zeros_like(self.G)
        new_G.scatter_(0, idx, group_ids.to(torch.int32))
        self.G = new_G

    def _wof_select_x_primes(self, pop: torch.Tensor, fit: torch.Tensor, q: int) -> torch.Tensor:
        # Blueprint 4: Logic for Method 3
        # 1. Closest to axes
        best_idx = torch.argmin(fit, dim=0)
        # 2. Fill remaining with random
        rand_idx = torch.randperm(pop.shape[0], device=pop.device)

        # Combine indices (ensure unique if possible, but q is small)
        combined_idx = torch.cat([best_idx, rand_idx])
        selected_idx = combined_idx[:q]
        return pop[selected_idx]

    def _pso_update(self, x: torch.Tensor, v: torch.Tensor, pbest: torch.Tensor, gbest: torch.Tensor) -> tuple:
        # Blueprint 4: PSO Update with Boundary Clamping (Bug #38)
        phi = 0.7298
        c1, c2 = 1.496, 1.496
        r1 = torch.rand_like(x)
        r2 = torch.rand_like(x)

        v_new = phi * (v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x))
        x_new = x + v_new

        # Boundary logic
        out_mask = (x_new < self.lb) | (x_new > self.ub)
        v_new = torch.where(out_mask, -v_new, v_new)
        x_new = torch.clamp(x_new, self.lb, self.ub)
        return x_new, v_new

    def step(self) -> None:
        device = self.lb.device
        N = self.pop_size

        # --- Phase 1: Weight Optimization (W-Phase) ---
        # Select x' (Method 3: q=1)
        x_prime = self._wof_select_x_primes(self.pop, self.fit, 1)  # [1, D]

        # Generate weight population [10, gamma]
        w_pop = torch.rand(self.w_pop_size, self.gamma, device=device) * 2.0

        # Transform weights to decision space (Blueprint 3.B)
        W_batch = w_pop[:, self.G.long()]  # [10, D]
        mask = W_batch > 1.0
        X_weight = torch.where(mask, x_prime + (W_batch - 1.0) * (self.ub - x_prime), self.lb + W_batch * (x_prime - self.lb))

        # Evaluate weight-optimized solutions
        fit_weight = self.evaluate(X_weight)

        # --- Phase 2: Standard Search (X-Phase) ---
        # Use NSGA-II style mating
        rank = non_dominate_rank(self.fit)
        # Crowding distance requires mask (Bug #6)
        cd_mask = torch.ones(N, dtype=torch.bool, device=device)
        dist = crowding_distance(self.fit, cd_mask)

        mating_pool = tournament_selection_multifit(N, [-dist, rank.float()], tournament_size=2)
        offspring = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub, pro_m=1.0 / self.dim, dis_m=20.0)
        offspring = clamp(offspring, self.lb, self.ub)
        fit_off = self.evaluate(offspring)

        # --- Phase 3: Environmental Selection ---
        combined_pop = torch.cat([self.pop, X_weight, offspring], dim=0)
        combined_fit = torch.cat([self.fit, fit_weight, fit_off], dim=0)

        # Unique Filter (Bug #3)
        u_pop, u_idx = unique_rows_sorted(combined_pop)
        u_fit = combined_fit[u_idx]

        # NSGA-II Selection (Blueprint 3.C)
        n_total = u_pop.shape[0]
        ranks = non_dominate_rank(u_fit)

        selected_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        num_selected = 0
        current_rank = 0

        # Peeling Loop (Bug #9)
        # We use a fixed upper bound for rank iterations to stay JIT-safe
        for _ in range(n_total):
            if num_selected >= N:
                break

            curr_front_mask = ranks == current_rank
            num_in_front = curr_front_mask.sum()

            # Deadlock Breaker (Bug #9)
            if num_in_front == 0:
                remaining_needed = N - num_selected
                available_mask = ~selected_mask
                # Just take the first available ones
                available_indices = torch.where(available_mask, torch.arange(n_total, device=device), self.sentinel)
                take_idx = torch.sort(available_indices)[0][:remaining_needed]
                selected_mask[take_idx] = True
                num_selected = N
            elif num_selected + num_in_front <= N:
                selected_mask |= curr_front_mask
                num_selected += num_in_front
            else:
                # Crowding Distance inside loop (Bug #6, #21)
                dist_front = crowding_distance(u_fit, curr_front_mask)
                # Lexsort (Bug #25): Primary key (dist) last
                # We want high distance, so use -dist

                # Select top-k from this front
                remaining_needed = N - num_selected
                # A simpler JIT-safe way:
                # Use a very large penalty for things not in front
                penalty = torch.where(curr_front_mask, 0.0, 1e9)
                combined_crit_limited = torch.stack([-dist_front + penalty, ranks.float() + penalty])
                final_sort = lexsort(combined_crit_limited)

                selected_mask[final_sort[:remaining_needed]] = True
                num_selected = N

            current_rank += 1

        self.pop = u_pop[selected_mask][:N]
        self.fit = u_fit[selected_mask][:N]


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = WOF(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
