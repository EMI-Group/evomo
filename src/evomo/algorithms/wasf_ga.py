import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp, lexsort


class WASFGA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, point: torch.Tensor = None, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # Algorithm Parameters
        self.ro = Parameter(0.0001)

        # Preferred Point (Default to zeros if not provided)
        if point is None:
            point = torch.zeros((1, n_objs), device=device)
        self.point = Mutable(point)

        # Weight Vector Generation
        if n_objs == 2:
            eps = 1e-6
            w1 = torch.linspace(eps, 1.0 - eps, pop_size, device=device)
            vectors = torch.stack([w1, 1.0 - w1], dim=1)
        else:
            vectors, _ = uniform_sampling(pop_size, n_objs)
            vectors = vectors.to(device=device)

        self.vectors = Mutable(vectors)
        self.v_size = vectors.shape[0]
        # Adjust pop_size to match vector count if necessary
        self.pop_size = self.v_size

        # Initialize State
        self.pop = Mutable(torch.rand(self.pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        self.front_no = Mutable(torch.full((self.pop_size,), 1, dtype=torch.int32, device=device))
        self.crowd_dis = Mutable(torch.zeros(self.pop_size, device=device))

    def _calculate_wasf_matrix(self, fit: torch.Tensor) -> torch.Tensor:
        # fit: [2N, M], self.point: [1, M], self.vectors: [V, M]
        f_diff = fit - self.point  # [2N, M]
        # Broadcasting: [2N, 1, M] * [1, V, M] -> [2N, V, M]
        term1 = f_diff.unsqueeze(1) * self.vectors.unsqueeze(0)
        max_val = torch.max(term1, dim=-1).values  # [2N, V]
        sum_val = torch.sum(term1, dim=-1)  # [2N, V]
        return max_val + self.ro * sum_val  # [2N, V]

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        # Initial selection to set front_no and crowd_dis
        self._environmental_selection(self.pop, self.fit)

    def _environmental_selection(self, combined_pop: torch.Tensor, combined_fit: torch.Tensor):
        N2 = combined_pop.shape[0]
        V = self.v_size
        sentinel = torch.iinfo(torch.int32).max

        # 1. WASF Matrix
        S = self._calculate_wasf_matrix(combined_fit)  # [2N, V]

        # 2. Peeling Loop
        front_no = torch.full((N2,), sentinel, dtype=torch.int32, device=combined_pop.device)
        crowd_dis = torch.zeros(N2, device=combined_pop.device)

        current_front = 1
        max_fronts = (N2 // V) + 1

        # JIT-compliant loop (fixed iteration count)
        for _ in range(max_fronts):
            mask = front_no == sentinel
            # Find best individual for each weight vector among available candidates
            # We use a large value for masked individuals to exclude them from argmin
            S_masked = torch.where(mask.unsqueeze(1), S, torch.tensor(float("inf"), device=S.device))

            # For each weight vector i, find the best individual j
            best_indices = torch.argmin(S_masked, dim=0)  # [V]

            # Update front_no for these individuals
            # Note: Multiple weight vectors might pick the same individual
            front_no[best_indices] = current_front

            # Calculate Crowding Distance for this front immediately
            current_mask = front_no == current_front
            crowd_dis = torch.where(current_mask, crowding_distance(combined_fit, current_mask), crowd_dis)

            current_front += 1

            # Check if all assigned (using any for JIT)
            if not torch.any(front_no == sentinel):
                # This break is technically allowed in some JIT contexts but we follow
                # the "no break" rule by letting the loop finish if needed,
                # though the logic remains stable.
                pass

        # Deadlock Breaker: Assign remaining to the last front
        remaining_mask = front_no == sentinel
        front_no = torch.where(remaining_mask, current_front, front_no)
        crowd_dis = torch.where(remaining_mask, crowding_distance(combined_fit, remaining_mask), crowd_dis)

        # 3. Final Selection
        # Primary key: front_no (min), Secondary key: -crowd_dis (max)
        # lexsort: primary key last
        idx = lexsort(torch.stack([-crowd_dis, front_no.float()]))[: self.pop_size]

        self.pop = combined_pop[idx]
        self.fit = combined_fit[idx]
        self.front_no = front_no[idx]
        self.crowd_dis = crowd_dis[idx]

    def step(self) -> None:
        # 1. Mating
        # Tournament selection: minimize front_no, maximize crowd_dis
        mating_pool = tournament_selection_multifit(
            self.pop_size, fitnesses=[self.front_no.float(), -self.crowd_dis], tournament_size=2
        )

        crossovered = simulated_binary(self.pop[mating_pool])
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(offspring)

        # 3. Merge and Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        self._environmental_selection(combined_pop, combined_fit)


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = WASFGA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
