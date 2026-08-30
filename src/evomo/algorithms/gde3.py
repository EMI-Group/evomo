import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.selection import crowding_distance
from evox.utils import clamp, lexsort, randint

from evomo.operators.selection import non_dominate_rank


class GDE3(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, F: float = 0.5, CR: float = 0.5):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # Hyperparameters
        self.F = Parameter(F)
        self.CR = Parameter(CR)

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

    def step(self) -> None:
        device = self.pop.device
        N = self.pop_size

        # 1. Mating (Vectorized DE/rand/1/bin)
        # Generate r1, r2, r3 such that r1 != r2 != r3 != i
        r1 = randint(0, N, (N,), device=device)
        r2 = randint(0, N, (N,), device=device)
        r3 = randint(0, N, (N,), device=device)

        # Simple shift to avoid identity (Bug #29 compliance - vectorized)
        idx = torch.arange(N, device=device)
        r1 = torch.where(r1 == idx, (r1 + 1) % N, r1)
        r2 = torch.where((r2 == idx) | (r2 == r1), (r2 + 2) % N, r2)
        r3 = torch.where((r3 == idx) | (r3 == r1) | (r3 == r2), (r3 + 3) % N, r3)

        mutant = self.pop[r1] + self.F * (self.pop[r2] - self.pop[r3])

        # Crossover
        rand_mask = torch.rand((N, self.dim), device=device) < self.CR
        # Ensure at least one dimension is swapped
        j_rand = randint(0, self.dim, (N,), device=device)
        force_mask = torch.nn.functional.one_hot(j_rand, num_classes=self.dim).bool()
        rand_mask = rand_mask | force_mask

        off_pop = torch.where(rand_mask, mutant, self.pop)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(off_pop)

        # 3. Selection
        self.pop, self.fit = self._gde3_selection(self.pop, self.fit, off_pop, off_fit)

    def _gde3_selection(self, pop, fit, off_pop, off_fit):
        device = pop.device
        N = self.pop_size

        # Phase A: One-to-One Comparison (Pareto Dominance Bug #24)
        off_dom_parent = (off_fit <= fit).all(dim=1) & (off_fit < fit).any(dim=1)
        parent_dom_off = (fit <= off_fit).all(dim=1) & (fit < off_fit).any(dim=1)

        # Logic:
        # replace_mask: offspring replaces parent
        # add_mask: non-dominated, both enter Phase B
        replace_mask = off_dom_parent
        add_mask = ~(off_dom_parent | parent_dom_off)

        updated_pop = torch.where(replace_mask.unsqueeze(1), off_pop, pop)
        updated_fit = torch.where(replace_mask.unsqueeze(1), off_fit, fit)

        combined_pop = torch.cat([updated_pop, off_pop[add_mask]], dim=0)
        combined_fit = torch.cat([updated_fit, off_fit[add_mask]], dim=0)

        # Phase B: Global Reduction (Non-dominated Sorting)
        ranks = non_dominate_rank(combined_fit)

        # Peeling Logic (Bug #9, #41)
        num_combined = combined_fit.shape[0]
        selected_mask = torch.zeros(num_combined, dtype=torch.bool, device=device)
        current_count = 0

        # Iterate through possible ranks (max possible rank is num_combined)
        # We use a loop but the logic inside is vectorized.
        for r in range(num_combined):
            front_mask = ranks == r
            num_in_front = torch.sum(front_mask.int())

            # Check if we can add the whole front
            can_add_all = (current_count + num_in_front) <= N

            # Case 1: Add whole front
            add_now = front_mask & can_add_all
            selected_mask = selected_mask | add_now

            # Case 2: Front overflows N - Crowding Distance Selection
            # We only process the overflow if we haven't reached N yet and this front would exceed N
            is_overflow_front = (~can_add_all) & (current_count < N)

            # Calculate CD only for the overflow front (Bug #9, #21)
            # We use a dummy CD for others to keep it JIT friendly
            num_needed = N - current_count

            # This block executes once for the overflow front
            if is_overflow_front.any():
                cd = crowding_distance(combined_fit, front_mask)
                # lexsort: primary key last. We want largest CD, so use -cd.
                # Bug #25: lexsort(torch.stack([-cd]))
                front_indices = torch.where(front_mask)[0]
                cd_values = cd[front_indices]

                # Sort indices of the front by CD descending
                rel_idx = lexsort(torch.stack([-cd_values]))
                sel_rel_idx = rel_idx[:num_needed]
                selected_mask[front_indices[sel_rel_idx]] = True

            current_count = current_count + num_in_front

            # Termination check (JIT friendly via mask count)
            if current_count >= N:
                # We use a logical trick: if we have enough, the loop continues
                # but selected_mask won't change because current_count < N will be false.
                pass

        # Final slice to ensure exactly N (handles edge cases)
        # In GDE3, we take the first N based on the mask
        final_indices = torch.where(selected_mask)[0][:N]
        return combined_pop[final_indices], combined_fit[final_indices]


# === FIXED DEMO BLOCK ===
# This block MUST be appended at the end of the file.
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    # GDE3 must be replaced by your actual class name
    algo = GDE3(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
