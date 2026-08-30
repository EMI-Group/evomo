import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp, lexsort


class GWASFGA(Algorithm):
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

        # Weight Vectors (Bug #13)
        v, _ = uniform_sampling(pop_size, n_objs)
        self.vectors = Mutable(v.to(device))

        # Reference Points
        self.utop = Mutable(torch.zeros(n_objs, device=device))
        self.nadir = Mutable(torch.zeros(n_objs, device=device))

        # Metrics
        sentinel = torch.iinfo(torch.int32).max
        self.front_no = Mutable(torch.full((pop_size,), sentinel, dtype=torch.int32, device=device))
        self.crowd_dis = Mutable(torch.zeros(pop_size, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        # Initial Reference Points
        self.utop = torch.min(self.fit, dim=0)[0] - 0.01
        self.nadir = torch.max(self.fit, dim=0)[0] + 0.01

        # Initial Sorting
        f_no, c_dis = _gwasf_sort(self.fit, self.vectors, self.utop, self.nadir, self.pop_size)
        self.front_no = f_no
        self.crowd_dis = c_dis

    def step(self) -> None:
        # 1. Mating (Selection Pressure Bug #27)
        # Primary key: front_no (min), Secondary key: -crowd_dis (max)
        mating_pool = tournament_selection_multifit(self.pop_size, [self.front_no.float(), -self.crowd_dis], tournament_size=2)

        # Variation
        crossovered = simulated_binary(self.pop[mating_pool])
        offspring = polynomial_mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(offspring)

        # 3. Environmental Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        # Update Reference Points
        self.utop = torch.min(combined_fit, dim=0)[0] - 0.01
        self.nadir = torch.max(combined_fit, dim=0)[0] + 0.01

        # Core GWASF Sorting
        f_no, c_dis = _gwasf_sort(combined_fit, self.vectors, self.utop, self.nadir, self.pop_size)

        # Truncation using Lexsort (Bug #25: Primary key last)
        # We want to minimize front_no and maximize crowd_dis
        idx = lexsort(torch.stack([-c_dis, f_no.float()]))
        survivor_idx = idx[: self.pop_size]

        self.pop = combined_pop[survivor_idx]
        self.fit = combined_fit[survivor_idx]
        self.front_no = f_no[survivor_idx]
        self.crowd_dis = c_dis[survivor_idx]


def _gwasf_sort(
    fit: torch.Tensor, vectors: torch.Tensor, utop: torch.Tensor, nadir: torch.Tensor, N: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_combined = fit.shape[0]
    device = fit.device
    sentinel = torch.iinfo(torch.int32).max

    front_no = torch.full((num_combined,), sentinel, dtype=torch.int32, device=device)
    crowd_dis = torch.zeros(num_combined, device=device)

    # Precompute ASF matrices (3D Broadcasting: (2N, 1, M) - (1, L, M))
    # Utopian ASF
    diff_u = fit.unsqueeze(1) - utop.unsqueeze(0).unsqueeze(0)
    asf_u_mat = torch.max(diff_u * vectors.unsqueeze(0), dim=-1)[0] + 0.0001 * torch.sum(diff_u * vectors.unsqueeze(0), dim=-1)

    # Nadir ASF
    diff_n = nadir.unsqueeze(0).unsqueeze(0) - fit.unsqueeze(1)
    asf_n_mat = torch.max(diff_n * vectors.unsqueeze(0), dim=-1)[0] + 0.0001 * torch.sum(diff_n * vectors.unsqueeze(0), dim=-1)

    selected_mask = torch.zeros(num_combined, dtype=torch.bool, device=device)
    current_front = 1
    count = 0

    # Peeling Loop (Bug #9, #41)
    for _ in range(num_combined):
        if count >= N:
            break

        # Select ASF matrix based on parity
        asf_mat = torch.where(torch.tensor(current_front % 2 != 0, device=device), asf_u_mat, asf_n_mat)

        # Mask already selected individuals
        masked_asf = torch.where(selected_mask.unsqueeze(1), torch.tensor(float("inf"), device=device), asf_mat)

        # For each vector, find best individual
        best_indices = torch.argmin(masked_asf, dim=0)  # (L,)

        # Deadlock Breaker (Bug #9): If all remaining are inf, this front is empty
        valid_vector_mask = torch.any(masked_asf < float("inf"), dim=0)

        if not torch.any(valid_vector_mask):
            # Force select all remaining individuals into the current front
            remaining_mask = ~selected_mask
            front_no = torch.where(remaining_mask, torch.tensor(current_front, dtype=torch.int32, device=device), front_no)
            # Crowding distance for remaining
            crowd_dis += crowding_distance(fit, remaining_mask)
            count = num_combined
        else:
            # Extract unique indices from the vectors that pointed to valid individuals
            current_front_indices = best_indices[valid_vector_mask]
            # Use a mask to mark them in this front
            in_front_mask = torch.zeros(num_combined, dtype=torch.bool, device=device)
            in_front_mask[current_front_indices] = True
            # Ensure we don't re-select
            in_front_mask = in_front_mask & (~selected_mask)

            num_in_front = torch.sum(in_front_mask.int())

            front_no = torch.where(in_front_mask, torch.tensor(current_front, dtype=torch.int32, device=device), front_no)
            # Calculate Crowding Distance immediately (Bug #9, #21)
            crowd_dis += crowding_distance(fit, in_front_mask)

            selected_mask = selected_mask | in_front_mask
            count += num_in_front
            current_front += 1

    return front_no, crowd_dis


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = GWASFGA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
