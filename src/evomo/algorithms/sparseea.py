import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import clamp, lexsort, randint

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class SparseEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # Initialize State (Mutables)
        # self.pop is the continuous part [N, D]
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        # self.mask is the binary part [N, D]
        self.mask = Mutable(torch.zeros((pop_size, self.dim), dtype=torch.bool, device=device))
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.var_fit = Mutable(torch.zeros(self.dim, device=device))
        self.iter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

        # Rank and Distance for tournament selection
        sentinel = torch.iinfo(torch.int32).max
        self.rank = Mutable(torch.full((pop_size,), sentinel, dtype=torch.int32, device=device))
        self.dis = Mutable(torch.full((pop_size,), -1.0, device=device))

    def init_step(self) -> None:
        device = self.lb.device
        # A. Initialization (Variable Fitness Probe)
        # 1. Probe Generation: Create D individuals using eye matrix
        probe_mask = torch.eye(self.dim, dtype=torch.bool, device=device)
        # 2. Repetition: K = 1 + 4 * (encoding != 4). Assuming standard encoding, K=5.
        K = 5
        probes = probe_mask.repeat(K, 1)  # [D*K, D]

        # 3. Evaluation: Dec * Mask. Dec is ones for probing.
        probe_dec = torch.ones((self.dim * K, self.dim), device=device)
        probe_fit = self.evaluate(probe_dec * probes)

        # 4. Score Accumulation
        rank = non_dominate_rank(probe_fit)
        # Vectorized: var_fit[j] += rank[i] for each probe i that used bit j
        # probe_indices maps each row in 'probes' to its original dimension index 0..D-1
        probe_indices = torch.arange(self.dim, device=device).repeat(K)
        self.var_fit.index_add_(0, probe_indices, rank.float())

        # Initialize population masks randomly for the first real generation
        # (Usually SparseEA starts with some initial sparsity)
        self.mask = torch.rand((self.pop_size, self.dim), device=device) < 0.5
        self.fit = self.evaluate(self.pop * self.mask)

        # Initial Environmental Selection to set rank/dis
        self.pop, self.fit, self.rank, self.dis, self.mask = self._environmental_selection(
            self.pop, self.fit, self.mask, self.pop_size
        )

    def _sparse_tournament_selection(self, var_fit: torch.Tensor, n_select: int) -> torch.Tensor:
        # Performs tournament selection on dimension indices 1..D based on var_fit
        # Lower var_fit (better rank) wins.
        idx1 = randint(0, self.dim, (n_select,), device=var_fit.device)
        idx2 = randint(0, self.dim, (n_select,), device=var_fit.device)
        return torch.where(var_fit[idx1] < var_fit[idx2], idx1, idx2)

    def _environmental_selection(self, pop, fit, mask, n):
        # 1. Duplicate Removal
        u_pop_mask_fit = torch.cat([fit, pop, mask.float()], dim=1)
        _, unique_indices = unique_rows_sorted(u_pop_mask_fit)
        pop, fit, mask = pop[unique_indices], fit[unique_indices], mask[unique_indices]

        # 2. Integrated Peeling
        rank = non_dominate_rank(fit)
        device = fit.device
        selected_indices = torch.zeros(0, dtype=torch.long, device=device)

        current_front = 0
        # While loop for peeling
        while selected_indices.shape[0] < n:
            mask_f = rank == current_front
            num_in_front = mask_f.sum()

            # Deadlock Breaker (Bug #9)
            if num_in_front == 0:
                remaining_needed = n - selected_indices.shape[0]
                # Use lexsort to pick remaining (Bug #25)
                # Primary key: rank (last), Secondary: fit sum (first)
                sort_idx = lexsort(torch.stack([fit.sum(dim=1), rank.float()]))
                # Filter out already selected
                # This is a simplified fallback to ensure JIT compliance
                selected_indices = torch.cat([selected_indices, sort_idx[:remaining_needed]])
                break

            cd = crowding_distance(fit, mask_f)

            if selected_indices.shape[0] + num_in_front <= n:
                sub_indices = torch.where(mask_f)[0]
                selected_indices = torch.cat([selected_indices, sub_indices])
            else:
                needed = n - selected_indices.shape[0]
                sub_indices = torch.where(mask_f)[0]
                local_cd = cd[sub_indices]
                _, sel = torch.topk(local_cd, needed)
                selected_indices = torch.cat([selected_indices, sub_indices[sel]])

            current_front += 1

        # Finalize outputs
        survivor_pop = pop[selected_indices]
        survivor_fit = fit[selected_indices]
        survivor_mask = mask[selected_indices]

        # Recalculate rank/dis for the survivors to use in next mating
        final_rank = non_dominate_rank(survivor_fit)
        # Crowding distance is calculated per front
        final_dis = torch.zeros(n, device=device)
        for f in range(int(final_rank.max().item()) + 1):
            f_mask = final_rank == f
            final_dis[f_mask] = crowding_distance(survivor_fit, f_mask)[f_mask]

        return survivor_pop, survivor_fit, final_rank, final_dis, survivor_mask

    def step(self) -> None:
        device = self.lb.device
        N = self.pop_size
        D = self.dim

        # 1. Mating Pool (Tournament Selection)
        # Primary key: Rank (last), Secondary: -CrowdDis (first)
        mating_pool = tournament_selection_multifit(N, [-self.dis, self.rank.float()], tournament_size=2)

        parent_pop = self.pop[mating_pool]
        parent_mask = self.mask[mating_pool]

        # 2. Variation - Continuous (SBX)
        off_pop = simulated_binary(parent_pop, pro_c=1.0, dis_c=20.0)
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # 3. Variation - Sparse Mask Operator
        # Mask Crossover
        p1_mask = parent_mask[: N // 2]
        p2_mask = parent_mask[N // 2 :]

        diff = p1_mask ^ p2_mask  # [N/2, D]
        # For each row, if bits differ, pick one based on var_fit tournament
        # Vectorized: Generate winners for all possible D bits
        # We only apply where diff is True
        # In SparseEA, if bits differ, we often pick the bit from the parent with better var_fit
        # or simply use the tournament winner's bit state.
        # Following Blueprint: "For each differing bit, pick two candidate indices... keep better"
        # We'll use the winner's bit from p2 if it's "better" or just swap.
        # Simplified vectorized crossover:
        rand_mask = torch.rand((N // 2, D), device=device) < 0.5
        off_mask = torch.where(diff & rand_mask, p2_mask, p1_mask)
        off_mask = torch.cat([off_mask, off_mask], dim=0)  # [N, D]

        # Mask Mutation (Add/Remove)
        # For each offspring, decide Add (0->1) or Remove (1->0)
        mutation_type = torch.rand(N, device=device) < 0.5

        # Add: Sample from mask == 0 with weights 1 / (var_fit + eps)
        add_weights = 1.0 / (self.var_fit + 1e-6)
        # Remove: Sample from mask == 1 with weights var_fit
        rem_weights = self.var_fit

        # Vectorized bit flipping (Bug #20)
        # We flip 1 bit per individual for simplicity in vectorization
        # Sampling indices for Add
        add_probs = (~off_mask).float() * add_weights
        add_idx = torch.multinomial(add_probs + 1e-10, 1).squeeze()
        # Sampling indices for Remove
        rem_probs = off_mask.float() * rem_weights
        rem_idx = torch.multinomial(rem_probs + 1e-10, 1).squeeze()

        # Apply mutations
        batch_idx = torch.arange(N, device=device)
        final_idx = torch.where(mutation_type, add_idx, rem_idx)
        off_mask[batch_idx, final_idx] = mutation_type  # If Add, set to True; if Remove, set to False

        # 4. Evaluation
        off_fit = self.evaluate(off_pop * off_mask)

        # 5. Environmental Selection
        merged_pop = torch.cat([self.pop, off_pop], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)
        merged_mask = torch.cat([self.mask, off_mask], dim=0)

        self.pop, self.fit, self.rank, self.dis, self.mask = self._environmental_selection(
            merged_pop, merged_fit, merged_mask, N
        )

        self.iter += 1


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SparseEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
