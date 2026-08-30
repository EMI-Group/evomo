import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class eMOEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, epsilon: float = 0.05, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()

        # Algorithm Parameters
        self.epsilon = Parameter(torch.tensor(epsilon, device=device))

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Archive is dynamic, but we initialize with a fixed max capacity for JIT-friendly masking
        self.archive_pop = Mutable(torch.zeros(pop_size * 2, self.dim, device=device))
        self.archive_fit = Mutable(torch.full((pop_size * 2, n_objs), torch.inf, device=device))
        self.archive_mask = Mutable(torch.zeros(pop_size * 2, dtype=torch.bool, device=device))

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)

        # Grid Calculation for Archive Seed
        f_min = torch.min(self.fit, dim=0)[0]
        pop_grid = torch.floor((self.fit - f_min) / (self.epsilon + 1e-6))

        # Only first front based on grid values
        rank = non_dominate_rank(pop_grid)
        first_front_mask = rank == 1

        # Update Archive
        num_seeds = torch.sum(first_front_mask.int())
        indices = torch.arange(self.archive_pop.shape[0], device=self.pop.device)
        fill_mask = indices < num_seeds

        self.archive_pop[fill_mask] = self.pop[first_front_mask]
        self.archive_fit[fill_mask] = self.fit[first_front_mask]
        self.archive_mask[fill_mask] = True

    def step(self) -> None:
        device = self.pop.device

        # 1. Selection & Mating
        # Manual Vectorized Tournament Selection (Binary)
        idx1 = torch.randint(0, self.pop_size, (self.pop_size,), device=device)
        idx2 = torch.randint(0, self.pop_size, (self.pop_size,), device=device)
        fit1 = self.fit[idx1]
        fit2 = self.fit[idx2]
        # Dominance check for tournament
        dom12 = (fit1 <= fit2).all(-1) & (fit1 < fit2).any(-1)
        dom21 = (fit2 <= fit1).all(-1) & (fit2 < fit1).any(-1)
        parent_idx_pop = torch.where(dom21 & ~dom12, idx2, idx1)
        parents_pop = self.pop[parent_idx_pop]

        # Archive Selection (Random)
        valid_arc_indices = torch.where(self.archive_mask)[0]
        num_arc = torch.tensor(valid_arc_indices.shape[0], device=device)
        # Ensure we don't crash if archive is empty
        safe_num_arc = torch.where(num_arc > 0, num_arc, torch.ones_like(num_arc))
        rand_idx = torch.randint(0, safe_num_arc, (self.pop_size,), device=device)
        parents_arc = self.archive_pop[valid_arc_indices[rand_idx]]

        # Variation (SBX + PM)
        interleaved_parents = torch.empty((self.pop_size * 2, self.dim), device=device)
        interleaved_parents[0::2] = parents_pop
        interleaved_parents[1::2] = parents_arc

        off_pop = simulated_binary(interleaved_parents, pro_c=1.0, dis_c=20.0)
        off_pop = off_pop[: self.pop_size]
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(off_pop)

        # 3. Population Update (Vectorized Steady-State)
        # Check if population members dominate offspring
        pop_dom_off = (self.fit.unsqueeze(1) <= off_fit.unsqueeze(0)).all(-1) & (
            self.fit.unsqueeze(1) < off_fit.unsqueeze(0)
        ).any(-1)
        is_dominated_by_pop = pop_dom_off.any(dim=0)

        # Check if offspring dominate population members
        off_dom_pop = (off_fit.unsqueeze(1) <= self.fit.unsqueeze(0)).all(-1) & (
            off_fit.unsqueeze(1) < self.fit.unsqueeze(0)
        ).any(-1)

        # For each offspring, if not dominated by pop, find a target to replace
        # To keep it vectorized and JIT-friendly, we use a priority-based replacement
        # Priority 1: A member dominated by the offspring
        # Priority 2: A random member

        # Generate random targets for all offspring as fallback
        random_targets = torch.randint(0, self.pop_size, (self.pop_size,), device=device)

        # Find first dominated member for each offspring (if any)
        # off_dom_pop is (N_off, N_pop). We want an index i for each j.
        has_dominated = off_dom_pop.any(dim=1)
        # Use argmax to find the first True index in each row
        dominated_idx = torch.argmax(off_dom_pop.float(), dim=1)

        target_indices = torch.where(has_dominated, dominated_idx, random_targets)

        # Only update if offspring is not dominated by the current population
        update_mask = ~is_dominated_by_pop

        # Apply updates. Since multiple offspring might target the same index,
        # the last one in the batch wins, which is acceptable for steady-state.
        self.pop[target_indices[update_mask]] = off_pop[update_mask]
        self.fit[target_indices[update_mask]] = off_fit[update_mask]

        # 4. Archive Update (Epsilon-Dominance)
        self.archive_pop, self.archive_fit, self.archive_mask = _update_archive(
            self.archive_pop, self.archive_fit, self.archive_mask, off_pop, off_fit, self.epsilon
        )


def _update_archive(arc_pop, arc_fit, arc_mask, off_pop, off_fit, epsilon):
    device = arc_pop.device
    valid_idx = torch.where(arc_mask)[0]
    curr_arc_pop = arc_pop[valid_idx]
    curr_arc_fit = arc_fit[valid_idx]

    total_pop = torch.cat([curr_arc_pop, off_pop], dim=0)
    total_fit = torch.cat([curr_arc_fit, off_fit], dim=0)

    f_min = torch.min(total_fit, dim=0)[0]
    G = torch.floor((total_fit - f_min) / (epsilon + 1e-6))

    # Grid Dominance
    dom = (G.unsqueeze(1) <= G.unsqueeze(0)).all(-1) & (G.unsqueeze(1) < G.unsqueeze(0)).any(-1)
    is_dominated = dom.any(dim=0)

    # Distance to Corner (Tie-breaker)
    corner = G * epsilon + f_min
    dist = torch.sqrt(torch.sum((total_fit - corner) ** 2, dim=1) + 1e-6)

    # Lexsort keys: Distance (primary), then Grid coordinates (secondary)
    keys = [dist]
    G_cols = torch.unbind(G, dim=1)
    for col in reversed(G_cols):
        keys.append(col)

    sort_idx = lexsort(torch.stack(keys))
    sorted_G = G[sort_idx]

    # Find first occurrence in sorted list (min distance in each cell)
    diff = torch.cat([torch.tensor([True], device=device), (sorted_G[1:] != sorted_G[:-1]).any(dim=1)])
    best_in_cell_idx = sort_idx[diff]

    final_keep_mask = torch.zeros(total_pop.shape[0], dtype=torch.bool, device=device)
    final_keep_mask[best_in_cell_idx] = True
    final_keep_mask = final_keep_mask & (~is_dominated)

    new_pop = total_pop[final_keep_mask]
    new_fit = total_fit[final_keep_mask]
    num_new = torch.tensor(new_pop.shape[0], device=device)

    max_cap = arc_pop.shape[0]
    num_to_copy = torch.where(num_new < max_cap, num_new, torch.tensor(max_cap, device=device))

    out_pop = torch.zeros_like(arc_pop)
    out_fit = torch.full_like(arc_fit, torch.inf)
    out_mask = torch.zeros_like(arc_mask)

    indices = torch.arange(max_cap, device=device)
    copy_mask = indices < num_to_copy
    out_pop[copy_mask] = new_pop[:num_to_copy]
    out_fit[copy_mask] = new_fit[:num_to_copy]
    out_mask[copy_mask] = True

    return out_pop, out_fit, out_mask


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = eMOEA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
