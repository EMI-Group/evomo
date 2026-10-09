import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance
from evox.utils import clamp, lexsort

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    rank_with_constraints,
    take_violation,
)
from evomo.utils import parse_evaluate, register_lazy_buffer


class SIBEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the SIBEA population and optimization state.

        :param pop_size: Required. Requested number of candidate solutions. Use a positive integer; algorithm-specific
            minimums and reference-vector sampling are described below.
        :type pop_size: int
        :param n_objs: Required. Number of objectives, matching the second dimension of the problem's objective tensor.
            Objectives are minimized; this library targets two or more objectives.
        :type n_objs: int
        :param lb: Required. Lower decision bounds of shape ``(D,)``, where ``D`` is the number of decision variables.
            Use floating-point bounds with the same shape, dtype and device as ``ub``, and ``lb <= ub`` elementwise.
        :type lb: torch.Tensor
        :param ub: Required. Upper decision bounds of shape ``(D,)``. The decision dimension is inferred from the
            bounds, rather than passed separately. Match ``lb`` in shape, dtype and device.
        :type ub: torch.Tensor
        :param kwargs: Default: ``{}``. Extra keyword arguments are accepted for constructor compatibility but are not
            read by this implementation. In particular, passing ``device=...`` here does not move tensors; place both
            bounds on the intended device before construction.
        :type kwargs: dict

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            accepts an objective tensor or a ``(fitness, constraint_violation)`` tuple.
            Constrained evaluations retain violations through selection and state updates.

            Tensor allocation uses ``lb.device``. There is no explicit device parameter; prepare both bounds on the intended
            device.
        """
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)  # [N,D]
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))  # [N,M]

        register_lazy_buffer(self, "cv", device_like="pop")

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))

    def _cal_hv_contribution(self, fit: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Monte Carlo HV Contribution (Bug #34 & #41 Compliance)
        """
        num_samples = 1000000
        L, M = fit.shape
        device = fit.device

        # Deterministic seed for JIT compliance
        generator = torch.Generator(device=device).manual_seed(42)

        min_val = torch.min(fit, dim=0).values
        # Scale samples to the bounding box [min_val, ref]
        samples = torch.rand((num_samples, M), device=device, generator=generator) * (ref - min_val + 1e-6) + min_val

        # Dominance Check: (S, 1, M) <= (1, L, M) -> (S, L)
        # A sample is dominated by individual i if all objectives of i are <= sample
        # Using broadcasting to avoid loops
        is_dominated = (fit.unsqueeze(0) <= samples.unsqueeze(1)).all(dim=-1)

        # Exclusive Contribution: Sample is dominated by exactly one individual
        dom_counts = is_dominated.sum(dim=1)
        exclusive_mask = (dom_counts == 1).unsqueeze(1) & is_dominated
        count_per_ind = exclusive_mask.sum(dim=0).float()

        return count_per_ind

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        N = self.pop_size
        device = self.lb.device

        # 1. Mating / Variation
        # SIBEA uses random selection for mating pool (MATLAB: randi)
        mating_pool_idx = torch.randint(0, N, (N,), device=device)
        parents = self.pop[mating_pool_idx]

        # GA Variation (Bug #2: (N+1)//2 pairing handled inside SBX)
        off_pop = simulated_binary(parents, pro_c=1.0, dis_c=20.0)
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)

        # 2. Evaluation
        off_fit, off_cv = parse_evaluate(self.evaluate(off_pop))

        # 3. Merge
        merged_pop = torch.cat([self.pop, off_pop], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)

        # 4. Environmental Selection
        merged_cv = cat_violation(self.cv, off_cv)
        front_no = rank_with_constraints(merged_fit, merged_cv)

        # Identify fronts to keep
        # We use a safe way to find MaxFNo without Python loops
        max_rank = int(torch.max(front_no))
        ranks = torch.arange(1, max_rank + 1, device=device) if merged_cv is None else torch.arange(max_rank + 1, device=device)
        # Count individuals in each front
        counts = (front_no.unsqueeze(1) == ranks).sum(dim=0)
        cumulative_counts = torch.cumsum(counts, dim=0)

        # Find MaxFNo (the first rank where cumulative count >= N)
        # Use torch.where to find indices where cumulative_counts >= N
        ge_n = torch.where(cumulative_counts >= N, torch.ones_like(cumulative_counts), torch.zeros_like(cumulative_counts))
        max_f_no_idx = torch.argmax(ge_n)  # Returns first index of 1
        max_f_no = ranks[max_f_no_idx]

        mask_keep = front_no < max_f_no
        mask_last = front_no == max_f_no

        num_kept = mask_keep.sum()
        num_needed = N - num_kept

        # Metrics for Last Front
        last_fit = merged_fit[mask_last]
        # Reference point for HV (Bug #10)
        ref = torch.max(merged_fit, dim=0).values + 0.1

        hv_contrib = self._cal_hv_contribution(last_fit, ref)
        # Crowding distance as tie-breaker (Bug #34)
        # crowding_distance expects the full merged_fit and a mask
        cd_all = crowding_distance(merged_fit, mask_last)
        cd = cd_all[mask_last]

        # Lexsort (Bug #25): Primary key (HV) must be last in the stack
        # We want to maximize HV and CD, so we sort by -CD then -HV
        # lexsort sorts in ascending order, so smaller values come first.
        # To get "descending" for HV and CD, we negate them.
        sort_idx = lexsort(torch.stack([-cd, -hv_contrib]))

        # Select survivors
        last_front_indices = torch.where(mask_last)[0]
        selected_from_last = last_front_indices[sort_idx[:num_needed]]

        keep_indices = torch.where(mask_keep)[0]
        final_indices = torch.cat([keep_indices, selected_from_last])

        self.pop = merged_pop[final_indices]
        self.fit = merged_fit[final_indices]
        self.cv = take_violation(merged_cv, final_indices)


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SIBEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
