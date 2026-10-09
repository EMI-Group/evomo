import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.selection import crowding_distance
from evox.utils import clamp, lexsort, randint

from evomo.operators.selection import non_dominate_rank
from evomo.operators.selection.constraint_handling import (
    constrained_crowding_selection,
    constrained_dominates,
)
from evomo.utils import parse_evaluate, register_lazy_buffer


def _map_parent_draws(
    index: torch.Tensor,
    first_draw: torch.Tensor,
    second_draw: torch.Tensor,
    third_draw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map shrinking-range draws to three distinct population indices.

    Each input draw is uniform on a compact range with respectively one, two,
    or three indices removed.  The order-preserving remapping is a bijection,
    so the resulting ordered parent triple is uniform without replacement.
    """
    dtype = index.dtype

    r1 = first_draw + (first_draw >= index).to(dtype)

    lower = torch.minimum(index, r1)
    upper = torch.maximum(index, r1)
    r2 = second_draw
    r2 = r2 + (second_draw >= lower).to(dtype)
    r2 = r2 + (second_draw >= upper - 1).to(dtype)

    lower = torch.minimum(torch.minimum(index, r1), r2)
    upper = torch.maximum(torch.maximum(index, r1), r2)
    middle = index + r1 + r2 - lower - upper
    r3 = third_draw
    r3 = r3 + (third_draw >= lower).to(dtype)
    r3 = r3 + (third_draw >= middle - 1).to(dtype)
    r3 = r3 + (third_draw >= upper - 2).to(dtype)
    return r1, r2, r3


def _sample_parent_indices(pop_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample DE/rand/1 parents in O(pop_size) memory."""
    if pop_size < 4:
        raise ValueError("GDE3 requires pop_size >= 4 to sample three parents distinct from each target.")

    index = torch.arange(pop_size, device=device)
    first_draw = randint(0, pop_size - 1, (pop_size,), device=device)
    second_draw = randint(0, pop_size - 2, (pop_size,), device=device)
    third_draw = randint(0, pop_size - 3, (pop_size,), device=device)
    return _map_parent_draws(index, first_draw, second_draw, third_draw)


class GDE3(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, F: float = 0.5, CR: float = 0.5):
        """Initialize GDE3 with differential evolution and nondominated environmental selection.

        :param pop_size: Required. Requested number of candidate solutions. Use a positive integer; algorithm-specific
            minimums and reference-vector sampling are described below. At least four candidates are required for three
            distinct DE parents per target.
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
        :param F: Default: ``0.5``. Differential-evolution scale multiplying a difference between parent decision
            vectors. Use a positive value; larger values produce larger differential steps before bound repair.
        :type F: float
        :param CR: Default: ``0.5``. Binomial crossover probability in ``[0, 1]`` of taking a decision component from
            the mutant rather than the target. The implementation also forces a mutant component in each trial.
        :type CR: float

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            accepts an objective tensor or a ``(fitness, constraint_violation)`` tuple.
            Constrained evaluations retain violations through selection and state updates.

            Tensor allocation uses ``lb.device``. There is no explicit device parameter; prepare both bounds on the intended
            device.
        """
        super().__init__()
        if pop_size < 4:
            raise ValueError("GDE3 requires pop_size >= 4 to sample three parents distinct from each target.")
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

        register_lazy_buffer(self, "cv", device_like="pop")

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        device = self.pop.device
        N = self.pop_size

        # 1. Mating (Vectorized DE/rand/1/bin)
        # Draw an ordered triple uniformly without replacement, excluding the
        # target index.  This uses only O(N) temporary storage.
        r1, r2, r3 = _sample_parent_indices(N, device)

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
        off_fit, off_cv = parse_evaluate(self.evaluate(off_pop))

        # 3. Selection
        self.pop, self.fit = self._gde3_selection(self.pop, self.fit, off_pop, off_fit, off_cv)

    def _gde3_selection(self, pop, fit, off_pop, off_fit, off_cv=None):
        if off_cv is not None:
            off_dom = constrained_dominates(off_fit, off_cv, fit, self.cv)
            parent_dom = constrained_dominates(fit, self.cv, off_fit, off_cv)
            add = ~(off_dom | parent_dom)
            updated_pop = torch.where(off_dom[:, None], off_pop, pop)
            updated_fit = torch.where(off_dom[:, None], off_fit, fit)
            cv_shape = (-1,) + (1,) * (self.cv.ndim - 1)
            updated_cv = torch.where(off_dom.reshape(cv_shape), off_cv, self.cv)
            combined_pop = torch.cat([updated_pop, off_pop])
            combined_fit = torch.cat([updated_fit, torch.where(add[:, None], off_fit, torch.inf)])
            combined_cv = torch.cat([updated_cv, torch.where(add.reshape(cv_shape), off_cv, torch.inf)])
            indices, _, _ = constrained_crowding_selection(combined_fit, combined_cv, self.pop_size)
            self.cv = combined_cv[indices]
            return combined_pop[indices], combined_fit[indices]
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
