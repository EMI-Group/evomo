import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    rank_with_constraints,
    take_violation,
)
from evomo.operators.selection.distance_truncation import crowding_distance_by_rank
from evomo.utils import parse_evaluate, register_lazy_buffer


class SNSGA2(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        sLower: float = 0.75,
        sUpper: float = 1.0,
        **kwargs,
    ):
        """Initialize the SNSGA2 population and optimization state.

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
        :param sLower: Default: ``0.75``. Lower endpoint of the initial fraction of zero-valued decision components. Use
            ``0 <= sLower <= sUpper <= 1``. The active count is ``round((1 - s) * D)`` as ``s`` varies across the initial
            population.
        :type sLower: float
        :param sUpper: Default: ``1.0``. Upper endpoint of the initial fraction of zero-valued decision components.
            Larger values initialize sparser solutions; this is an initialization setting, not a guaranteed sparsity
            level for subsequent generations.
        :type sUpper: float
        :param kwargs: Optional ``max_fe`` bounds actual evaluations, including probes and initialization.
            Place bounds on the intended device; ``device=...`` does not move them.
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
        self.dim = lb.numel()
        self.sLower = sLower
        self.sUpper = sUpper

        # Initialize State (Mutables)
        # Initialization (VSSPS Logic)
        rho = torch.linspace(1 - sLower, 1 - sUpper, pop_size, device=device)
        widths = (rho * self.dim).round().clamp_max(int((1 - sLower) * self.dim)).long()
        mask = torch.zeros(pop_size, self.dim, dtype=torch.bool, device=device)
        cycles = []
        current, width = [], 0
        for i in range(pop_size):
            value = int(widths[i])
            if current and width + value > self.dim:
                cycles.append(current)
                current, width = [], 0
            current.append(i)
            width += value
        if current:
            cycles.append(current)
        for c, cycle in enumerate(cycles):
            gap = self.dim - sum(int(widths[i]) for i in cycle)
            gap_size = (gap + len(cycle) - 1) // len(cycle)
            position = 0
            for i in cycle:
                extra = gap_size if gap > 0 else 0
                gap -= extra
                end = position + int(widths[i]) + (extra if c < len(cycles) - 1 else 0)
                mask[i, position : min(end, self.dim)] = True
                position += int(widths[i]) + extra

        initial_pop = (self.lb + torch.rand((pop_size, self.dim), device=device) * (self.ub - self.lb)) * mask
        self.pop = Mutable(initial_pop)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        sentinel = torch.iinfo(torch.int32).max
        self.front_no = Mutable(torch.full((pop_size,), sentinel, dtype=torch.int32, device=device))
        self.crowd_dis = Mutable(torch.full((pop_size,), -1.0, dtype=torch.float32, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        self.max_fe = kwargs.get("max_fe")
        self.fe = Mutable(torch.tensor(0, dtype=torch.int64, device=device))

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        self.fe = self.fe + len(self.pop)
        # Initial Environmental Selection
        self.pop, self.fit, self.front_no, self.crowd_dis = self._environmental_selection(self.pop, self.fit, self.cv)

    def _sparse_sbx(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        # SSBX Logic
        mask1, mask2 = (p1 != 0), (p2 != 0)
        matching = mask1 == mask2

        # Standard SBX on matching
        off = simulated_binary(torch.cat([p1, p2], dim=0), pro_c=1.0, dis_c=20.0)
        half = p1.shape[0]
        off1, off2 = off[:half], off[half:]

        # Swap on non-matching with 0.5 probability
        different = ~matching
        scores = torch.where(different, torch.rand_like(p1), torch.inf).flatten()
        order = scores.argsort(stable=True)
        positions = torch.empty_like(order).scatter(0, order, torch.arange(order.numel(), device=p1.device))
        count = different.sum()
        swaps = count - torch.round(count * torch.rand((), device=p1.device)).long()
        swap_mask = different & (positions.reshape_as(p1) < swaps)
        res1 = torch.where(swap_mask, p2, p1)
        res2 = torch.where(swap_mask, p1, p2)

        # Combine matching SBX and non-matching swaps
        final1 = torch.where(matching, off1, res1)
        final2 = torch.where(matching, off2, res2)
        return torch.cat([final1, final2], dim=0)

    def _sparse_mutate_mask(self, pop: torch.Tensor, target_W: torch.Tensor) -> torch.Tensor:
        N, D = pop.shape
        curr_W = (pop != 0).sum(1)
        diff = target_W - curr_W

        to_add = diff.clamp(min=0)
        to_rem = (-diff).clamp(min=0)

        add_noise = torch.where(pop == 0, torch.rand((N, D), device=pop.device), -torch.inf)
        rem_noise = torch.where(pop != 0, torch.rand((N, D), device=pop.device), -torch.inf)

        add_rank = torch.argsort(add_noise, dim=1, descending=True)
        rem_rank = torch.argsort(rem_noise, dim=1, descending=True)

        col_idx = torch.arange(D, device=pop.device).unsqueeze(0).expand(N, D)
        add_mask = col_idx < to_add.unsqueeze(1)
        rem_mask = col_idx < to_rem.unsqueeze(1)

        final_add = torch.zeros_like(pop, dtype=torch.bool).scatter_(1, add_rank, add_mask)
        final_rem = torch.zeros_like(pop, dtype=torch.bool).scatter_(1, rem_rank, rem_mask)

        new_pop = pop.clone()
        vals = self.lb + torch.rand((N, D), device=pop.device) * (self.ub - self.lb)
        new_pop = torch.where(final_add, vals, new_pop)
        new_pop = torch.where(final_rem, torch.zeros_like(new_pop), new_pop)

        return new_pop

    def _environmental_selection(self, merge_pop, merge_fit, merge_cv=None):
        rank = rank_with_constraints(merge_fit, merge_cv)
        distance = crowding_distance_by_rank(merge_fit, rank)
        indices = lexsort([-distance, rank])[: self.pop_size].sort().values
        self.cv = take_violation(merge_cv, indices)
        return merge_pop[indices], merge_fit[indices], rank[indices], distance[indices]

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        count = 2 * (self.pop_size // 2)
        if self.max_fe is not None:
            count = min(count, max(0, self.max_fe - int(self.fe)))
        if not count:
            return
        parent_count = self.pop_size if self.max_fe is None or count == 2 * (self.pop_size // 2) else 2 * ((count + 1) // 2)
        mating_pool_idx = tournament_selection_multifit(
            parent_count, [-self.crowd_dis, self.front_no.float()], tournament_size=2
        )
        parents = self.pop[mating_pool_idx]

        half = parents.shape[0] // 2
        off_pop = self._sparse_sbx(parents[:half], parents[half : 2 * half])[:count]

        off_pop = torch.where(off_pop != 0, polynomial_mutation(off_pop, self.lb, self.ub), off_pop)

        sparsity = (off_pop == 0).to(off_pop.dtype).mean(1, keepdim=True)
        mutated = polynomial_mutation(sparsity, torch.zeros_like(sparsity[0]), torch.ones_like(sparsity[0]))
        gate = torch.rand(len(off_pop), 1, device=off_pop.device) < 1 / self.dim
        target_sparsity = torch.where(gate, mutated, sparsity).clamp(0, 1)
        target_W = ((1 - target_sparsity.squeeze(1)) * self.dim).round().clamp(0, self.dim).long()
        off_pop = self._sparse_mutate_mask(off_pop, target_W)

        off_pop = clamp(off_pop, self.lb, self.ub)
        off_fit, off_cv = parse_evaluate(self.evaluate(off_pop))
        self.fe = self.fe + len(off_pop)

        merge_pop = torch.cat([self.pop, off_pop], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit, self.front_no, self.crowd_dis = self._environmental_selection(
            merge_pop, merge_fit, cat_violation(self.cv, off_cv)
        )


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SNSGA2(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
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
