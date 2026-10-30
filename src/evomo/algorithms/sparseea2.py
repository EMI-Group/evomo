import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary_half
from evox.operators.selection import tournament_selection_multifit

from evomo.operators.selection.constraint_handling import (
    cat_violation,
)
from evomo.utils import register_lazy_buffer

from .sparseea import (
    _evaluate_sparse,
    _initialize_sparse,
    _select_sparse,
    _sparse_mask_variation,
    _sparse_offspring_count,
)


def _grouped_real_mutation(decisions, lb, ub, groups=4):
    count, dim = decisions.shape
    order = decisions.argsort(dim=1, stable=True)
    ranks = torch.empty_like(order).scatter(1, order, torch.arange(dim, device=decisions.device).expand_as(order))
    width = dim // groups
    labels = (ranks // width).clamp_max(groups - 1) if width else torch.full_like(ranks, groups - 1)
    chosen = torch.randint(groups, (count, 1), device=decisions.device)
    selected = labels == chosen
    x = torch.maximum(torch.minimum(decisions, ub), lb)
    span = (ub - lb).clamp_min(torch.finfo(x.dtype).tiny)
    mu = torch.rand(count, 1, dtype=x.dtype, device=x.device)
    lower, upper = (x - lb) / span, (ub - x) / span
    left = (2 * mu + (1 - 2 * mu) * (1 - lower).pow(21)).pow(1 / 21) - 1
    right = 1 - (2 * (1 - mu) + 2 * (mu - 0.5) * (1 - upper).pow(21)).pow(1 / 21)
    delta = torch.where(mu <= 0.5, left, right)
    return torch.where(selected & (ub > lb), x + delta * (ub - lb), x), selected


class SparseEA2(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the SparseEA2 population and optimization state.

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

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.mask = Mutable(torch.zeros((pop_size, self.dim), dtype=torch.bool, device=device))
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.var_fit = Mutable(torch.zeros(self.dim, device=device))
        self.rank = Mutable(torch.zeros(pop_size, device=device))
        self.dis = Mutable(torch.zeros(pop_size, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        self.max_fe = kwargs.get("max_fe")
        self.fe = Mutable(torch.tensor(0, dtype=torch.int64, device=device))

    def init_step(self) -> None:
        """Initialize from the random variable probes and sparse population together."""
        _initialize_sparse(self)

    def _sparse_tournament(self, indices: torch.Tensor, var_fit: torch.Tensor) -> torch.Tensor:
        # Performs tournament selection on the var_fit values
        N, D = indices.shape
        rand_indices = torch.randint(0, self.dim, (N, D), device=indices.device)
        fit_a = var_fit[indices]
        fit_b = var_fit[rand_indices]
        return torch.where(fit_a < fit_b, indices, rand_indices)

    def _grouped_mutation(self, pop, mask):
        decisions, groups = _grouped_real_mutation(pop, self.lb, self.ub)
        return decisions, _sparse_mask_variation(mask, mask, self.var_fit, groups)

    def step(self) -> None:
        n = _sparse_offspring_count(self)
        if n == 0:
            return
        mating = tournament_selection_multifit(2 * n, [-self.dis, self.rank.float()], tournament_size=2)
        parents = self.pop[mating]
        masks = self.mask[mating]
        decisions = simulated_binary_half(parents, pro_c=1.0, dis_c=20.0)
        decisions, groups = _grouped_real_mutation(decisions, self.lb, self.ub)
        mask = _sparse_mask_variation(masks[:n], masks[n:], self.var_fit, groups)
        fit, cv = _evaluate_sparse(self, decisions * mask)
        self.pop, self.mask, self.fit, self.cv, self.rank, self.dis = _select_sparse(
            torch.cat([self.pop, decisions]),
            torch.cat([self.mask, mask]),
            torch.cat([self.fit, fit]),
            cat_violation(self.cv, cv),
            self.pop_size,
        )


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SparseEA2(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
