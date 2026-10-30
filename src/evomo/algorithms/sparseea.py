import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary_half
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort, randint

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    rank_with_constraint_objective,
    rank_with_constraints,
    take_violation,
    total_violation,
)
from evomo.operators.selection.distance_truncation import crowding_distance_by_rank
from evomo.utils import parse_evaluate, register_lazy_buffer, unique_rows_sorted


def _evaluate_sparse(algorithm, decisions):
    result = parse_evaluate(algorithm.evaluate(decisions))
    algorithm.fe = algorithm.fe + len(decisions)
    return result


def _sparse_offspring_count(algorithm, half=False):
    count = algorithm.pop_size // 2 if half else algorithm.pop_size
    return count if algorithm.max_fe is None else min(count, max(0, algorithm.max_fe - int(algorithm.fe)))


def _sparse_tournament_variable(eligible, fitness, remove=False):
    active = eligible.any(1)
    weights = torch.where(active[:, None], eligible, True).to(fitness.dtype)
    first, second = torch.multinomial(weights, 2, replacement=True).unbind(1)
    score = -fitness if remove else fitness
    selected = torch.where(score[first] < score[second], first, second)
    return selected, active


def _sparse_mask_variation(first, second, fitness, groups=None):
    count, dim = first.shape
    remove = torch.rand(count, device=first.device) < 0.5
    eligible = torch.where(remove[:, None], first & ~second, ~first & second)
    a, active = _sparse_tournament_variable(eligible & remove[:, None], fitness, remove=True)
    b, adding = _sparse_tournament_variable(eligible & ~remove[:, None], fitness)
    indices = torch.where(remove, a, b)
    active = active | adding
    mask = first.clone()
    old = mask.gather(1, indices[:, None]).squeeze(1)
    mask.scatter_(1, indices[:, None], torch.where(active, ~remove, old)[:, None])
    remove = torch.rand(count, device=first.device) < 0.5
    eligible = torch.where(remove[:, None], mask, ~mask)
    if groups is not None:
        eligible = eligible & groups
    a, active = _sparse_tournament_variable(eligible & remove[:, None], fitness, remove=True)
    b, adding = _sparse_tournament_variable(eligible & ~remove[:, None], fitness)
    indices = torch.where(remove, a, b)
    old = mask.gather(1, indices[:, None]).squeeze(1)
    mask.scatter_(1, indices[:, None], torch.where(active | adding, ~remove, old)[:, None])
    return mask


def _select_sparse(pop, mask, fit, cv, count):
    key = fit if cv is None else torch.cat([fit, total_violation(cv)[:, None]], 1)
    _, indices = unique_rows_sorted(key)
    # Fixed-size EvoX state retains duplicate slots if the reference would shrink below N.
    if len(indices) >= count:
        pop, mask, fit, cv = pop[indices], mask[indices], fit[indices], take_violation(cv, indices)
    rank = rank_with_constraints(fit, cv)
    distance = crowding_distance_by_rank(fit, rank)
    selected = lexsort([-distance, rank])[:count].sort().values
    return pop[selected], mask[selected], fit[selected], take_violation(cv, selected), rank[selected], distance[selected]


def _initialize_sparse(algorithm, repeats=5):
    n, d = algorithm.pop_size, algorithm.dim
    all_pop, all_mask, all_fit, all_cv = [], [], [], []
    fitness = algorithm.pop.new_zeros(d)
    for _ in range(repeats):
        decisions = algorithm.lb + torch.rand(d, d, dtype=algorithm.pop.dtype, device=algorithm.pop.device) * (
            algorithm.ub - algorithm.lb
        )
        mask = torch.eye(d, dtype=torch.bool, device=algorithm.pop.device)
        fit, cv = _evaluate_sparse(algorithm, decisions * mask)
        fitness = fitness + rank_with_constraint_objective(fit, cv).to(fitness.dtype)
        all_pop.append(decisions)
        all_mask.append(mask)
        all_fit.append(fit)
        all_cv.append(cv)
    algorithm.var_fit = fitness
    decisions = algorithm.lb + torch.rand(n, d, dtype=algorithm.pop.dtype, device=algorithm.pop.device) * (
        algorithm.ub - algorithm.lb
    )
    count = torch.ceil(torch.rand(n, 1, device=algorithm.pop.device) * d).long()
    first = torch.randint(d, (n, d), device=algorithm.pop.device)
    second = torch.randint(d, (n, d), device=algorithm.pop.device)
    winners = torch.where(fitness[first] < fitness[second], first, second)
    active = torch.arange(d, device=algorithm.pop.device)[None] < count
    mask = torch.zeros_like(winners).scatter_add(1, winners, active.long()) > 0
    fit, cv = _evaluate_sparse(algorithm, decisions * mask)
    all_pop.append(decisions)
    all_mask.append(mask)
    all_fit.append(fit)
    all_cv.append(cv)
    merged_cv = None if cv is None else torch.cat(all_cv)
    algorithm.pop, algorithm.mask, algorithm.fit, algorithm.cv, algorithm.rank, algorithm.dis = _select_sparse(
        torch.cat(all_pop), torch.cat(all_mask), torch.cat(all_fit), merged_cv, n
    )


class SparseEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the SparseEA population and optimization state.

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

        register_lazy_buffer(self, "cv", device_like="pop")
        self.max_fe = kwargs.get("max_fe")
        self.fe = Mutable(torch.tensor(0, dtype=torch.int64, device=device))

    def init_step(self) -> None:
        """Initialize from the random variable probes and sparse population together."""
        _initialize_sparse(self)

    def _sparse_tournament_selection(self, var_fit: torch.Tensor, n_select: int) -> torch.Tensor:
        # Performs tournament selection on dimension indices 1..D based on var_fit
        # Lower var_fit (better rank) wins.
        idx1 = randint(0, self.dim, (n_select,), device=var_fit.device)
        idx2 = randint(0, self.dim, (n_select,), device=var_fit.device)
        return torch.where(var_fit[idx1] < var_fit[idx2], idx1, idx2)

    def _environmental_selection(self, pop, fit, mask, n, cv=None):
        pop, mask, fit, self.cv, rank, distance = _select_sparse(pop, mask, fit, cv, n)
        return pop, fit, rank, distance, mask

    def step(self) -> None:
        n = _sparse_offspring_count(self)
        if n == 0:
            return
        mating = tournament_selection_multifit(2 * n, [-self.dis, self.rank.float()], tournament_size=2)
        parents = self.pop[mating]
        masks = self.mask[mating]
        decisions = simulated_binary_half(parents, pro_c=1.0, dis_c=20.0)
        decisions = clamp(polynomial_mutation(decisions, self.lb, self.ub), self.lb, self.ub)
        mask = _sparse_mask_variation(masks[:n], masks[n:], self.var_fit)
        fit, cv = _evaluate_sparse(self, decisions * mask)
        self.pop, self.mask, self.fit, self.cv, self.rank, self.dis = _select_sparse(
            torch.cat([self.pop, decisions]),
            torch.cat([self.mask, mask]),
            torch.cat([self.fit, fit]),
            cat_violation(self.cv, cv),
            self.pop_size,
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
