"""Tensorized MaOEA-HAP, adapted from BIMK/PlatEMO (GPL-3.0), 2026."""

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import lexsort

from evomo.operators.selection import non_dominate_rank
from evomo.operators.selection.constraint_handling import worst_constraint_index
from evomo.operators.selection.distance_truncation import crowding_distance_by_rank
from evomo.utils import (
    masked_select_indices,
    minmax_normalize,
    pairwise_cosine_similarity,
    parse_evaluate,
    register_lazy_buffer,
    sample_masked_indices,
    unique_row_mask,
)


def _ranks(fit, valid):
    """Use the library rank operator with fixed-shape padding excluded."""
    rank = non_dominate_rank(torch.where(valid[:, None], fit, torch.inf)).long()
    return torch.where(valid, rank, fit.shape[0])


def _indicator_select(fit, valid, capacity, cv=None):
    """PlatEMO UpdateCA/UpdateArchive, including its row-scaled deletion update."""
    normalized = minmax_normalize(fit, valid)
    indicator = (normalized[:, None, :] - normalized[None, :, :]).amax(2)
    scale = torch.where(valid[:, None], indicator.abs(), 0).amax(0).clamp_min(torch.finfo(fit.dtype).eps)
    fitness = -torch.where(valid[:, None], torch.exp((-indicator / scale[None, :] / 0.05).clamp_max(80)), 0).sum(0) + 1

    def cond(active, scores):
        return active.sum() > capacity

    def body(active, scores):
        index = worst_constraint_index(scores, active, cv).reshape(1)
        row = indicator.index_select(0, index).squeeze(0)
        scores = scores + torch.exp((-row / scale.gather(0, index) / 0.05).clamp_max(80))
        return active.scatter(0, index, False), scores

    active, _ = torch.while_loop(cond, body, (valid.clone(), fitness))
    return masked_select_indices(active, capacity)


def _ndq(fit, valid):
    """First front of PlatEMO NDQSort, including its angle-dependent ENS rule."""
    original_valid = valid
    valid = unique_row_mask(fit, valid)
    order = lexsort(torch.where(valid[:, None], fit, torch.inf).T.flip(0).unbind(0))
    values, active = fit[order], valid[order]
    angles = torch.acos(pairwise_cosine_similarity(values, values))
    angles = torch.where(torch.eye(fit.shape[0], device=fit.device, dtype=torch.bool), torch.pi / 2, angles)
    angles = torch.where(active[None, :], angles, torch.inf)
    count = min(fit.shape[1], fit.shape[0])
    smallest = angles.topk(count, largest=False).values
    weights = torch.arange(1, count + 1, device=fit.device, dtype=fit.dtype)
    association = torch.where(torch.isfinite(smallest), smallest / weights, 0).sum(1)
    positions = torch.arange(fit.shape[0], device=fit.device)

    def cond(i, front):
        return i < values.shape[0]

    def body(i, front):
        current = values.index_select(0, i.reshape(1)).squeeze(0)
        predecessors = front & (positions < i)
        dominated = (values <= current).all(1) & (association > association.gather(0, i.reshape(1)))
        if fit.shape[1] == 2:
            # ENS_SS stops after the nearest accepted predecessor for two objectives.
            last = torch.where(predecessors, positions, -1).amax()
            dominated = dominated & (positions == last)
        accept = active.gather(0, i.reshape(1)).squeeze(0) & ~(predecessors & dominated).any()
        return i + 1, front.scatter(0, i.reshape(1), accept.reshape(1))

    _, front = torch.while_loop(cond, body, (positions.new_zeros(()), torch.zeros_like(active)))
    standard = _ranks(fit, valid) == 0
    result = torch.zeros_like(valid).scatter(0, order, front)
    if fit.shape[1] >= 3:
        result = torch.where(valid.sum() >= 500, standard, result)
    # NDQSort maps unique-row ranks back to every original occurrence.
    equal = (fit[:, None] == fit[None, :]).all(2)
    return (equal & result[None, :]).any(1) & original_valid


def _shape(fit, valid, ideal, nadir):
    """Estimate Pareto-front curvature over the 0.5..2 grid with box-plot denoising."""
    front = (_ranks(fit, valid) == 0) & valid
    normalized = ((fit - ideal) / torch.where(nadir > ideal, nadir - ideal, 1)).clamp_min(0)
    powers = torch.linspace(0.5, 2, 16, device=fit.device, dtype=fit.dtype)
    lengths = normalized[None].pow(powers[:, None, None]).sum(2).pow(1 / powers[:, None])
    sorted_lengths = torch.where(front[None, :], lengths, torch.inf).sort(1).values
    count = front.sum()
    q1 = sorted_lengths.index_select(1, ((count * 0.25).long().clamp_min(1) - 1).reshape(1))
    q3 = sorted_lengths.index_select(1, ((count * 0.75).long().clamp_min(1) - 1).reshape(1))
    keep = front[None, :] & (lengths <= q3 + 1.5 * (q3 - q1))
    maximum = torch.where(keep, lengths, 0).amax(1, keepdim=True).clamp_min(torch.finfo(fit.dtype).eps)
    scaled = lengths / maximum
    number = keep.sum(1).clamp_min(1)
    mean = torch.where(keep, scaled, 0).sum(1) / number
    variance = torch.where(keep, (scaled - mean[:, None]).square(), 0).sum(1) / (number - 1).clamp_min(1)
    return powers.gather(0, variance.argmin().reshape(1)).squeeze(0)


def _distance_to_surface(fit, valid, p):
    """Compute each valid row's Lp distance to the estimated front surface."""
    ideal = torch.where(valid[:, None], fit, torch.inf).amin(0)
    nadir = torch.where(valid[:, None], fit, -torch.inf).amax(0)
    shape = _shape(fit, valid, ideal, nadir)
    shifted = (fit - ideal).clamp_min(0) + 1e-6
    projected = shifted / shifted.pow(shape).sum(1, keepdim=True).pow(1 / shape).clamp_min(1e-30)
    return (shifted - projected).abs().pow(p).sum(1).pow(1 / p)


def _diversity_select(fit, valid, capacity, p, phase):
    valid = _ndq(fit, valid)
    compact, valid = masked_select_indices(valid, fit.shape[0])
    fit = fit[compact]
    overflow = valid.sum() > capacity
    normalized = minmax_normalize(fit, valid)
    chosen = torch.zeros_like(valid)
    m = fit.shape[1]
    weights = torch.eye(m, device=fit.device, dtype=fit.dtype) + 1e-6
    for j in range(m):
        scores = (normalized / weights[j]).amax(1) + 0.1 * normalized[:, j] / 1e-6
        eligible = valid & ~chosen
        best = torch.where(eligible, scores, torch.inf).argmin().reshape(1)
        # Preserve the published UpdateDA's filtered-list extreme index convention.
        local = (eligible.long().cumsum(0) - 1).gather(0, best).clamp_min(0)
        chosen = chosen.scatter(0, local, True)
    chosen = chosen & valid
    distance = _distance_to_surface(fit, valid, p).clamp_min(torch.finfo(fit.dtype).eps)
    angles = torch.acos(pairwise_cosine_similarity(normalized, normalized))

    def cond(selected):
        return selected.sum() < capacity

    def body(selected):
        separation = torch.where(selected[None, :], angles, torch.inf).amin(1)
        # UpdateDA's FE/maxFE factors are shared by all candidates and cancel
        # from their ordering while FE <= maxFE.
        score = torch.where(phase, separation, separation / distance)
        index = torch.where(valid & ~selected, score, -torch.inf).argmax().reshape(1)
        return (selected.scatter(0, index, True),)

    # If the archive is short, keep its first front without invoking truncation.
    (selected,) = torch.while_loop(lambda selected: overflow & cond(selected), body, (chosen,))
    index, occupied = masked_select_indices(torch.where(overflow, selected, valid), capacity)
    return compact[index], occupied


class MaOEAHAP(Algorithm):
    """A tensorized implementation of MaOEA-HAP for many-objective minimization.

    The convergence archive uses an additive epsilon indicator; the diversity archive
    uses angle-dependent NDQ sorting, front-shape estimation and adaptive exploration.
    ``pop`` and ``fit`` expose the diversity archive, with shapes ``(N, D)`` and
    ``(N, n_objs)``. ``valid`` marks occupied slots; padding repeats an occupied row.
    The convergence archive is available as ``ca_pop`` and ``ca_fit``.

    .. note::

        The current implementation accepts an objective tensor of shape ``(B, n_objs)``.
        General constraint tuples raise ``ValueError``. The upstream CrowdDistance's
        partial CV path is not included. Each step evaluates ``2 * pop_size`` offspring.
        Use ``algorithm.fit[algorithm.valid]`` when reading actual archive members.

    :references:
        [1] X. Yue, W. Wen, Y. Jiang, Y. Tian, and H. Peng, "A hyper-curvature balanced indicator and adaptive
            phase exploration co-driven evolutionary algorithm for many-objective optimization,"
            Swarm and Evolutionary Computation, 2026.
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        max_evaluations: int = 100000,
        device: torch.device | str | None = None,
    ):
        """Initialize MaOEAHAP population and optimization state.

        :param pop_size: Required. Population/archive size N; require N >= max(4, n_objs).
        :type pop_size: int
        :param n_objs: Required. Number of minimized objectives, at least two; must match fitness columns.
        :type n_objs: int
        :param lb: Required. Lower bounds of shape ``(D,)``. Use floating tensors and match ``ub`` in
            shape, dtype and device; satisfy ``lb <= ub`` elementwise. Decision dimension is inferred.
        :type lb: torch.Tensor
        :param ub: Required. Upper bounds of shape ``(D,)``, matching ``lb``.
        :type ub: torch.Tensor
        :param max_evaluations: Default: ``100000``. Nominal evaluation budget, including initialization. The time-dependent
            scalar factors
            in the source do not change candidate ordering while the budget is respected.
        :type max_evaluations: int
        :param device: Default: ``None``. Optional device override. ``None`` preserves the bounds
            device; an explicit value copies both bounds to that device.
        :type device: torch.device, str or None

        .. note::

            Connect the algorithm through a workflow and call ``workflow.init_step()`` before stepping
            or compiling ``workflow.step``. The caller controls termination. Device and dtype follow
            the bounds unless an explicit device is supplied.

            The current implementation accepts an objective tensor of shape ``(B, n_objs)``.
            General constraint tuples raise ``ValueError``. The upstream CrowdDistance's
            partial CV path is not included. Each step evaluates ``2 * pop_size`` offspring.
            Use ``algorithm.fit[algorithm.valid]`` when reading actual archive members.
        """
        super().__init__()
        if pop_size < max(4, n_objs) or n_objs < 2 or max_evaluations < pop_size:
            raise ValueError("Require pop_size >= max(4, n_objs), n_objs >= 2 and max_evaluations >= pop_size")
        if lb.ndim != 1 or lb.shape != ub.shape or not lb.is_floating_point() or lb.dtype != ub.dtype:
            raise ValueError("lb and ub must be one-dimensional floating tensors with matching shapes and dtypes")
        self.pop_size, self.n_objs, self.dim = pop_size, n_objs, lb.numel()
        self.max_evaluations = max_evaluations
        self.lb, self.ub = lb.to(device=device), ub.to(device=device)
        self.pop = Mutable(
            self.lb + torch.rand(pop_size, self.dim, device=self.lb.device, dtype=lb.dtype) * (self.ub - self.lb)
        )
        self.fit = Mutable(self.pop.new_full((pop_size, n_objs), torch.inf))
        self.valid = Mutable(torch.ones(pop_size, device=self.lb.device, dtype=torch.bool))
        self.evaluations = Mutable(torch.zeros((), device=self.lb.device, dtype=torch.long))
        register_lazy_buffer(self, "cv", device_like="pop")
        self.ca_pop = Mutable(self.pop.clone())
        self.ca_fit = Mutable(self.fit.clone())
        self.phase = Mutable(torch.zeros((), dtype=torch.bool, device=self.pop.device))
        self.p = 1 / n_objs

    def _evaluate(self, pop):
        """Evaluate a decision batch, returning objectives and optional violations.

        Every evaluated row contributes to ``self.evaluations``.
        """
        fit, cv = parse_evaluate(self.evaluate(pop))
        self.evaluations = self.evaluations + pop.shape[0]
        return fit, cv

    def _variation(self, parents, crossover=True, mutation=True, distribution=20):
        """Generate bounded offspring with SBX and/or polynomial mutation."""
        if crossover:
            parents = simulated_binary(parents[: 2 * (parents.shape[0] // 2)], pro_c=1.0, dis_c=float(distribution))
        if mutation:
            parents = polynomial_mutation(parents, self.lb, self.ub, pro_m=1.0, dis_m=float(distribution))
        return parents.clamp(self.lb, self.ub)

    def init_step(self):
        """Evaluate the initial population and initialize both archives.

        Invoke through the workflow before the first step or compilation.

        :returns: ``None``; population, fitness and auxiliary state are updated in place.
        """
        self.fit, self.cv = self._evaluate(self.pop)
        if self.cv is not None:
            raise ValueError("MaOEA-HAP supports objective-only real-valued evaluations")
        self.ca_pop, self.ca_fit = self.pop.clone(), self.fit.clone()
        first = _ndq(self.fit, self.valid)
        index, self.valid = masked_select_indices(first, self.pop_size)
        self.pop, self.fit = self.pop[index], self.fit[index]

    def step(self):
        """Generate 2N offspring, update both archives, and refresh the exploration phase.

        Invoke through the workflow after initialization; the caller controls termination.

        :returns: ``None``; population, fitness and auxiliary state are updated in place.
        """
        n = self.pop_size
        half = (n + 1) // 2
        pair = torch.randint(n, (2, half), device=self.pop.device)
        left, right = self.ca_fit[pair[0]], self.ca_fit[pair[1]]
        better = (left <= right).all(1) & (left < right).any(1)
        first = self.ca_pop[torch.where(better, pair[0], pair[1])]
        cross = self._variation(torch.cat((first, self.pop[sample_masked_indices(self.valid, half)])), mutation=False)
        rank = _ranks(self.ca_fit, torch.ones_like(self.valid))
        distance = crowding_distance_by_rank(self.ca_fit, rank, constant_boundaries=True)
        first_rival = torch.randint(n, (n,), device=self.pop.device)
        pair = torch.stack((first_rival, (first_rival + torch.randint(1, n, (n,), device=self.pop.device)) % n))
        choose_left = (distance[pair[0]] > distance[pair[1]]) | (
            (distance[pair[0]] == distance[pair[1]]) & (torch.rand(n, device=self.pop.device) < 0.5)
        )
        mutation = self._variation(self.ca_pop[torch.where(choose_left, pair[0], pair[1])][: 2 * (n // 2)], crossover=False)
        offspring = torch.cat((cross, mutation))
        off_fit, _ = self._evaluate(offspring)
        ca_pop, ca_fit = torch.cat((self.ca_pop, offspring)), torch.cat((self.ca_fit, off_fit))
        ca_valid = torch.ones(ca_fit.shape[0], dtype=torch.bool, device=self.pop.device)
        selected, _ = _indicator_select(ca_fit, ca_valid, n)
        self.ca_pop, self.ca_fit = ca_pop[selected], ca_fit[selected]
        pop, fit = torch.cat((self.pop, offspring)), torch.cat((self.fit, off_fit))
        valid = torch.cat((self.valid, ca_valid[n:]))
        selected, self.valid = _diversity_select(fit, valid, n, self.p, self.phase)
        self.pop, self.fit = pop[selected], fit[selected]
        self.phase = torch.where(self.valid, _distance_to_surface(self.fit, self.valid, self.p) < 1, True).all()
