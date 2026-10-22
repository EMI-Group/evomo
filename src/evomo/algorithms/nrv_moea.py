"""Tensorized NRV-MOEA, adapted from BIMK/PlatEMO (GPL-3.0)."""

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import lexsort

from evomo.operators.selection import non_dominate_rank
from evomo.operators.selection.constraint_handling import worst_constraint_index
from evomo.utils import (
    masked_select_indices,
    minmax_normalize,
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


def _vertmap(archive, archive_valid, points):
    """Project objective rows orthogonally onto the hyperplane defined by archive extremes."""
    m = points.shape[1]
    used = torch.zeros_like(archive_valid)
    extremes = []
    for j in range(m):
        available = archive_valid & ~used
        eligible = torch.where(available.any(), available, archive_valid)
        index = torch.where(eligible, archive[:, j], -torch.inf).argmax().reshape(1)
        extremes.append(index)
        used = used.scatter(0, index, True)
    extreme = archive[torch.cat(extremes)]
    plane, info = torch.linalg.solve_ex(extreme, points.new_ones(m), check_errors=False)
    plane = torch.where((info == 0) & torch.isfinite(plane).all(), plane, torch.ones_like(plane))
    denominator = plane.square().sum().clamp_min(torch.finfo(points.dtype).eps)
    return points - ((points @ plane - 1) / denominator)[:, None] * plane, plane


def _ward(points, valid, clusters):
    """Exact Ward agglomeration with fixed-shape centroids and cluster labels."""
    n = points.shape[0]
    labels = torch.arange(n, device=points.device)
    upper = labels[:, None] < labels[None, :]

    def cond(centroids, sizes, groups):
        return (sizes > 0).sum() > clusters.clamp_min(1)

    def body(centroids, sizes, groups):
        total = sizes[:, None] + sizes[None, :]
        weight = sizes[:, None] * sizes[None, :] / total.clamp_min(1)
        distance = (centroids[:, None] - centroids[None, :]).square().sum(2) * weight
        eligible = (sizes[:, None] > 0) & (sizes[None, :] > 0) & upper
        pair = torch.where(eligible, distance, torch.inf).flatten().argmin()
        i, j = (pair // n).reshape(1), (pair % n).reshape(1)
        si, sj = sizes.gather(0, i), sizes.gather(0, j)
        new = (centroids.index_select(0, i) * si[:, None] + centroids.index_select(0, j) * sj[:, None]) / (si + sj).clamp_min(
            1
        )[:, None]
        centroids = centroids.scatter(0, i[:, None].expand(-1, points.shape[1]), new)
        sizes = sizes.scatter(0, i, si + sj).scatter(0, j, 0)
        groups = torch.where(groups == j.squeeze(0), i.squeeze(0), groups)
        return centroids, sizes, groups

    centers, sizes, labels = torch.while_loop(cond, body, (points.clone(), valid.to(points.dtype), labels))
    return labels, centers, sizes


def _archive(pop, fit, valid, capacity):
    """Update the objective-unique non-dominated archive using masked indicator truncation."""
    order = lexsort(torch.where(valid[:, None], fit, torch.inf).T.flip(0).unbind(0))
    values, occupied = fit[order], valid[order]
    occupied = unique_row_mask(values, occupied)
    occupied = occupied & (_ranks(values, occupied) == 0)
    overflow = occupied.sum() > capacity
    selected, occupied = _indicator_select(values, occupied, capacity)
    points = values[selected]
    ideal = torch.where(occupied[:, None], points, torch.inf).amin(0)
    distance = (points - ideal).norm(dim=1)
    mean = torch.where(occupied, distance, 0).sum() / occupied.sum().clamp_min(1)
    occupied = occupied & (~overflow | (distance <= 10 * mean))
    return order[selected], occupied


class NRVMOEA(Algorithm):
    """A tensorized implementation of the normal-reference-vector NRV-MOEA.

    Objective vectors are projected onto an archive-defined hyperplane and grouped
    using Ward clustering. A non-dominated external archive guides the projection.
    After initialization, ``pop`` and ``fit`` have capacity ``N + n_objs``, matching
    the source's possible extreme-point append operation. ``valid`` identifies actual
    members; padded slots repeat an occupied row and never enter mating or selection.
    ``identity`` retains evaluated-solution identity independently of objective ties.

    .. note::

        The current implementation accepts objective tensors of shape ``(B, n_objs)``;
        constraint tuples raise ``ValueError``. Use float32 or float64 bounds for the
        hyperplane solve. Each step evaluates ``2 * floor(pop_size / 2)`` offspring.
        Use ``algorithm.fit[algorithm.valid]`` when reading actual population members.

    :references:
        [1] Y. Hua, Q. Liu, and K. Hao, "Adaptive normal vector guided evolutionary multi- and many-objective
            optimization," Complex & Intelligent Systems, vol. 10, pp. 3709-3726, 2024.
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
        """Initialize NRVMOEA population and optimization state.

        :param pop_size: Required. Population/archive size N; require N >= max(4, n_objs).
        :type pop_size: int
        :param n_objs: Required. Number of minimized objectives, at least two; must match fitness columns.
        :type n_objs: int
        :param lb: Required. Lower bounds of shape ``(D,)``. Use floating tensors and match ``ub`` in
            shape, dtype and device; satisfy ``lb <= ub`` elementwise. Decision dimension is inferred.
        :type lb: torch.Tensor
        :param ub: Required. Upper bounds of shape ``(D,)``, matching ``lb``.
        :type ub: torch.Tensor
        :param max_evaluations: Default: ``100000``. Evaluation horizon for the 10% scale-update and 90% late-stage selection
            schedules.
            Counts include initial evaluations and offspring; it does not terminate the workflow.
        :type max_evaluations: int
        :param device: Default: ``None``. Optional device override. ``None`` preserves the bounds
            device; an explicit value copies both bounds to that device.
        :type device: torch.device, str or None

        .. note::

            Connect the algorithm through a workflow and call ``workflow.init_step()`` before stepping
            or compiling ``workflow.step``. The caller controls termination. Device and dtype follow
            the bounds unless an explicit device is supplied.

            The current implementation accepts objective tensors of shape ``(B, n_objs)``;
            constraint tuples raise ``ValueError``. Use float32 or float64 bounds for the
            hyperplane solve. Each step evaluates ``2 * floor(pop_size / 2)`` offspring.
            Use ``algorithm.fit[algorithm.valid]`` when reading actual population members.
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
        self.archive_pop = Mutable(self.pop.clone())
        self.archive_fit = Mutable(self.fit.clone())
        self.archive_valid = Mutable(self.valid.clone())
        self.scale = Mutable(self.pop.new_ones(n_objs))
        self.identity = Mutable(torch.arange(pop_size, device=self.pop.device))
        self.archive_identity = Mutable(self.identity.clone())

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
        """Evaluate N initial solutions, initialize the archive, and pad population state to capacity N+M.

        Invoke through the workflow before the first step or compilation.

        :returns: ``None``; population, fitness and auxiliary state are updated in place.
        """
        self.fit, self.cv = self._evaluate(self.pop)
        if self.cv is not None:
            raise ValueError("NRV-MOEA supports objective-only real-valued evaluations")
        self.scale = self.fit.amax(0) - self.fit.amin(0)
        index, self.archive_valid = _archive(self.pop, self.fit, self.valid, self.pop_size)
        self.archive_pop, self.archive_fit = self.pop[index], self.fit[index]
        self.archive_identity = self.identity[index]
        slots = torch.arange(self.pop_size + self.n_objs, device=self.pop.device)
        index = slots.clamp_max(self.pop_size - 1)
        self.pop, self.fit, self.valid = self.pop[index], self.fit[index], slots < self.pop_size
        self.identity = self.identity[index]

    def step(self):
        """Update the archive, reproduce, and select survivors by projection and Ward clustering.

        Invoke through the workflow after initialization; the caller controls termination.

        :returns: ``None``; population, fitness and auxiliary state are updated in place.
        """
        n = self.pop_size
        pool = torch.cat((self.archive_pop, self.pop))
        values = torch.cat((self.archive_fit, self.fit))
        occupied = torch.cat((self.archive_valid, self.valid))
        identities = torch.cat((self.archive_identity, self.identity))
        index, self.archive_valid = _archive(pool, values, occupied, n)
        self.archive_pop, self.archive_fit = pool[index], values[index]
        self.archive_identity = identities[index]
        pool, values = torch.cat((self.pop, self.archive_pop)), torch.cat((self.fit, self.archive_fit))
        occupied = torch.cat((self.valid, self.archive_valid))
        identities = torch.cat((self.identity, self.archive_identity))
        offspring = self._variation(pool[sample_masked_indices(occupied, n)])
        off_fit, _ = self._evaluate(offspring)
        off_identity = torch.arange(off_fit.shape[0], device=pool.device) + self.evaluations - off_fit.shape[0]
        identities = torch.cat((identities, off_identity))
        pool, values = torch.cat((pool, offspring)), torch.cat((values, off_fit))
        occupied = torch.cat((occupied, torch.ones(off_fit.shape[0], dtype=torch.bool, device=pool.device)))
        front = _ranks(values, occupied)
        cutoff = front.kthvalue(n).values
        complete, last = front < cutoff, front == cutoff
        # Source adds complete-front points if the cutoff front has fewer than M.
        random = torch.rand(values.shape[0], device=pool.device)
        order = torch.argsort(torch.where(complete, random, torch.inf), stable=True)
        position = torch.empty_like(order).scatter(0, order, torch.arange(order.numel(), device=pool.device))
        last = last | ((last.sum() < self.n_objs) & complete & (position < (10 - last.sum())))
        ideal = torch.where(occupied[:, None], values, torch.inf).amin(0)
        maximum = torch.where(last[:, None], values, -torch.inf).amax(0)
        interval = max(1, (self.max_evaluations + 9) // 10)
        self.scale = torch.where(self.evaluations.remainder(interval) == 0, maximum - ideal, self.scale)
        self.scale = torch.where(self.scale != 0, self.scale, 1e-5)
        normalized = (values - ideal) / self.scale
        projected, plane = _vertmap((self.archive_fit - ideal) / self.scale, self.archive_valid, normalized)
        groups, centers, _ = _ward(projected, last, n - complete.sum())
        distance = (projected - centers[groups]).norm(dim=1) + (normalized @ plane - 1) / plane.norm().clamp_min(1e-12)
        best = values.new_full((values.shape[0],), torch.inf).scatter_reduce(
            0, groups, torch.where(last, distance, torch.inf), reduce="amin"
        )
        candidate = last & (distance == best[groups])
        ids = torch.arange(values.shape[0], device=pool.device)
        minimum = ids.new_full((ids.numel(),), ids.numel()).scatter_reduce(
            0, groups, torch.where(candidate, ids, ids.numel()), reduce="amin"
        )
        selected = complete | (candidate & (ids == minimum[groups]))
        # EP is unique by SOLUTION identity, not by objective value: repeated
        # archive/population identities are removed before final truncation.
        same = identities[:, None] == identities[None, :]
        selected = selected & ~(same & selected[:, None] & (ids[:, None] < ids[None, :])).any(0)
        truncate = (selected.sum() > n) | (self.evaluations >= 0.9 * self.max_evaluations)
        front = _ranks(values, selected)
        ep_cutoff = front.kthvalue(n).values
        eligible = selected & (~truncate | (front <= ep_cutoff))
        extreme = torch.zeros_like(eligible)
        for j in range(self.n_objs):
            index = torch.where(eligible & ~extreme, values[:, j], torch.inf).argmin().reshape(1)
            extreme = extreme.scatter(0, index, True)
        remaining = eligible & ~extreme
        pair_distance = torch.cdist(normalized, normalized)
        eye = torch.eye(values.shape[0], device=pool.device, dtype=torch.bool)

        def cond(active):
            return truncate & (active.sum() > n)

        def body(active):
            distance = torch.where(active[:, None] & active[None, :] & ~eye, pair_distance, torch.inf).amin(1)
            index = torch.where(active, distance, torch.inf).argmin().reshape(1)
            return (active.scatter(0, index, False),)

        (remaining,) = torch.while_loop(cond, body, (remaining,))
        selected = torch.where(truncate, remaining | extreme, selected)
        index, self.valid = masked_select_indices(selected, n + self.n_objs)
        self.pop, self.fit = pool[index], values[index]
        self.identity = identities[index]
