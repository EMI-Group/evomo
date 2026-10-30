"""Batched graph/attention GCNMOEA adapted from BIMK/PlatEMO (GPL-3.0).

Graph neighborhoods and PSO particles are processed synchronously. This is an
explicit tensorized variant: the MATLAB implementation updates overlapping
neighborhoods and the PSO global best sequentially. See the class docstring.
"""

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling

from evomo.operators.selection import non_dominate_rank
from evomo.operators.selection.constraint_handling import prefer_by_constraint, total_violation, update_last_proposals
from evomo.operators.selection.non_dominate import dominate_relation
from evomo.utils import minmax_normalize, pairwise_cosine_similarity, parse_evaluate, register_lazy_buffer


def _graph(decisions, connections, assortative):
    centered = decisions - decisions.mean(1, keepdim=True)
    similarity = pairwise_cosine_similarity(centered, centered)
    n = decisions.shape[0]
    # MATLAB sorts R(:) in column-major order and uses stable sorting at ties.
    score = torch.where(assortative, -similarity.T, similarity.T)
    order = torch.argsort(score.flatten(), stable=True)

    def cond(position, adjacency, degree):
        return (position < n * n) & (degree < connections).any()

    def body(position, adjacency, degree):
        flat = order.gather(0, position.reshape(1)).squeeze(0)
        i, j = flat % n, flat // n
        allowed = (degree.gather(0, i.reshape(1)) < connections) & (degree.gather(0, j.reshape(1)) < connections) & (i != j)
        pair = torch.stack((i * n + j, j * n + i))
        previous = adjacency.flatten().gather(0, pair)
        added = allowed & ~previous[:1]
        adjacency = adjacency.flatten().scatter(0, pair, allowed.expand(2) | previous).reshape(n, n)
        degree = degree.scatter_add(0, torch.stack((i, j)), added.long().expand(2))
        return position + 1, adjacency, degree

    _, adjacency, _ = torch.while_loop(
        cond, body, (order.new_zeros(()), torch.zeros((n, n), dtype=torch.bool, device=decisions.device), order.new_zeros(n))
    )
    return adjacency


def _centrality(adjacency, dtype):
    """PageRank and exact unweighted undirected betweenness on tensors."""
    n = adjacency.shape[0]
    matrix = adjacency.to(dtype)
    degree = matrix.sum(1)
    transition = matrix / degree[:, None].clamp_min(1)
    rank = matrix.new_full((n,), 1 / n)

    def pr_cond(i, previous, current):
        return (i < 100) & ((previous - current).abs().sum() > 1e-10)

    def pr_body(i, previous, current):
        updated = 0.85 * (current @ transition + (current * (degree == 0)).sum() / n) + 0.15 / n
        return i + 1, current.clone(), updated

    _, _, rank = torch.while_loop(pr_cond, pr_body, (degree.new_zeros((), dtype=torch.long), rank.new_zeros(n), rank))
    eye = torch.eye(n, device=matrix.device, dtype=torch.bool)
    distance = torch.where(eye, matrix.new_zeros(()), torch.inf)
    sigma = eye.to(dtype)

    def bfs_cond(depth, dist, paths, frontier):
        return (depth <= n) & (frontier > 0).any()

    def bfs_body(depth, dist, paths, frontier):
        new_paths = torch.where(torch.isinf(dist), frontier @ matrix, 0)
        dist = torch.where(new_paths > 0, depth.to(dtype), dist)
        return depth + 1, dist, paths + new_paths, new_paths

    depth, distance, sigma, _ = torch.while_loop(
        bfs_cond, bfs_body, (degree.new_ones((), dtype=torch.long), distance, sigma, sigma.clone())
    )

    def back_cond(level, dependency):
        return level > 1

    def back_body(level, dependency):
        weighted = torch.where(distance == level - 1, (1 + dependency) / sigma.clamp_min(1), 0)
        addition = sigma * (weighted @ matrix.T)
        return level - 1, dependency + torch.where(distance == level - 2, addition, 0)

    _, dependency = torch.while_loop(back_cond, back_body, (depth, torch.zeros_like(matrix)))
    between = torch.where(eye, 0, dependency).sum(0) / 2
    return 0.5 * rank + 0.5 * between


def _attention(decisions, adjacency, weights, attention, valid):
    """Masked graph attention, with optional leading particle/batch dimensions."""
    transformed = decisions @ weights
    d = transformed.shape[-1]
    left = (transformed * attention[..., None, :d]).sum(-1)
    right = (transformed * attention[..., None, d:]).sum(-1)
    logits = (left[..., :, None] + right[..., None, :]).tanh()
    edges = adjacency & valid[..., :, None] & valid[..., None, :]
    # Isolated nodes retain their transformed features instead of producing NaNs.
    fallback = torch.eye(decisions.shape[-2], device=decisions.device, dtype=torch.bool)
    edges = edges | (~edges.any(-1, keepdim=True) & fallback)
    probabilities = torch.where(edges, logits.exp(), 0)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-30)
    return probabilities @ transformed


def _exact_hv(fit, reference, valid):
    """Exact small-set HV by inclusion-exclusion, batched over PSO particles."""
    k = fit.shape[-2]
    fit = torch.where(valid[..., :, None] & (fit < reference[..., None, :]).all(-1, keepdim=True), fit, reference[..., None, :])
    intersections = fit.new_full((*fit.shape[:-2], 1, fit.shape[-1]), -torch.inf)
    sign = fit.new_ones(1)
    for i in range(k):
        intersections = torch.cat((intersections, torch.maximum(intersections, fit[..., i : i + 1, :])), dim=-2)
        sign = torch.cat((sign, -sign))
    volumes = (reference[..., None, :] - intersections[..., 1:, :]).clamp_min(0).prod(-1)
    return -(volumes * sign[1:]).sum(-1)


def _spd_selection(fit, reference, count):
    """PlatEMO's order-dependent SPDSort, without replacing it by Pareto ranks."""
    n = fit.shape[0]
    valid = torch.ones(n, dtype=torch.bool, device=fit.device)
    normalized = minmax_normalize(fit, valid)
    similarity = pairwise_cosine_similarity(normalized, reference)
    projection = normalized.norm(dim=1, keepdim=True) * similarity
    residual = normalized.norm(dim=1, keepdim=True) * (1 - similarity.square()).clamp_min(0).sqrt()
    d2, region = residual.min(1)
    d1 = projection.gather(1, region[:, None]).squeeze(1)
    pareto = non_dominate_rank(fit) == 0
    extreme = torch.where(pareto[:, None], normalized, -torch.inf).argmax(0)
    is_extreme = torch.zeros_like(valid).scatter(0, extreme, True)
    d1, d2 = torch.where(is_extreme, 0, d1), torch.where(is_extreme, 0, d2)
    dominates = dominate_relation(fit, fit)
    unrelated = ~dominates & ~dominates.T
    score = d1 + 5 * d2
    relation = dominates | (unrelated & (region[:, None] == region[None, :]) & (score[:, None] < score[None, :]))
    ids = torch.arange(n, device=fit.device)

    def cond(level, front):
        return (front < n).sum() < count

    def body(level, front):
        def scan_cond(i, excluded, current):
            return i < n

        def scan_body(i, excluded, current):
            idx = i.reshape(1)
            eligible = ~excluded.gather(0, idx).squeeze(0)
            later = (ids > i) & ~excluded
            incoming = relation.index_select(1, idx).squeeze(1) & later
            first = torch.where(incoming, ids, n).amin()
            outgoing = relation.index_select(0, idx).squeeze(0) & later & (ids < first) & eligible
            excluded = excluded | outgoing
            rejected = eligible & (first < n)
            excluded = excluded.scatter(0, idx, (excluded.gather(0, idx) | rejected).reshape(1))
            current = current.scatter(
                0, idx, torch.where(eligible & ~rejected, level, current.gather(0, idx).squeeze(0)).reshape(1)
            )
            return i + 1, excluded, current

        _, _, front = torch.while_loop(scan_cond, scan_body, (ids.new_zeros(()), front < n, front.clone()))
        return level + 1, front

    _, front = torch.while_loop(cond, body, (ids.new_zeros(()), ids.new_full((n,), n)))
    cutoff = front.kthvalue(count).values
    # All complete fronts, then smallest perpendicular distance in the last front.
    order = torch.argsort(d2, stable=True)
    order = order[torch.argsort((front[order] >= cutoff).long(), stable=True)]
    order = order[torch.argsort((front[order] > cutoff).long(), stable=True)]
    selected = torch.sort(order[:count]).values
    return selected, front[selected], d2[selected]


def _reference_selection(fit, reference):
    """Choose reference representatives sequentially from complete cutoff fronts."""
    n = reference.shape[0]
    rank = non_dominate_rank(fit)
    active = rank <= rank.kthvalue(n).values
    result = []
    for i in range(n):
        normalized = minmax_normalize(fit, active)
        score = pairwise_cosine_similarity(normalized, reference[i : i + 1]).squeeze(1)
        index = torch.where(active, score, -torch.inf).argmax().reshape(1)
        result.append(index)
        active = active.scatter(0, index, False)
    return torch.cat(result)


class GCNMOEA(Algorithm):
    """A synchronous tensorized variant of the graph-attention algorithm GCNMOEA.

    A degree-capped correlation graph supplies red/green/black neighborhoods and
    maximum-clique subsets. Graph-attention parameters are optimized with PSO driven
    by exact local hypervolume. Neighborhood proposals and PSO particles update
    synchronously, changing the sequential update order of the PlatEMO implementation.
    Overlapping eligible replacement proposals use a deterministic last-source rule.

    ``pop`` and ``fit`` have shapes ``(N, D)`` and ``(N, n_objs)``. ``evaluations``
    counts every evaluated row, including internal PSO probes and fixed-shape padding.
    ``search_evaluations`` excludes PSO probes and controls the outer search schedule.
    The original source resets probe counts; include them when comparing budgets.

    .. note::

        Evaluation accepts objectives or ``(fitness, constraint_violation)``. Violations
        of shape ``(B,)`` or ``(B, C)`` are retained in ``cv``. Neighborhood replacement
        compares total positive violation, then scalarized objectives at equal violation.
        Other selection stages and the PSO hypervolume objective remain objective-based;
        this is the source's partial constraint rule, not a full feasibility-first selector.

        Uniform reference sampling determines the actual population size. Inner tensor
        loops require functional evaluation and monitoring; the default monitor is
        supported. Exact local HV grows exponentially with neighborhood size. The defaults
        ``graph_samples=50``, ``particles=30`` and ``pso_iterations=100`` can be expensive.
        Fullgraph capture does not establish single-kernel execution, CUDA Graph support,
        or vmap compatibility.

    :references:
        [1] P. Yan, Y. Tian, and Y. Liu, "An indicator-based multi-objective evolutionary algorithm assisted by
            improved graph convolutional networks," Swarm and Evolutionary Computation, vol. 94, 101892, 2025.
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        max_evaluations: int = 100000,
        graph_samples: int = 50,
        particles: int = 30,
        pso_iterations: int = 100,
        device: torch.device | str | None = None,
    ):
        """Initialize GCNMOEA population and optimization state.

        :param pop_size: Required. Requested population size; uniform reference sampling sets the actual N.
        :type pop_size: int
        :param n_objs: Required. Number of minimized objectives, at least two; must match fitness columns.
        :type n_objs: int
        :param lb: Required. Lower bounds of shape ``(D,)``. Use floating tensors and match ``ub`` in
            shape, dtype and device; satisfy ``lb <= ub`` elementwise. Decision dimension is inferred.
        :type lb: torch.Tensor
        :param ub: Required. Upper bounds of shape ``(D,)``, matching ``lb``.
        :type ub: torch.Tensor
        :param max_evaluations: Default: ``100000``. Outer-search horizon used with ``search_evaluations`` to choose phases. PSO
            probes
            are excluded from this schedule but included in ``evaluations``. It does not terminate the workflow.
        :type max_evaluations: int
        :param graph_samples: Default: ``50``. Positive number of red-node samples per graph phase,
            sampled with replacement. Each sample supplies at most eight green neighborhoods.
        :type graph_samples: int
        :param particles: Default: ``30``. Positive PSO particle count for each sampled red neighborhood.
            Each particle contains a decision-space transformation and attention parameters.
        :type particles: int
        :param pso_iterations: Default: ``100``. Positive number of synchronous PSO updates.
            Initial particle evaluation is additional to these iterations.
        :type pso_iterations: int
        :param device: Default: ``None``. Optional device override. ``None`` preserves the bounds
            device; an explicit value copies both bounds to that device.
        :type device: torch.device, str or None

        .. note::

            Connect the algorithm through a workflow and call ``workflow.init_step()`` before stepping
            or compiling ``workflow.step``. The caller controls termination. Device and dtype follow
            the bounds unless an explicit device is supplied.

            Evaluation accepts objectives or ``(fitness, constraint_violation)``. Violations
            of shape ``(B,)`` or ``(B, C)`` are retained in ``cv``. Neighborhood replacement
            compares total positive violation, then scalarized objectives at equal violation.
            Other selection stages and the PSO hypervolume objective remain objective-based;
            this is the source's partial constraint rule, not a full feasibility-first selector.

            Uniform reference sampling determines the actual population size. Inner tensor
            loops require functional evaluation and monitoring; the default monitor is
            supported. Exact local HV grows exponentially with neighborhood size. The defaults
            ``graph_samples=50``, ``particles=30`` and ``pso_iterations=100`` can be expensive.
            Fullgraph capture does not establish single-kernel execution, CUDA Graph support,
            or vmap compatibility.
        """
        ref, size = uniform_sampling(pop_size, n_objs)
        super().__init__()
        pop_size = size
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
        if graph_samples < 1 or particles < 1 or pso_iterations < 1:
            raise ValueError("graph_samples, particles and pso_iterations must be positive")
        self.reference = ref.to(self.pop)
        self.graph_samples, self.particles, self.pso_iterations = graph_samples, particles, pso_iterations
        self.green_count, self.black_count = min(8, size - 1), min(10, size - 2)
        self.local_size = self.black_count + 1
        bit_ids = torch.arange(2**self.black_count, device=self.pop.device)
        self.subsets = ((bit_ids[:, None] >> torch.arange(self.black_count, device=self.pop.device)) & 1).bool()
        self.ideal = Mutable(self.pop.new_full((n_objs,), torch.inf))
        self.front = Mutable(torch.zeros(size, device=self.pop.device, dtype=torch.long))
        self.distance = Mutable(self.pop.new_zeros(size))
        self.search_evaluations = Mutable(self.evaluations.clone())

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
        """Evaluate initial solutions and initialize the ideal point, SPD scores and search counter.

        Invoke through the workflow before the first step or compilation.

        :returns: ``None``; population, fitness and auxiliary state are updated in place.
        """
        self.fit, self.cv = self._evaluate(self.pop)
        self.ideal = self.fit.amin(0)
        _, self.front, self.distance = _spd_selection(self.fit, self.reference, self.pop_size)
        self.search_evaluations = self.evaluations.clone()

    def _raw_evaluate(self, decisions):
        """Evaluate leading batches and return matching objective and violation tensors without updating counters."""
        shape = decisions.shape[:-1]
        fit, cv = parse_evaluate(self.evaluate(decisions.reshape(-1, self.dim)))
        cv = fit.new_zeros((fit.shape[0], 0)) if cv is None else cv.reshape(fit.shape[0], -1)
        return fit.reshape(*shape, self.n_objs), cv.reshape(*shape, cv.shape[-1])

    def _neighborhoods(self, adjacency):
        """Sample fixed-capacity red/green/black neighborhoods and maximum-clique masks."""
        r, g, b = self.graph_samples, self.green_count, self.black_count
        red = torch.randint(self.pop_size, (r,), device=self.pop.device)
        noise = torch.rand(r, self.pop_size, device=self.pop.device)
        green = torch.where(adjacency[red], noise, torch.inf).argsort(1)[:, :g]
        green_valid = adjacency[red[:, None], green]
        neighbors = adjacency[green] & (
            torch.arange(self.pop_size, device=self.pop.device)[None, None, :] != red[:, None, None]
        )
        noise = torch.rand(r, g, self.pop_size, device=self.pop.device)
        black = torch.where(neighbors, noise, torch.inf).argsort(2)[:, :, :b]
        valid = neighbors.gather(2, black) & green_valid[:, :, None]
        subgraph = adjacency[black[..., :, None], black[..., None, :]]
        conflict = ~subgraph & ~torch.eye(b, device=self.pop.device, dtype=torch.bool)
        membership = self.subsets
        bad = (conflict[:, :, None] & membership[None, None, :, :, None] & membership[None, None, :, None, :]).any((-1, -2))
        bad = bad | (membership[None, None] & ~valid[:, :, None]).any(-1)
        sizes = membership.sum(1)
        best = torch.where(~bad, sizes[None, None, :], -1).argmax(2)
        clique = membership[best] & valid
        fallback = valid & (valid.long().cumsum(2) <= 4)
        chosen = torch.where(clique.sum(2, keepdim=True) < 2, fallback, clique)
        indices = torch.cat((green[:, :, None], black), 2)
        local_valid = torch.cat((green_valid[:, :, None], chosen), 2)
        local_adj = adjacency[indices[..., :, None], indices[..., None, :]]
        return red, green, green_valid, indices, local_valid, local_adj

    def _optimize(self, decisions, fit, adjacency, valid):
        """Return batched PSO attention parameters and the number of internal probe evaluations."""
        r, k = decisions.shape[:2]
        p, d = self.particles, self.dim
        reference = fit.amax(1) + 0.1
        weights = torch.rand(r, p, d, d, device=decisions.device, dtype=decisions.dtype)
        attention = torch.rand(r, p, 2 * d, device=decisions.device, dtype=decisions.dtype)
        vw, va = torch.rand_like(weights), torch.rand_like(attention)
        features = decisions[:, None].expand(-1, p, -1, -1)
        edges, occupied = adjacency[:, None], valid[:, None]
        offspring = _attention(features, edges, weights, attention, occupied).clamp(self.lb, self.ub)
        values, _ = self._raw_evaluate(offspring)
        scores = _exact_hv(values, reference[:, None], occupied)
        best = scores.argmax(1)
        row = torch.arange(r, device=decisions.device)
        global_w, global_a, global_hv = weights[row, best], attention[row, best], scores[row, best]
        temperature = decisions.new_tensor(100.0)
        iteration = row.new_zeros(())

        def cond(i, w, a, velocity_w, velocity_a, best_w, best_a, best_hv, gw, ga, ghv, temp):
            return i < self.pso_iterations

        def body(i, w, a, velocity_w, velocity_a, best_w, best_a, best_hv, gw, ga, ghv, temp):
            inertia = 0.9 - 0.5 * (i + 1) / self.pso_iterations
            random = torch.rand(r, p, 4, device=decisions.device, dtype=decisions.dtype)
            velocity_w = (
                inertia * velocity_w
                + 3 * random[..., 0, None, None] * (best_w - w)
                + 3 * random[..., 1, None, None] * (gw[:, None] - w)
            )
            velocity_a = (
                inertia * velocity_a + 3 * random[..., 2, None] * (best_a - a) + 3 * random[..., 3, None] * (ga[:, None] - a)
            )
            w, a = w + velocity_w, a + velocity_a
            proposed = _attention(features, edges, w, a, occupied).clamp(self.lb, self.ub)
            values, _ = self._raw_evaluate(proposed)
            current = _exact_hv(values, reference[:, None], occupied)
            accept = ((current - best_hv) / temp).exp() > torch.rand(r, p, device=decisions.device, dtype=decisions.dtype)
            best_w = torch.where(accept[..., None, None], w, best_w)
            best_a = torch.where(accept[..., None], a, best_a)
            best_hv = torch.where(accept, current, best_hv)
            index = current.argmax(1)
            improve = current[row, index] > ghv
            gw = torch.where(improve[:, None, None], w[row, index], gw)
            ga = torch.where(improve[:, None], a[row, index], ga)
            ghv = torch.maximum(ghv, current[row, index])
            return i + 1, w, a, velocity_w, velocity_a, best_w, best_a, best_hv, gw, ga, ghv, temp * 0.99

        state = torch.while_loop(
            cond,
            body,
            (
                iteration,
                weights,
                attention,
                vw,
                va,
                weights.clone(),
                attention.clone(),
                scores,
                global_w,
                global_a,
                global_hv,
                temperature,
            ),
        )
        return state[8], state[9], r * p * k * (self.pso_iterations + 1)

    def _graph_step(self, pop, fit, cv, ideal):
        """Return graph-based survivor state and physical/search evaluation increments."""
        adjacency = _graph(pop, min(15, self.pop_size - 1), torch.rand((), device=pop.device) < 0.3)
        red, green, green_valid, indices, occupied, local_adj = self._neighborhoods(adjacency)
        local_pop, local_fit = pop[indices], fit[indices]
        first = (occupied[:, :, 1:].sum(2) > 1).long().argmax(1)
        rows = torch.arange(self.graph_samples, device=pop.device)
        weights, attention, probe_count = self._optimize(
            local_pop[rows, first], local_fit[rows, first], local_adj[rows, first], occupied[rows, first]
        )
        off = _attention(local_pop, local_adj, weights[:, None], attention[:, None], occupied).clamp(self.lb, self.ub)
        off_fit, off_cv = self._raw_evaluate(off)
        ideal = torch.minimum(ideal, torch.where(occupied[..., None], off_fit, torch.inf).amin((0, 1, 2)))
        old_score = ((local_fit - ideal).abs() * self.reference[indices]).amax(3)
        new_score = ((off_fit - ideal).abs() * self.reference[indices]).amax(3)
        old_cv = total_violation(cv[indices].reshape(indices.numel(), cv.shape[-1])).reshape_as(indices)
        new_cv = total_violation(off_cv.reshape(indices.numel(), cv.shape[-1])).reshape_as(indices)
        replace = occupied & prefer_by_constraint(new_score <= old_score, new_cv, old_cv)
        replace = replace & (replace.long().cumsum(2) <= 2)
        pop, fit, cv = update_last_proposals(
            pop,
            fit,
            off.reshape(-1, self.dim),
            off_fit.reshape(-1, self.n_objs),
            indices.reshape(-1, 1),
            replace.reshape(-1, 1),
            cv=cv,
            off_cv=off_cv.reshape(indices.numel(), cv.shape[-1]),
        )
        importance = _centrality(adjacency, pop.dtype)
        score = torch.where(green_valid, importance[green], -torch.inf)
        best = score.argsort(1, descending=True)[:, :2]
        neighbor = green.gather(1, best)
        # Differential evolution on graph-central nodes, including the source's
        # polynomial mutation. Small neighborhoods use the two-child GA path.
        trial = pop[red] + 0.5 * (pop[neighbor[:, 0]] - pop[neighbor[:, 1]])
        trial = polynomial_mutation(trial.clamp(self.lb, self.ub), self.lb, self.ub, pro_m=1.0, dis_m=20.0)
        # Parent halves match OperatorGA's pairing convention.
        ga = self._variation(torch.cat((pop[red], pop[neighbor[:, 0]])), distribution=10)
        ga_first, ga_second = ga[: self.graph_samples], ga[self.graph_samples :]
        small = green_valid.sum(1) <= 2
        trial = torch.cat((torch.where(small[:, None], ga_first, trial), ga_second))
        trial_fit, trial_cv = self._raw_evaluate(trial)
        targets = torch.cat((red, neighbor[:, 0]))
        proposal = torch.cat((green_valid.any(1), small & green_valid.any(1)))
        pop, fit, cv = update_last_proposals(
            pop, fit, trial, trial_fit, targets[:, None], proposal[:, None], cv=cv, off_cv=trial_cv
        )
        ideal = torch.minimum(ideal, trial_fit.amin(0))
        selected, front, distance = _spd_selection(fit, self.reference, self.pop_size)
        count = indices.numel() + 2 * self.graph_samples
        return (
            pop[selected],
            fit[selected],
            cv[selected],
            ideal,
            front,
            distance,
            self.evaluations.new_tensor(probe_count + count),
            occupied.long().sum() + torch.where(small, 2, 1).sum(),
        )

    def _ordinary_step(self, pop, fit, cv, ideal):
        """Return ordinary reference-selection state and its evaluation increments."""
        rivals = torch.randint(self.pop_size, (2, self.pop_size), device=pop.device)
        left, right = rivals[0], rivals[1]
        better = (self.front[left] < self.front[right]) | (
            (self.front[left] == self.front[right]) & (self.distance[left] < self.distance[right])
        )
        off = self._variation(pop[torch.where(better, left, right)])
        off_fit, off_cv = self._raw_evaluate(off)
        merged, values, violations = torch.cat((pop, off)), torch.cat((fit, off_fit)), torch.cat((cv, off_cv))
        selected = _reference_selection(values, self.reference)
        chosen = values[selected]
        front = non_dominate_rank(chosen).long()
        normalized = minmax_normalize(chosen, torch.ones_like(front, dtype=torch.bool))
        region = pairwise_cosine_similarity(normalized, self.reference).argmax(1)
        density = front.new_zeros(self.pop_size).scatter_add(0, region, torch.ones_like(region))[region].to(fit.dtype)
        count = self.evaluations.new_tensor(off.shape[0])
        return (
            merged[selected],
            chosen,
            violations[selected],
            torch.minimum(ideal, off_fit.amin(0)),
            front,
            density,
            count,
            count.clone(),
        )

    def step(self):
        """Run a graph/PSO or reference-selection phase and update both evaluation counters.

        Invoke through the workflow after initialization; the caller controls termination.

        :returns: ``None``; population, fitness and auxiliary state are updated in place.
        """
        cv = self.fit.new_zeros((self.pop_size, 0)) if self.cv is None else self.cv.reshape(self.pop_size, -1)
        graph_phase = (torch.rand((), device=self.pop.device) < 0.4) | (self.search_evaluations < 0.5 * self.max_evaluations)
        operands = (self.pop, self.fit, cv, self.ideal)
        if torch.compiler.is_compiling():
            state = torch.cond(graph_phase, self._graph_step, self._ordinary_step, operands)
        elif graph_phase:
            state = self._graph_step(*operands)
        else:
            state = self._ordinary_step(*operands)
        self.pop, self.fit, violations, self.ideal, self.front, self.distance, count, search_count = state
        if self.cv is not None:
            self.cv = violations.reshape_as(self.cv)
        self.evaluations = self.evaluations + count
        self.search_evaluations = self.search_evaluations + search_count
