import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    prefer_by_constraint,
    rank_with_constraints,
    take_violation,
    total_violation,
    worst_constraint_index,
)
from evomo.utils import pairwise_max_difference, parse_evaluate, register_lazy_buffer


class SSCEA(Algorithm):
    def __init__(
        self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, max_gen: int = 100, max_fe: int | None = None
    ):
        """Initialize the SSCEA population and optimization state.

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
        :param max_gen: Default: ``100``. Used to infer an evaluation horizon when ``max_fe`` is omitted.
            Archive size and subspace changes follow actual evaluation progress, including initialization.
        :type max_gen: int
        :param max_fe: Optional explicit evaluation budget. With a budget, the last offspring batch is clipped
            and subsequent steps after exhaustion do not evaluate further candidates.
        :type max_fe: int | None

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
        self.max_gen = Parameter(max_gen)
        self.max_fe = (
            max_fe if max_fe is not None else pop_size + 250 * self.dim + (2 * ((pop_size + 1) // 2) + pop_size) * max_gen
        )
        self.limit_fe = max_fe is not None
        self.fe = Mutable(torch.tensor(0, dtype=torch.int64, device=device))

        # Initialize State (Mutables)
        # Convergence Archive (CA) - Dynamic Size
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Diversity Archive (DA) - Fixed Size
        self.archive_pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Subspace Masks
        self.dv_mask = Mutable(torch.zeros(self.dim, dtype=torch.bool, device=device))
        self.cv_mask = Mutable(torch.zeros(self.dim, dtype=torch.bool, device=device))
        self.gen_counter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        register_lazy_buffer(self, "archive_cv", device_like="pop")

    def _evaluate(self, pop):
        result = parse_evaluate(self.evaluate(pop))
        self.fe = self.fe + pop.shape[0]
        return result

    def _variable_clustering(self, population=None, fitness=None, violation=None):
        population = self.pop if population is None else population
        fitness = self.fit if fitness is None else fitness
        rank = rank_with_constraints(fitness, violation)
        front = fitness[rank == 0]
        minimum, maximum = front.amin(0), front.amax(0)
        if bool((maximum == minimum).any()):
            minimum, maximum = torch.zeros_like(minimum), torch.ones_like(maximum)
        sample = population[torch.randint(len(population), (5,), device=population.device)]
        angles = fitness.new_zeros((self.dim, 5))
        errors = torch.zeros_like(angles)
        constraint_sensitive = torch.zeros(self.dim, dtype=torch.bool, device=population.device)
        for j in range(self.dim):
            decisions = sample.repeat(50, 1)
            decisions[:, j] = self.lb[j] + torch.rand(250, device=population.device) * (self.ub[j] - self.lb[j])
            values, cv = self._evaluate(decisions)
            points = ((values - minimum) / (maximum - minimum)).reshape(50, 5, self.n_objs).transpose(0, 1)
            points = points - points.mean(1, keepdim=True)
            _, _, vectors = torch.linalg.svd(points, full_matrices=False)
            direction = vectors[:, 0]
            residual = points - (points * direction[:, None]).sum(-1, keepdim=True) * direction[:, None]
            errors[j] = residual.square().sum((1, 2)).sqrt()
            cosine = direction.sum(-1).abs() / self.n_objs**0.5
            angles[j] = torch.acos(cosine.clamp(0, 1)) * (180 / torch.pi)
            if cv is not None:
                constraint_sensitive[j] = total_violation(cv).reshape(50, 5).std(0).amax() > 1e-6
        # Two direction clusters; convergence variables have a linear response and the smaller angle.
        means = angles.mean(1)
        centers = angles[torch.stack([means.argmin(), means.argmax()])].clone()
        labels = torch.zeros(self.dim, dtype=torch.long, device=population.device)
        for _ in range(100):
            new_labels = torch.cdist(angles, centers).argmin(1)
            if torch.equal(new_labels, labels) and _ > 0:
                break
            labels = new_labels
            for group in range(2):
                members = labels == group
                if bool(members.any()):
                    centers[group] = angles[members].mean(0)
        convergence = errors.mean(1) < 1e-2
        if bool((convergence & (labels == 0)).any()) and bool((convergence & (labels == 1)).any()):
            group_means = torch.stack([angles[convergence & (labels == group)].mean() for group in range(2)])
            convergence = convergence & (labels == group_means.argmin())
        convergence = convergence | constraint_sensitive
        return convergence, ~convergence

    def _update_ca_size(self):
        ratio = (self.fe.to(self.fit.dtype) / self.max_fe).clamp(0, 1)
        size = (self.pop_size / 10 + 0.9 * self.pop_size * ratio.square()).long()
        return size.clamp(1, self.pop_size)

    def _select_ca(self, pop, fit, cv, count):
        if len(pop) <= count:
            return pop, fit, cv
        normalized = (fit - fit.amin(0)) / (fit.amax(0) - fit.amin(0)).clamp_min(torch.finfo(fit.dtype).tiny)
        indicator = pairwise_max_difference(normalized, normalized)
        scale = indicator.abs().amax(0).clamp_min(torch.finfo(fit.dtype).eps)
        score = -torch.exp(-indicator / scale[None] / 0.05).sum(0) + 1
        active = torch.ones(len(pop), dtype=torch.bool, device=pop.device)
        for _ in range(len(pop) - int(count)):
            worst = worst_constraint_index(score, active, cv)
            active[worst] = False
            score = score + torch.exp(-indicator[worst] / scale[worst] / 0.05)
        return pop[active], fit[active], take_violation(cv, active)

    def _select_da(self, pop, fit, cv):
        rank = rank_with_constraints(fit, cv)
        cutoff = 0 if cv is None else rank.kthvalue(min(self.pop_size, len(fit))).values
        front = rank <= cutoff
        pop, fit, cv = pop[front], fit[front], take_violation(cv, front)
        rank = rank[front]
        if len(pop) <= self.pop_size:
            return pop, fit, cv
        normalized = (fit - fit.amin(0)) / (fit.amax(0) - fit.amin(0)).clamp_min(torch.finfo(fit.dtype).tiny)
        selected = rank < cutoff if cv is not None else torch.zeros(len(pop), dtype=torch.bool, device=pop.device)
        seeded = int(selected.sum())
        extremes = min(self.n_objs, self.pop_size - seeded)
        weights = torch.eye(self.n_objs, dtype=fit.dtype, device=fit.device) + 1e-6
        for objective in range(extremes):
            asf = (fit / weights[objective]).amax(1) + 0.1 * fit[:, objective] / 1e-6
            best = torch.where(~selected, asf, torch.inf).argmin()
            selected[best] = True
        unit = normalized / normalized.norm(dim=1, keepdim=True).clamp_min(torch.finfo(fit.dtype).tiny)
        angles = torch.acos((unit @ unit.T).clamp(-1, 1))
        for _ in range(self.pop_size - seeded - extremes):
            nearest = angles.masked_fill(~selected[None], torch.inf).amin(1)
            best = torch.where(~selected, nearest, -torch.inf).argmax()
            selected[best] = True
        return pop[selected], fit[selected], take_violation(cv, selected)

    def init_step(self) -> None:
        initial_pop = self.pop
        initial_fit, initial_cv = self._evaluate(initial_pop)
        self.pop, self.fit, self.cv = self._select_ca(initial_pop, initial_fit, initial_cv, max(1, self.pop_size // 10))
        self.archive_pop, self.archive_fit, self.archive_cv = self._select_da(initial_pop, initial_fit, initial_cv)
        self.cv_mask, self.dv_mask = self._variable_clustering(initial_pop, initial_fit, initial_cv)

    def step(self) -> None:
        if self.limit_fe and int(self.fe) >= self.max_fe:
            return
        self.gen_counter = self.gen_counter + 1
        device = self.lb.device
        half = (self.pop_size + 1) // 2
        candidates = torch.randint(0, len(self.pop), (2, half), device=device)
        first_fit, second_fit = self.fit[candidates[0]], self.fit[candidates[1]]
        first_better = (first_fit < second_fit).any(1) & ~(first_fit > second_fit).any(1)
        if self.cv is not None:
            totals = total_violation(self.cv)
            first_better = prefer_by_constraint(first_better, totals[candidates[0]], totals[candidates[1]])
        ca_indices = torch.where(first_better, candidates[0], candidates[1])
        da_indices = torch.randint(0, len(self.archive_pop), (half,), device=device)
        parents_c = torch.cat([self.pop[ca_indices], self.archive_pop[da_indices]])
        parents_m = self.pop[torch.randint(0, len(self.pop), (self.pop_size,), device=device)]
        ratio = self.fe.to(self.fit.dtype) / self.max_fe
        convergence = (ratio < 0.5) | ((ratio > 0.5) & (torch.rand(1, device=device) < 0.5))
        subspace = torch.where(convergence, self.cv_mask, self.dv_mask)
        crossed = simulated_binary(parents_c, pro_c=1.0, dis_c=15.0)
        probability = self.dim / (2 * subspace.sum().clamp_min(1))
        mutated = polynomial_mutation(parents_m, self.lb, self.ub, pro_m=probability, dis_m=15.0)
        offspring = clamp(
            torch.cat([torch.where(subspace, crossed, parents_c), torch.where(subspace, mutated, parents_m)]), self.lb, self.ub
        )
        if self.limit_fe:
            offspring = offspring[: self.max_fe - int(self.fe)]
        fit, cv = self._evaluate(offspring)
        ca_pop, ca_fit = torch.cat([self.pop, offspring]), torch.cat([self.fit, fit])
        ca_cv = cat_violation(self.cv, cv)
        if ca_cv is not None:
            # A growing constrained CA reuses better DA incumbents before admitting worse offspring.
            ca_pop = torch.cat([self.archive_pop, ca_pop])
            ca_fit = torch.cat([self.archive_fit, ca_fit])
            ca_cv = torch.cat([self.archive_cv, ca_cv])
        self.pop, self.fit, self.cv = self._select_ca(ca_pop, ca_fit, ca_cv, self._update_ca_size())
        da_pop, da_fit = torch.cat([self.archive_pop, offspring]), torch.cat([self.archive_fit, fit])
        self.archive_pop, self.archive_fit, self.archive_cv = self._select_da(
            da_pop, da_fit, cat_violation(self.archive_cv, cv)
        )
