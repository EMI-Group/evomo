"""Tensorized HEA, adapted from BIMK/PlatEMO (GPL-3.0)."""

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling

from evomo.utils import masked_select_indices, pairwise_cosine_similarity, parse_evaluate, register_lazy_buffer


def _domination(fit, ideal, nadir, threshold):
    """Return the hyper-dominance survivor mask and each candidate's degree."""
    normalized = fit / (nadir - ideal).clamp_min(1e-6)
    error = normalized[:, None, :] - normalized[None, :, :]
    high, low = error.amax(2), error.amin(2)
    ratio = -low / torch.where(high != 0, high, 1)
    ratio = torch.where(high <= 0, torch.inf, ratio)
    hd = ratio.amin(1)
    n = fit.shape[0]
    earlier = torch.arange(n, device=fit.device)[:, None] < torch.arange(n, device=fit.device)[None, :]
    duplicate = ((fit[:, None] == fit[None, :]).all(2) & earlier).any(0)
    return (hd >= threshold) & ~duplicate, hd


def _selection(fit, hd, valid, ideal, nadir, reference):
    """Select reference representatives and fill remaining slots by hyper-dominance degree."""
    normalized = (fit - ideal) / (nadir - ideal).clamp_min(1e-6)
    nearest = torch.where(valid[:, None], pairwise_cosine_similarity(normalized, reference), -torch.inf).argmax(0)
    chosen = torch.zeros_like(valid).scatter(0, nearest, True)
    size = reference.shape[0]
    initial, _ = masked_select_indices(chosen, size)
    # PlatEMO's fill deliberately includes already chosen reference representatives.
    rest = torch.argsort(torch.where(valid, hd, -torch.inf), descending=True, stable=True)
    slot = torch.arange(size, device=fit.device)
    return torch.where(slot < chosen.sum(), initial, rest[(slot - chosen.sum()).clamp_min(0)])


class HEA(Algorithm):
    """A tensorized implementation of the hyper-dominance evolutionary algorithm HEA.

    The algorithm filters candidates using hyper-dominance degree, then combines
    reference-direction representatives with degree-based filling. ``pop`` and
    ``fit`` expose the surviving solutions, while ``mating_pop`` stores the
    degree-refined parent population used by the next step. ``threshold`` increases
    from zero according to ``max_t`` and the nominal evaluation horizon.

    .. note::

        The current implementation accepts objective tensors of shape ``(B, n_objs)``;
        constraint tuples raise ``ValueError``. Uniform reference sampling may reduce the
        requested population size; read ``algorithm.pop_size`` for the actual count.
        Each step evaluates ``2 * floor(algorithm.pop_size / 2)`` offspring.

    :references:
        [1] Z. Liu, F. Han, Q. Ling, H. Han, and J. Jiang, "A many-objective optimization evolutionary algorithm
            based on hyper-dominance degree," Swarm and Evolutionary Computation, vol. 83, 101411, 2023.
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        max_t: float = 0.05,
        max_evaluations: int = 100000,
        device: torch.device | str | None = None,
    ):
        """Initialize HEA population and optimization state.

        :param pop_size: Required. Requested population size; uniform reference sampling sets the actual N.
        :type pop_size: int
        :param n_objs: Required. Number of minimized objectives, at least two; must match fitness columns.
        :type n_objs: int
        :param lb: Required. Lower bounds of shape ``(D,)``. Use floating tensors and match ``ub`` in
            shape, dtype and device; satisfy ``lb <= ub`` elementwise. Decision dimension is inferred.
        :type lb: torch.Tensor
        :param ub: Required. Upper bounds of shape ``(D,)``, matching ``lb``.
        :type ub: torch.Tensor
        :param max_t: Default: ``0.05``. Target hyper-dominance tolerance. The threshold starts at zero
            and increases by ``max_t * actual_pop_size / max_evaluations`` per step.
        :type max_t: float
        :param max_evaluations: Default: ``100000``. Evaluation horizon used to compute the threshold increment; includes
            initialization.
            It does not terminate the workflow.
        :type max_evaluations: int
        :param device: Default: ``None``. Optional device override. ``None`` preserves the bounds
            device; an explicit value copies both bounds to that device.
        :type device: torch.device, str or None

        .. note::

            Connect the algorithm through a workflow and call ``workflow.init_step()`` before stepping
            or compiling ``workflow.step``. The caller controls termination. Device and dtype follow
            the bounds unless an explicit device is supplied.

            The current implementation accepts objective tensors of shape ``(B, n_objs)``;
            constraint tuples raise ``ValueError``. Uniform reference sampling may reduce the
            requested population size; read ``algorithm.pop_size`` for the actual count.
            Each step evaluates ``2 * floor(algorithm.pop_size / 2)`` offspring.
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
        self.reference = ref.to(self.pop)
        self.max_t = max_t
        self.threshold = Mutable(self.pop.new_zeros(()))
        self.ideal = Mutable(self.pop.new_full((n_objs,), torch.inf))
        self.nadir = Mutable(self.pop.new_full((n_objs,), -torch.inf))
        self.mating_pop = Mutable(self.pop.clone())

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
        """Evaluate initial solutions and initialize the ideal, nadir and mating population.

        Invoke through the workflow before the first step or compilation.

        :returns: ``None``; population, fitness and auxiliary state are updated in place.
        """
        self.fit, self.cv = self._evaluate(self.pop)
        if self.cv is not None:
            raise ValueError("PlatEMO HEA supports objective-only evaluations; use a box-constrained problem")
        self.ideal = self.fit.amin(0)
        self.nadir = torch.maximum(self.fit.amax(0), self.ideal + 1e-6)
        self.mating_pop = self.pop.clone()

    def step(self):
        """Generate offspring and update hyper-dominance, reference selection and mating state.

        Invoke through the workflow after initialization; the caller controls termination.

        :returns: ``None``; population, fitness and auxiliary state are updated in place.
        """
        order = torch.randperm(self.pop_size, device=self.pop.device)
        parents = self.mating_pop.gather(0, order[:, None].expand(-1, self.dim))
        off = self._variation(parents)
        off_fit, _ = self._evaluate(off)
        pop, fit = torch.cat((off, self.pop)), torch.cat((off_fit, self.fit))
        self.ideal = torch.minimum(self.ideal, fit.amin(0))
        valid, hd = _domination(fit, self.ideal, self.nadir, self.threshold)
        self.nadir = torch.maximum(torch.where(valid[:, None], fit, -torch.inf).amax(0), self.ideal + 1e-6)
        selected = _selection(fit, hd, valid, self.ideal, self.nadir, self.reference)
        self.pop, self.fit = pop[selected], fit[selected]
        self.threshold = self.threshold + self.max_t * self.pop_size / self.max_evaluations
        rival = torch.randint(self.pop_size, (self.pop_size,), device=pop.device)
        selected_hd = hd[selected]
        self.mating_pop = self.pop[
            torch.where(selected_hd < selected_hd[rival], rival, torch.arange(self.pop_size, device=pop.device))
        ]
