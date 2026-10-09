from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection
from evox.utils import clamp

from evomo.operators.selection.constraint_handling import constraint_priority, total_violation
from evomo.utils import parse_evaluate, register_lazy_buffer


def cal_max(pop_obj1, pop_obj2):
    """Calculates the maximum difference between elements of two objective tensors."""
    diff = pop_obj1.unsqueeze(1) - pop_obj2.unsqueeze(0)
    return torch.max(diff, dim=2)[0]


class IBEA(Algorithm):
    """
    The tensorized version of IBEA algorithm.

    :references:
        [1] E. Zitzler and S. Künzli, "Indicator-based selection in multiobjective search," in Proceedings of the International
            Conference on Parallel Problem Solving from Nature, 2004, pp. 832-842. Available:
            https://link.springer.com/chapter/10.1007/978-3-540-30217-9_84
    """

    def __init__(
        self,
        n_objs: int,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        kappa: float = 0.05,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initialize the IBEA population and optimization state.

        :param n_objs: Required. Number of objectives, matching the second dimension of the problem's objective tensor.
            Objectives are minimized; this library targets two or more objectives.
        :type n_objs: int
        :param pop_size: Required. Requested number of candidate solutions. Use a positive integer; algorithm-specific
            minimums and reference-vector sampling are described below.
        :type pop_size: int
        :param lb: Required. Lower decision bounds of shape ``(D,)``, where ``D`` is the number of decision variables.
            Use floating-point bounds with the same shape, dtype and device as ``ub``, and ``lb <= ub`` elementwise.
        :type lb: torch.Tensor
        :param ub: Required. Upper decision bounds of shape ``(D,)``. The decision dimension is inferred from the
            bounds, rather than passed separately. Match ``lb`` in shape, dtype and device.
        :type ub: torch.Tensor
        :param kappa: Default: ``0.05``. Positive scale in the exponential indicator-fitness calculation. Smaller values
            make indicator differences more influential. Objective scaling also affects the resulting selection
            pressure.
        :type kappa: float
        :param mutation_op: Default: ``None``. Mutation callable ``mutation_op(offspring, lb, ub) ->
            mutated_offspring``. Input and output are decision tensors of shape ``(B, D)``; preserve device and return
            an appropriate decision dtype. ``None`` selects EvoX's ``polynomial_mutation``.
        :type mutation_op: Callable or None
        :param crossover_op: Default: ``None``. Crossover callable ``crossover_op(parents) -> offspring`` receiving a
            two-dimensional decision tensor. ``None`` selects EvoX's ``simulated_binary``. Preserve the decision
            dimension and device, and produce the offspring count expected by this algorithm.
        :type crossover_op: Callable or None
        :param device: Default: ``None``. Execution device. ``None`` uses ``torch.get_default_device()``; it does not
            infer the device from the bounds. Bounds are copied to this device. Pass ``torch.device('cuda')`` explicitly
            for GPU execution.
        :type device: torch.device or None

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. Evaluation accepts objectives
            or ``(fitness, cv)`` with violations of shape ``(B,)`` or ``(B, C)``. The constrained extension prioritizes
            total positive violation in mating and removes higher-violation individuals first during truncation.
            Equal violations retain IBEA's indicator comparison. Selected violations are saved in ``self.cv``.
        """
        super().__init__()
        if device is None:
            device = torch.get_default_device()

        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]

        self.n_objs = n_objs
        self.pop_size = pop_size
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)
        self.kappa = Parameter(kappa)

        self.selection = tournament_selection
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary

        population = torch.rand(self.pop_size, self.dim, device=device)
        population = population * (self.ub - self.lb) + self.lb
        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        register_lazy_buffer(self, "cv", device_like="pop")

        self.next_generation = Mutable(self.pop.clone())

    def init_step(self):
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))

    def step(self):
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """

        fit, _, _ = self.cal_fitness(self.fit.clone().detach(), self.kappa)
        mating_fitness = -fit if self.cv is None else constraint_priority(-fit, self.cv)
        selected = self.selection(n_round=self.pop_size, fitness=mating_fitness)
        crossovered = self.crossover(self.pop[selected])
        next_generation = self.mutation(crossovered, self.lb, self.ub)
        next_generation = clamp(next_generation, self.lb, self.ub)
        self.next_generation = next_generation

        next_gen_fitness, next_gen_cv = parse_evaluate(self.evaluate(self.next_generation))

        merged_pop = torch.cat([self.pop, self.next_generation], dim=0)
        merged_obj = torch.cat([self.fit, next_gen_fitness], dim=0)
        merged_fitness, indicator_matrix, C = self.cal_fitness(merged_obj, self.kappa)

        n = merged_pop.size(0)
        next_ind = torch.arange(n, device=merged_pop.device)
        if self.cv is not None:
            merged_cv = torch.cat([self.cv, next_gen_cv], dim=0)
            total_cv = total_violation(merged_cv)

        for _ in range(self.pop_size):
            if self.cv is None:
                x = torch.argmin(merged_fitness)
            else:
                alive = next_ind < n
                worst_cv = torch.where(alive, total_cv, -torch.inf).amax()
                eligible = alive & (total_cv == worst_cv)
                x = torch.argmin(torch.where(eligible, merged_fitness, torch.inf))
            if self.cv is None:
                merged_fitness += torch.exp(-indicator_matrix[x] / C[x] / self.kappa)
                merged_fitness[x] = torch.max(merged_fitness)
                next_ind = next_ind.index_put((x,), torch.tensor(n, device=merged_pop.device))
            else:
                index = x.reshape(1)
                row = indicator_matrix.index_select(0, index).squeeze(0)
                scale = C.gather(0, index).squeeze(0)
                merged_fitness = merged_fitness + torch.exp(-row / scale / self.kappa)
                merged_fitness = merged_fitness.scatter(0, index, merged_fitness.amax().expand(1))
                next_ind = next_ind.scatter(0, index, n)

        next_ind = torch.argsort(next_ind, stable=True)
        next_ind = next_ind[: self.pop_size]

        survivor = merged_pop[next_ind]
        survivor_fitness = merged_obj[next_ind]

        self.pop = survivor
        self.fit = survivor_fitness
        if self.cv is not None:
            self.cv = merged_cv[next_ind]

    def cal_fitness(self, pop_obj, kappa):
        """Calculate the indicator-based fitness, indicator matrix, and scaling factor."""
        span = pop_obj.max(dim=0).values - pop_obj.min(dim=0).values
        if self.cv is not None:
            span = torch.where(span > 0, span, torch.ones_like(span))
        pop_obj_normalized = (pop_obj - pop_obj.min(dim=0).values) / span
        indicator_matrix = cal_max(pop_obj_normalized, pop_obj_normalized)
        C = torch.max(torch.abs(indicator_matrix), dim=0)[0]
        if self.cv is not None:
            C = torch.where(C > 0, C, torch.ones_like(C))
        fit = torch.sum(-torch.exp(-indicator_matrix / C.unsqueeze(0) / kappa), dim=0) + 1
        return fit, indicator_matrix, C
