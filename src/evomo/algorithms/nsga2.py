from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection import nd_environmental_selection
from evomo.utils import parse_evaluate, register_lazy_buffer


class NSGA2(Algorithm):
    """
    A tensorized implementation of the Non-dominated Sorting Genetic Algorithm II (NSGA-II)
    for multi-objective optimization problems.

    :references:
        [1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II,"
            IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
            Available: https://ieeexplore.ieee.org/document/996017

        [2] Z. Liang, H. Li, N. Yu, K. Sun, and R. Cheng, "Bridging Evolutionary Multiobjective Optimization and
            GPU Acceleration via Tensorization," IEEE Transactions on Evolutionary Computation, 2025. Available:
            https://ieeexplore.ieee.org/document/10944658
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initialize NSGA-II with rank and crowding-distance selection.

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
        :param selection_op: Default: ``None``. Mating selection callable ``selection_op(pop_size, keys) -> indices``.
            ``keys`` is a list of one-dimensional sort-key tensors for crowding distance, rank and, when present,
            constraints. Return ``pop_size`` parent indices on the population device. ``None`` selects EvoX's
            ``tournament_selection_multifit``. This does not replace environmental selection.
        :type selection_op: Callable or None
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
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. Evaluation accepts an objective
            tensor of shape ``(B, n_objs)`` or ``(fitness, constraint_violation)``; violations are retained in
            selection.
        """

        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        if device is None:
            device = torch.get_default_device()
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # write to self
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = tournament_selection_multifit
        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary

        length = self.ub - self.lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + self.lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.rank = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.dis = Mutable(torch.empty(self.pop_size, device=device).fill_(-torch.inf))
        # The number of constraints is known only after the first evaluation.
        register_lazy_buffer(self, "cv", device_like="pop")

    def init_step(self):
        """Perform the initialization step of the workflow."""
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        self.pop, self.fit, self.rank, self.dis, self.cv = nd_environmental_selection(
            self.pop, self.fit, self.pop_size, self.cv
        )

    def step(self):
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        sort_keys = [-self.dis, self.rank]
        if self.cv is not None:
            sort_keys.append(self.cv.sum(dim=1) if self.cv.ndim > 1 else self.cv)

        mating_pool = self.selection(self.pop_size, sort_keys)
        crossovered = self.crossover(self.pop[mating_pool])
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        off_fit, off_cv = parse_evaluate(self.evaluate(offspring))

        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)
        merge_cv = torch.cat([self.cv, off_cv], dim=0) if self.cv is not None else None

        self.pop, self.fit, self.rank, self.dis, self.cv = nd_environmental_selection(
            merge_pop, merge_fit, self.pop_size, merge_cv
        )
