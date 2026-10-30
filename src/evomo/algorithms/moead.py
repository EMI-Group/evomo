import math
from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary_half
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, minimum

from evomo.operators.selection.constraint_handling import prefer_by_constraint, total_violation, update_by_proposals
from evomo.utils import parse_evaluate, register_lazy_buffer


def pbi(f: torch.Tensor, w: torch.Tensor, z: torch.Tensor):
    norm_w = torch.linalg.norm(w, dim=1)
    f = f - z

    d1 = torch.sum(f * w, dim=1) / norm_w

    d2 = torch.linalg.norm(f - (d1[:, None] * w / norm_w[:, None]), dim=1)
    return d1 + 5 * d2


class MOEAD(Algorithm):
    """
    Implementation of the Original MOEA/D algorithm.

    :references:
        [1] Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition,"
            IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, 2007. Available:
            https://ieeexplore.ieee.org/document/4358754

    :note: This implementation is based on the original paper and may not be the most efficient implementation.
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
        """Initialize the MOEAD population and optimization state.

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
        :param selection_op: Default: ``None``. Accepted and stored, but not called by the current optimization step.
            Mating uses random neighborhood parents internally, so a custom value does not change mating selection.
        :type selection_op: Callable or None
        :param mutation_op: Default: ``None``. Mutation callable ``mutation_op(offspring, lb, ub) ->
            mutated_offspring``. Input and output are decision tensors of shape ``(B, D)``; preserve device and return
            an appropriate decision dtype. ``None`` selects EvoX's ``polynomial_mutation``.
        :type mutation_op: Callable or None
        :param crossover_op: Default: ``None``. Crossover callable ``crossover_op(parents) -> offspring``. For each pair
            of parents, return one child. The step passes two parents and expects an offspring batch with one row.
            Preserve the decision dimension and device. ``None`` selects EvoX's ``simulated_binary_half``.
        :type crossover_op: Callable or None
        :param device: Default: ``None``. Execution device. ``None`` uses ``torch.get_default_device()``; it does not
            infer the device from the bounds. Bounds are copied to this device. Pass ``torch.device('cuda')`` explicitly
            for GPU execution.
        :type device: torch.device or None

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            accepts an objective tensor or a ``(fitness, constraint_violation)`` tuple.
            Constrained evaluations retain violations through selection and state updates.

            Reference-vector sampling can change the requested population size. Read ``self.pop.shape[0]`` for the actual size.

            The sampled population size must be greater than 10, providing at least two neighborhood parents.
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
        self.device = device

        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary_half

        w, _ = uniform_sampling(self.pop_size, self.n_objs)
        w = w.to(device=device)

        self.pop_size = w.size(0)
        self.n_neighbor = int(math.ceil(self.pop_size / 10))

        length = self.ub - self.lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + self.lb

        neighbors = torch.cdist(w, w)
        self.neighbors = torch.argsort(neighbors, dim=1, stable=True)[:, : self.n_neighbor]
        self.w = w

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.z = Mutable(torch.zeros((self.n_objs,), device=device))

        register_lazy_buffer(self, "cv", device_like="pop")

    def init_step(self):
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        self.z = torch.min(self.fit, dim=0)[0]

    def step(self):
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        for i in range(self.pop_size):
            parents = self.neighbors[i][torch.randperm(self.n_neighbor, device=self.device)]
            crossovered = self.crossover(self.pop[parents[:2]])
            offspring = self.mutation(crossovered, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)
            off_fit, off_cv = parse_evaluate(self.evaluate(offspring))

            self.z = minimum(self.z, off_fit)

            g_old = pbi(self.fit[parents], self.w[parents], self.z)
            g_new = pbi(off_fit, self.w[parents], self.z)

            if self.cv is not None:
                better = prefer_by_constraint(g_new <= g_old, total_violation(off_cv), total_violation(self.cv)[parents])
                self.pop, self.fit, self.cv = update_by_proposals(
                    self.pop, self.fit, self.cv, offspring, off_fit, off_cv, parents[None], better[None], g_new[None]
                )
            else:
                self.fit[parents[g_old >= g_new]] = off_fit
                self.pop[parents[g_old >= g_new]] = offspring
