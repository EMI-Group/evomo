from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, nanmax, nanmin, randint

from evomo.operators.selection import ref_vec_guided


class RVEA(Algorithm):
    """
    A tensorized implementation of the Reference Vector Guided Evolutionary Algorithm (RVEA)
    for multi-objective optimization problems.

    :references:
        [1] R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, "A reference vector guided evolutionary algorithm
            for many-objective optimization," IEEE Transactions on Evolutionary Computation, vol. 20, no. 5,
            pp. 773-791, 2016. Available: https://ieeexplore.ieee.org/document/7386636

        [2] Z. Liang, T. Jiang, K. Sun, and R. Cheng, "GPU-accelerated Evolutionary Multiobjective Optimization
            Using Tensorized RVEA," in Proceedings of the Genetic and Evolutionary Computation Conference,
            ser. GECCO ’24, 2024, pp. 566–575. Available: https://doi.org/10.1145/3638529.3654223
"""

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        alpha: float = 2.0,
        fr: float = 0.1,
        max_gen: int = 100,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initialize the RVEA population and optimization state.

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
        :param alpha: Default: ``2.0``. Exponent in the angle-penalty schedule ``(gen / max_gen) ** alpha`` used by
            reference-vector guided selection. A larger positive exponent delays the growth of angular selection
            pressure.
        :type alpha: float
        :param fr: Default: ``0.1``. Positive reference-vector adaptation rate. The current implementation adapts every
            ``max(round(1 / fr), 1)`` steps: ``0.1`` means every 10 steps. The interval is independent of ``max_gen``.
        :type fr: float
        :param max_gen: Default: ``100``. Positive generation horizon used to normalize the algorithm's adaptation
            schedule. Set it consistently with your run length. It does not stop the workflow; the caller controls the
            loop.
        :type max_gen: int
        :param selection_op: Default: ``None``. Environmental selection callable ``selection_op(pop, fit, vectors,
            theta) -> (pop, fit)``. The input includes parents and offspring. Return matching decision and fitness
            tensors on the input device; preserve reference-vector slots, including NaN padding where required. ``None``
            selects EvoMO's ``ref_vec_guided``.
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
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            expects an objective tensor of shape ``(B, n_objs)``. It does not consume a ``(fitness,
            constraint_violation)`` tuple.

            Reference-vector sampling can change the requested population size. Read ``self.pop.shape[0]`` for the actual size.
        """
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        device = torch.get_default_device() if device is None else device
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.size(0)
        # write to self
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        self.alpha = Parameter(alpha)
        self.fr = Parameter(fr)
        self.max_gen = Parameter(max_gen)

        self.rv_adapt_every = Mutable(torch.max(torch.round(1 / self.fr), torch.tensor(1.0)))

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op
        self.device = device

        if self.selection is None:
            self.selection = ref_vec_guided
        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary
        sampling, _ = uniform_sampling(self.pop_size, self.n_objs)

        v = sampling.to(device=device)

        v0 = v.clone()
        self.pop_size = v.size(0)
        length = self.ub - self.lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + self.lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.reference_vector = Mutable(v)
        self.init_v = v0
        self.gen = Mutable(torch.tensor(0, dtype=int, device=device))

    def init_step(self):
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.rv_adapt_every = torch.max(torch.round(1 / self.fr), torch.tensor(1.0))
        self.fit = self.evaluate(self.pop)

    def _rv_adaptation(self, pop_obj: torch.Tensor):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        return self.init_v.clone() * (max_vals - min_vals)

    def _no_rv_adaptation(self, pop_obj: torch.Tensor):
        return self.reference_vector.clone()

    def _mating_pool(self):
        valid_mask = ~torch.isnan(self.pop).all(dim=1)
        num_valid = torch.sum(valid_mask, dtype=torch.int32)
        mating_pool = randint(0, num_valid, (self.pop_size,), device=self.pop.device)
        sorted_indices = torch.where(valid_mask, torch.arange(self.pop_size, device=self.device), torch.iinfo(torch.int32).max)
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        pop = self.pop[sorted_indices[mating_pool]]
        return pop

    def _update_pop_and_rv(self, survivor, survivor_fit):
        self.pop = survivor
        self.fit = survivor_fit

        if torch.compiler.is_compiling():
            self.reference_vector = torch.cond(
                self.gen % self.rv_adapt_every == 0,
                self._rv_adaptation,
                self._no_rv_adaptation,
                (survivor_fit,)
            )
        else:
            if (self.gen % self.rv_adapt_every) == 0:
                self.reference_vector = self._rv_adaptation(survivor_fit)
            else:
                self.reference_vector = self._no_rv_adaptation(survivor_fit)

    def step(self):
        """Perform a single optimization step."""

        self.gen = self.gen + 1
        pop = self._mating_pool()
        crossovered = self.crossover(pop)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        survivor, survivor_fit = self.selection(
            merge_pop,
            merge_fit,
            self.reference_vector,
            (self.gen / self.max_gen) ** self.alpha,
        )

        self._update_pop_and_rv(survivor, survivor_fit)
