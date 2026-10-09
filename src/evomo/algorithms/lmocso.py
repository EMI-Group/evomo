from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import ref_vec_guided
from evox.utils import clamp, randint

from evomo.operators.selection import ref_vec_guided as constrained_ref_vec_guided
from evomo.operators.selection.constraint_handling import (
    cat_violation,
    prefer_by_constraint,
    total_violation,
)
from evomo.utils import parse_evaluate, register_lazy_buffer


class LMOCSO(Algorithm):
    """
        The tensorized version of LMOCSO algorithm.

    :reference:
        [1] Ye Tian, Xiutao Zheng, Xingyi Zhang and Yaochu Jin, "Efficient Large-Scale Multiobjective Optimization Based
            on a Competitive Swarm," in IEEE Transactions on Cybernetics , 2020, pp. 3696 - 3708. Available:
            https://ieeexplore.ieee.org/document/8681243

    """

    def __init__(
        self,
        n_objs: int,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        alpha: float = 2.0,
        max_gen: int = 100,
        mutation_op: Optional[Callable] = None,
        selection_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initialize the LMOCSO population and optimization state.

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
        :param alpha: Default: ``2.0``. Exponent in the angle-penalty schedule ``(gen / max_gen) ** alpha`` used by
            reference-vector guided selection. A larger positive exponent delays the growth of angular selection
            pressure.
        :type alpha: float
        :param max_gen: Default: ``100``. Positive generation horizon used to normalize the algorithm's adaptation
            schedule. Set it consistently with your run length. It does not stop the workflow; the caller controls the
            loop.
        :type max_gen: int
        :param mutation_op: Default: ``None``. Mutation callable ``mutation_op(offspring, lb, ub) ->
            mutated_offspring``. Input and output are decision tensors of shape ``(B, D)``; preserve device and return
            an appropriate decision dtype. ``None`` selects EvoX's ``polynomial_mutation``.
        :type mutation_op: Callable or None
        :param selection_op: Default: ``None``. Environmental selection callable ``selection_op(pop, fit, vectors,
            theta) -> (pop, fit)`` for the merged population. Return matching decision and fitness tensors on the input
            device with the reference-vector output shape. ``None`` selects EvoX's ``ref_vec_guided``.
        :type selection_op: Callable or None
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
        """
        super().__init__()

        if device is None:
            device = torch.get_default_device()
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # write to self
        self.n_objs = n_objs
        self.pop_size = pop_size
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)
        self.alpha = Parameter(alpha)
        self.max_gen = max_gen

        self.selection = selection_op
        self.mutation = mutation_op

        if self.selection is None:
            self.selection = ref_vec_guided
        if self.mutation is None:
            self.mutation = polynomial_mutation

        reference_vector, _ = uniform_sampling(n=self.pop_size, m=self.n_objs)
        reference_vector = reference_vector.to(device=device)
        self.pop_size = reference_vector.size(0)

        population = torch.rand(self.pop_size, self.dim, device=device)
        population = population * (self.ub - self.lb) + self.lb
        self.pop = Mutable(population)
        self.velocity = Mutable(torch.zeros(self.pop_size // 2 * 2, self.dim, device=device))

        self.reference_vector = Mutable(reference_vector)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.gen = Mutable(torch.tensor(0, dtype=int, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")

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

        valid_mask = ~torch.isnan(self.pop).all(axis=1)
        num_valid = torch.sum(valid_mask, dtype=torch.int32)
        mating_pool = randint(0, num_valid, (self.pop_size,), device=self.pop.device)

        sorted_indices = torch.where(
            valid_mask,
            torch.arange(self.pop_size, device=self.pop.device),
            torch.iinfo(torch.int32).max,
        )
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        selected_pop = self.pop[sorted_indices[mating_pool]]
        selected_fit = self.fit[sorted_indices[mating_pool]]

        sde_fitness = self.cal_fitness(selected_fit)

        randperm = torch.randperm(self.pop_size // 2 * 2, device=self.pop.device).reshape(2, -1)

        mask = sde_fitness[randperm[0, :]] > sde_fitness[randperm[1, :]]
        if self.cv is not None:
            selected_cv = total_violation(self.cv)[sorted_indices[mating_pool]]
            mask = prefer_by_constraint(mask, selected_cv[randperm[0]], selected_cv[randperm[1]])
        winner = torch.where(mask, randperm[0, :], randperm[1, :])
        loser = torch.where(mask, randperm[1, :], randperm[0, :])

        r0 = torch.rand(self.pop_size // 2, self.dim, device=self.pop.device)
        r1 = torch.rand(self.pop_size // 2, self.dim, device=self.pop.device)

        off_velocity = r0 * self.velocity[loser] + r1 * (selected_pop[winner] - selected_pop[loser])
        new_loser_population = clamp(
            selected_pop[loser] + off_velocity + r0 * (off_velocity - self.velocity[loser]),
            self.lb,
            self.ub,
        )

        new_population = selected_pop.clone()
        new_population[loser] = new_loser_population
        new_velocity = self.velocity.clone()
        new_velocity[loser] = off_velocity
        self.velocity = new_velocity

        next_generation = self.mutation(new_population, self.lb, self.ub)
        next_generation_fitness, next_cv = parse_evaluate(self.evaluate(next_generation))

        self.gen = self.gen + 1

        merged_pop = torch.cat([self.pop, next_generation], dim=0)
        merged_fitness = torch.cat([self.fit, next_generation_fitness], dim=0)
        # RVEA Selection
        if self.cv is None:
            survivor, survivor_fitness = self.selection(
                merged_pop, merged_fitness, self.reference_vector, (self.gen / self.max_gen) ** self.alpha
            )
        else:
            selector = constrained_ref_vec_guided if self.selection is ref_vec_guided else self.selection
            survivor, survivor_fitness, self.cv = selector(
                merged_pop,
                merged_fitness,
                self.reference_vector,
                (self.gen / self.max_gen) ** self.alpha,
                cat_violation(self.cv, next_cv),
            )

        self.pop = survivor
        self.fit = survivor_fitness

    def cal_fitness(self, obj):
        """
        Calculate the fitness by shift-based density
        """
        n = obj.shape[0]
        f_max, _ = torch.max(obj, dim=0, keepdim=True)
        f_min, _ = torch.min(obj, dim=0, keepdim=True)
        f = (obj - f_min) / (f_max - f_min + 1e-10)
        s_obj = torch.maximum(f.unsqueeze(1), f.unsqueeze(0))
        dis = torch.norm(f.unsqueeze(1) - s_obj, p=2, dim=2)
        dis = dis + torch.diag(torch.full((n,), float("inf"), device=obj.device))
        fitness, _ = torch.min(dis, dim=1)

        return fitness
