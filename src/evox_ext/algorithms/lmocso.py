from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import ref_vec_guided
from evox.utils import clamp, randint


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
        """
        Initializes the LMOCSO algorithm.

        :param n_objs: The number of objective functions in the optimization problem.
        :param pop_size: The size of the population.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param alpha: The parameter controlling the rate of change of penalty in RVEA selection. Defaults to 2.0.
        :param max_gen: The maximum number of generations for the optimization process. Defaults to 100.
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param selection_op: The selection operation, defaults to `ref_vec_guided` (Reference Vector Guided Selection) if not provided (optional).
        :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
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

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)

    def step(self):
        """Perform the optimization step of the workflow."""

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
        next_generation_fitness = self.evaluate(next_generation)

        self.gen = self.gen + 1

        merged_pop = torch.cat([self.pop, next_generation], dim=0)
        merged_fitness = torch.cat([self.fit, next_generation_fitness], dim=0)
        # RVEA Selection
        survivor, survivor_fitness = self.selection(merged_pop, merged_fitness, self.reference_vector,
                                                    (self.gen / self.max_gen) ** self.alpha)

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
        s_obj = torch.maximum(
            f.unsqueeze(1),
            f.unsqueeze(0)
        )
        dis = torch.norm(
            f.unsqueeze(1) - s_obj,
            p=2, dim=2
        )
        dis = dis + torch.diag(torch.full((n,), float('inf'), device=obj.device))
        fitness, _ = torch.min(dis, dim=1)

        return fitness

