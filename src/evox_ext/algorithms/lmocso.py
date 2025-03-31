from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import ref_vec_guided
from evox.utils import clamp


class LMOCSO(Algorithm):
    """
    # 1. LMOCSO algorithm is described in the following papers:
    #
    # Title: Efficient Large-Scale Multiobjective Optimization Based on a Competitive Swarm
    # Optimizer
    # Link: https://ieeexplore.ieee.org/document/8681243

    # 2. This code has been inspired by PlatEMO.
    # More information about PlatEMO can be found at the following URL:
    # GitHub Link: https://github.com/BIMK/PlatEMO
    """

    def __init__(
        self,
        n_objs: int,
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        alpha: float = 2.0,
        max_gen: int=100,
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
        self.device = device
        self.max_gen = max_gen

        self.selection = selection_op
        self.mutation = mutation_op

        if self.selection is None:
            self.selection = ref_vec_guided
        if self.mutation is None:
            self.mutation = polynomial_mutation

        population = torch.rand(self.pop_size, self.dim, device=device)
        population = population * (self.ub - self.lb) + self.lb
        self.pop = Mutable(population)
        self.velocity = Mutable(torch.zeros(self.pop_size // 2 * 2, self.dim, device=device))
        reference_vector,_ = uniform_sampling(n=self.pop_size, m=self.n_objs)
        self.reference_vector = Mutable(reference_vector)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.gen = 0
        self.next_generation = Mutable(self.pop.clone())
        self.generator = torch.Generator(device=device)

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit=self.evaluate(self.pop)


    def step(self):
        """Perform the optimization step of the workflow."""
        no_nan_pop = ~torch.isnan(self.pop).all(axis=1)
        valid_indices = torch.where(no_nan_pop)[0]
        max_idx = valid_indices.shape[0]

        if max_idx < 2:
            new_population = self.pop.clone()
            return

        mating_pool = torch.randint(0, max_idx, (self.pop_size,), generator=self.generator, device=self.device)
        selected_pop = self.pop[valid_indices][mating_pool]

        sde_fitness = self.cal_fitness(self.fit)

        randperm_size = max_idx // 2 * 2
        randperm = torch.randperm(randperm_size, device=self.device, generator=self.generator).reshape(2, -1)

        mask = sde_fitness[randperm[0, :]] > sde_fitness[randperm[1, :]]
        winner = torch.where(mask, randperm[0, :], randperm[1, :])
        loser = torch.where(mask, randperm[1, :], randperm[0, :])

        r0 = torch.rand(randperm_size // 2, self.dim, device=self.device)
        r1 = torch.rand(randperm_size // 2, self.dim, device=self.device)


        valid_loser_indices = valid_indices[loser]
        off_velocity = r0 * self.velocity[valid_loser_indices] + r1 * (
            selected_pop[winner] - selected_pop[loser]
        )
        new_loser_population = torch.clip(
            selected_pop[loser]
            + off_velocity
            + r0 * (off_velocity - self.velocity[valid_loser_indices]),
            self.lb,
            self.ub,
        )


        new_population = self.pop.clone()
        new_population[valid_loser_indices] = new_loser_population
        new_velocity = self.velocity.clone()
        new_velocity[valid_loser_indices] = off_velocity
        self.velocity = new_velocity

        next_generation = self.mutation(new_population, self.lb, self.ub)
        next_generation = clamp(next_generation, self.lb, self.ub)
        self.next_generation=next_generation

        current_gen = self.gen + 1
        v = self.reference_vector

        merged_pop = torch.cat([self.pop, self.next_generation], dim=0)
        next_generation_fitness = self.evaluate(self.next_generation)
        merged_fitness = torch.cat([self.fit, next_generation_fitness], dim=0)

        # RVEA Selection
        survivor, survivor_fitness = self.selection(
            merged_pop, merged_fitness, v, (current_gen / self.max_gen) ** self.alpha
        )
        self.pop=survivor
        self.fit=survivor_fitness
        self.gen = current_gen


    def cal_fitness(self, obj):
        """
        Calculate the fitness by shift-based density
        """
        n = obj.size(0)
        f_max = torch.max(obj, dim=0)[0]
        f_min = torch.min(obj, dim=0)[0]
        f = (obj - f_min) / (f_max - f_min)

        dis = torch.cdist(f, f)

        dis = dis.masked_fill(torch.eye(n, device=obj.device).bool(), float('inf'))

        fitness = torch.min(dis, dim=1)[0]

        return fitness
