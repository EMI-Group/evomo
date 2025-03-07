import torch
from typing import Callable, Optional

from evox.operators import selection, mutation, crossover
from evox.core import Algorithm, Mutable,Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import non_dominate_rank, tournament_selection
from evox.utils import clamp


def cal_max(pop_obj1, pop_obj2):
    """Calculates the maximum difference between elements of two objective tensors."""
    n1 = pop_obj1.shape[0]
    n2 = pop_obj2.shape[0]
    m = pop_obj1.shape[1]
    diff = pop_obj1.unsqueeze(1) - pop_obj2.unsqueeze(0)
    return torch.max(diff, dim=2)[0]


class IBEA(Algorithm):
    """The tensoried version of IBEA algorithm.

        IBEA algorithm is described in the following papers:

        Title: Indicator-Based Selection in Multiobjective Search
        Link: https://link.springer.com/chapter/10.1007/978-3-540-30217-9_84

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
        """Initializes the IBEA algorithm.

        :param n_objs: The number of objective functions in the optimization problem.
        :param pop_size: The size of the population.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param kappa: The scaling factor for fitness calculation in IBEA (optional, defaults to 0.05).
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary` if not provided (optional).
        :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
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
        population=population * (self.ub - self.lb) + self.lb
        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.next_generation = Mutable(self.pop.clone())

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop) 

    def step(self):
        """Perform the optimization step of the workflow."""
        
        fit, _, _ = self.cal_fitness(
            self.fit.clone().detach(), self.kappa
        )
        selected = self.selection(n_round=self.pop_size,fitness=-fit)
        crossovered = self.crossover(self.pop[selected])
        next_generation = self.mutation(crossovered, self.lb, self.ub)
        next_generation = clamp(next_generation, self.lb, self.ub)
        self.next_generation = next_generation

        
        next_gen_fitness = self.evaluate(self.next_generation)

        merged_pop = torch.cat([self.pop, self.next_generation], dim=0)
        merged_obj = torch.cat([self.fit, next_gen_fitness], dim=0)
        merged_fitness, I, C = self.cal_fitness(
            merged_obj, self.kappa
        )  

        n = merged_pop.shape[0]
        next_ind = torch.arange(n)

        for _ in range(self.pop_size):
            x = torch.argmin(merged_fitness)
            merged_fitness += torch.exp(-I[x, :] / C[x] / self.kappa)
            merged_fitness[x] = torch.max(merged_fitness)
            next_ind[x] = -1

        next_ind = next_ind[next_ind != -1]
        next_ind = next_ind[: self.pop_size]

        survivor = merged_pop[next_ind]
        survivor_fitness = merged_obj[next_ind]


        self.pop = survivor
        self.fit = Mutable(survivor_fitness)

    def cal_fitness(self, pop_obj, kappa):
        """Calculate the indicator-based fitness, indicator matrix, and scaling factor."""       
        n = pop_obj.shape[0]
        pop_obj_normalized = (pop_obj - pop_obj.min(dim=0).values) / (
            pop_obj.max(dim=0).values - pop_obj.min(dim=0).values
        )
        I = cal_max(pop_obj_normalized, pop_obj_normalized)
        C = torch.max(torch.abs(I), dim=0)[0]
        fit = torch.sum(-torch.exp(-I / C.unsqueeze(0) / kappa), dim=0) + 1
        return fit, I, C