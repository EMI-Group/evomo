import math
from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable, vmap
from evox.operators.crossover import simulated_binary_half
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp


def pbi(f, w, z):
    norm_w = torch.norm(w, dim=1)
    f = f - z
    d1 = torch.sum(f * w, dim=1) / norm_w
    d2 = torch.norm(f - (d1[:, None] * w / norm_w[:, None]), dim=1)
    return d1 + 5 * d2


def tchebycheff(f, w, z):
    return torch.max(torch.abs(f - z) * w, dim=1)[0]


def tchebycheff_norm(f, w, z, z_max):
    return torch.max(torch.abs(f - z) / (z_max - z) * w, dim=1)[0]


def modified_tchebycheff(f, w, z):
    return torch.max(torch.abs(f - z) / w, dim=1)[0]


def weighted_sum(f, w):
    return torch.sum(f * w, dim=1)


def shuffle_rows(matrix: torch.Tensor) -> torch.Tensor:
    """
    Shuffle each row of the given matrix independently without using a for loop.

    Args:
        matrix (torch.Tensor): A 2D tensor.

    Returns:
        torch.Tensor: A new tensor with each row shuffled differently.
    """
    rows, cols = matrix.size()

    permutations = torch.argsort(torch.rand(rows, cols, device=matrix.device), dim=1)
    return matrix.gather(1, permutations)

class TensorMOEAD(Algorithm):
    """
    TensorMOEA/D

    This is a tensorized implementation of the original MOEA/D algorithm, which incorporates GPU acceleration
    for improved computational performance in solving multi-objective optimization problems.

    :references:
        [1] Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition,"
            IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, 2007. Available:
            https://ieeexplore.ieee.org/document/4358754

        [2] Z. Liang, H. Li, N. Yu, K. Sun, and R. Cheng, "Bridging Evolutionary Multiobjective Optimization and
            GPU Acceleration via Tensorization," IEEE Transactions on Evolutionary Computation, 2025. Available:
            https://ieeexplore.ieee.org/document/10944658

    :note: This implementation differs from the original MOEA/D algorithm by incorporating tensorization for
            GPU acceleration, significantly improving performance for large-scale optimization tasks.
    """

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        aggregate_op=("pbi", "pbi"),
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initializes the TensorMOEA/D algorithm.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param aggregate_op: The aggregation function to use for the algorithm (optional).
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary` if not provided (optional).
        :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
        """

        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        device = torch.get_default_device() if device is None else device
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

        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary_half

        w, _ = uniform_sampling(self.pop_size, self.n_objs)
        w = w.to(device=device)

        self.pop_size = w.size(0)
        assert self.pop_size > 10, "Population size must be greater than 10. Please reset the population size."
        self.n_neighbor = int(math.ceil(self.pop_size / 10))

        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        neighbors = torch.cdist(w, w)
        self.neighbors = torch.argsort(neighbors, dim=1, stable=True)[:, : self.n_neighbor]
        self.w = w

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.z = Mutable(torch.zeros((self.n_objs,), device=device))

        self.aggregate_func1 = self.get_aggregation_function(aggregate_op[0])
        self.aggregate_func2 = self.get_aggregation_function(aggregate_op[1])

    def get_aggregation_function(self, name: str) -> Callable:
        aggregation_functions = {
            "pbi": pbi,
            "tchebycheff": tchebycheff,
            "tchebycheff_norm": tchebycheff_norm,
            "modified_tchebycheff": modified_tchebycheff,
            "weighted_sum": weighted_sum,
        }
        if name not in aggregation_functions:
            raise ValueError(f"Unsupported function: {name}")
        return aggregation_functions[name]

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0)[0]

    def step(self):
        """Perform the optimization step of the workflow."""
        parent = shuffle_rows(self.neighbors)
        selected_p = torch.cat([self.pop[parent[:, 0]], self.pop[parent[:, 1]]], dim=0)

        crossovered = self.crossover(selected_p)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        self.z = torch.min(self.z, torch.min(off_fit, dim=0)[0])

        sub_pop_indices = torch.arange(0, self.pop_size, device=self.pop.device)
        update_mask = torch.zeros((self.pop_size,), dtype=torch.bool, device=self.pop.device)

        def body(ind_p, ind_obj):
            g_old = self.aggregate_func1(self.fit[ind_p], self.w[ind_p], self.z)
            g_new = self.aggregate_func1(ind_obj, self.w[ind_p], self.z)
            temp_mask = update_mask.clone()
            temp_mask = torch.scatter(temp_mask, 0, ind_p, g_old > g_new)
            return torch.where(temp_mask, -1, sub_pop_indices.clone())

        replace_indices = vmap(body, in_dims=(0, 0))(self.neighbors, off_fit)

        def update_population(sub_indices, population, pop_obj, w_ind):
            f = torch.where(sub_indices[:, None] == -1, off_fit, pop_obj)
            x = torch.where(sub_indices[:, None] == -1, offspring, population)
            idx = torch.argmin(self.aggregate_func2(f, w_ind[None, :], self.z))
            return x[idx], f[idx]

        self.pop, self.fit = vmap(update_population, in_dims=(1, 0, 0, 0))(replace_indices, self.pop, self.fit, self.w)
