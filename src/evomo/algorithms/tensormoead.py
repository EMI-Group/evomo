import math
from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable, vmap
from evox.operators.crossover import simulated_binary_half
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp


def pbi(f, w, z, z_max=None):
    norm_w = torch.norm(w, dim=1)
    f = f - z
    d1 = torch.sum(f * w, dim=1) / norm_w
    d2 = torch.norm(f - (d1[:, None] * w / norm_w[:, None]), dim=1)
    return d1 + 5 * d2


def tchebycheff(f, w, z, z_max=None):
    return torch.max(torch.abs(f - z) * w, dim=1)[0]


def tchebycheff_norm(f, w, z, z_max):
    span = z_max - z
    # Constant objectives contribute zero without introducing a division by zero.
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    normalized = torch.where(span > 0, torch.abs(f - z) / safe_span, 0)
    return torch.max(normalized * w, dim=1)[0]


def modified_tchebycheff(f, w, z, z_max=None):
    return torch.max(torch.abs(f - z) / w, dim=1)[0]


def weighted_sum(f, w, z=None, z_max=None):
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
        """Initialize tensorized MOEA/D with configurable aggregation for two update stages.

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
        :param aggregate_op: Default: ``('pbi', 'pbi')``. Two aggregation names: the first compares offspring with
            incumbents, and the second chooses among eligible offspring for a subproblem. Each name must be ``pbi``,
            ``tchebycheff``, ``tchebycheff_norm``, ``modified_tchebycheff`` or ``weighted_sum``. These are names, not
            callables.
        :type aggregate_op: tuple[str, str]
        :param selection_op: Default: ``None``. Accepted and stored, but not called by the current optimization step.
            Neighborhood parents are shuffled internally, so a custom value has no effect on selection.
        :type selection_op: Callable or None
        :param mutation_op: Default: ``None``. Mutation callable ``mutation_op(offspring, lb, ub) ->
            mutated_offspring``. Input and output are decision tensors of shape ``(B, D)``; preserve device and return
            an appropriate decision dtype. ``None`` selects EvoX's ``polynomial_mutation``.
        :type mutation_op: Callable or None
        :param crossover_op: Default: ``None``. Crossover callable ``crossover_op(parents) -> offspring``. For each pair
            of parents, return one child. The step passes ``2 * N`` parents and expects ``N`` offspring rows. Preserve
            the decision dimension and device. ``None`` selects EvoX's ``simulated_binary_half``.
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

            The sampled population size must be greater than 10, providing at least two neighborhood parents.
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

        length = self.ub - self.lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + self.lb

        neighbors = torch.cdist(w, w)
        self.neighbors = torch.argsort(neighbors, dim=1, stable=True)[:, : self.n_neighbor]
        self.w = w

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.z = Mutable(torch.zeros((self.n_objs,), device=device))
        self.z_max = Mutable(torch.zeros((self.n_objs,), device=device))

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
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0)[0]
        self.z_max = torch.max(self.fit, dim=0)[0]

    def step(self):
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        parent = shuffle_rows(self.neighbors)
        selected_p = torch.cat([self.pop[parent[:, 0]], self.pop[parent[:, 1]]], dim=0)

        crossovered = self.crossover(selected_p)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        self.z = torch.min(self.z, torch.min(off_fit, dim=0)[0])
        # Use a common objective range for all parent/offspring comparisons this generation.
        self.z_max = torch.maximum(self.fit.amax(dim=0), off_fit.amax(dim=0))

        sub_pop_indices = torch.arange(0, self.pop_size, device=self.pop.device)
        update_mask = torch.zeros((self.pop_size,), dtype=torch.bool, device=self.pop.device)

        def body(ind_p, ind_obj):
            g_old = self.aggregate_func1(self.fit[ind_p], self.w[ind_p], self.z, self.z_max)
            g_new = self.aggregate_func1(ind_obj, self.w[ind_p], self.z, self.z_max)
            temp_mask = update_mask.clone()
            temp_mask = torch.scatter(temp_mask, 0, ind_p, g_old > g_new)
            return torch.where(temp_mask, -1, sub_pop_indices.clone())

        replace_indices = vmap(body, in_dims=(0, 0))(self.neighbors, off_fit)

        def update_population(sub_indices, population, pop_obj, w_ind):
            f = torch.where(sub_indices[:, None] == -1, off_fit, pop_obj)
            x = torch.where(sub_indices[:, None] == -1, offspring, population)
            idx = torch.argmin(self.aggregate_func2(f, w_ind[None, :], self.z, self.z_max))
            return x[idx], f[idx]

        self.pop, self.fit = vmap(update_population, in_dims=(1, 0, 0, 0))(replace_indices, self.pop, self.fit, self.w)
