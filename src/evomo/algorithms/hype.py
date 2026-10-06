from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


def cal_hv(fit: torch.Tensor, ref: torch.Tensor, pop_size: int, n_sample: int):
    n, m = fit.size()
    alpha = torch.cumprod(
        torch.cat(
            [
                torch.ones(1, device=fit.device),
                (pop_size - torch.arange(1, n, device=fit.device)) / (n - torch.arange(1, n, device=fit.device)),
            ]
        ),
        dim=0,
    ) / torch.arange(1, n + 1, device=fit.device)
    alpha = torch.nan_to_num(alpha)

    f_min = torch.min(fit, dim=0).values

    samples = torch.rand(n_sample, m, device=fit.device) * (ref - f_min) + f_min

    ds = torch.zeros(n_sample, dtype=torch.int64, device=fit.device)
    pds = (fit.unsqueeze(0).expand(n_sample, -1, -1) - samples.unsqueeze(1).expand(-1, n, -1) <= 0).all(dim=2)
    ds = torch.sum(torch.where(pds, ds.unsqueeze(1) + 1, ds.unsqueeze(1)), dim=1)
    ds = torch.where(ds == 0, ds, ds - 1)

    temp = torch.where(pds.T, ds.unsqueeze(0), -1)
    value = torch.where(temp != -1, alpha[temp], torch.tensor(0, dtype=torch.float32))
    f = torch.sum(value, dim=1)

    f = f * torch.prod(ref - f_min) / n_sample
    return f


class HypE(Algorithm):
    """
    The tensorized version of HypE algorithm.

    :references:
        [1] J. Bader and E. Zitzler, "HypE: An algorithm for fast hypervolume-based many-objective optimization,"
            Evolutionary Computation, vol. 19, no. 1, pp. 45-76, 2011. Available:
            https://direct.mit.edu/evco/article-abstract/19/1/45/1363/HypE-An-Algorithm-for-Fast-Hypervolume-Based-Many

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
        n_sample: int = 10000,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initialize the HypE population and optimization state.

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
        :param n_sample: Default: ``10000``. Positive number of Monte Carlo samples for each hypervolume contribution
            estimate. More samples reduce sampling noise at higher computation and memory cost.
        :type n_sample: int
        :param selection_op: Default: ``None``. Accepted in the signature, but the current constructor overwrites it
            with EvoX's ``tournament_selection``. A custom value has no effect.
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
        self.n_sample = n_sample

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        self.selection = tournament_selection
        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary

        length = self.ub - self.lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + self.lb

        self.ref = Mutable(torch.ones(n_objs, device=device))

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))

    def init_step(self):
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit = self.evaluate(self.pop)
        self.ref = torch.full((self.n_objs,), torch.max(self.fit).item() * 1.2, device=self.fit.device)

    def step(self):
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        hv = cal_hv(self.fit, self.ref, self.pop_size, self.n_sample)
        mating_pool = self.selection(self.pop_size, -hv)
        crossovered = self.crossover(self.pop[mating_pool])
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit = self.evaluate(offspring)

        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        rank = non_dominate_rank(merge_fit)
        order = torch.argsort(rank)
        worst_rank = rank[order[self.pop_size - 1]]
        mask = rank <= worst_rank

        hv = cal_hv(merge_fit, self.ref, torch.sum(mask) - self.pop_size, self.n_sample)
        dis = torch.where(mask, hv, -torch.inf)

        combined_indices = lexsort([-dis, rank])[: self.pop_size]

        self.pop = merge_pop[combined_indices]
        self.fit = merge_fit[combined_indices]
