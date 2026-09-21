from typing import Callable, Optional

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit

from evomo.operators.selection import non_dominate_rank
from evomo.operators.selection.non_dominate import _environmental_selection_rank


def _normalize(fit, mask=None):
    """Normalize objectives using ASF extremes, with a safe fallback for degenerate fronts."""
    if mask is None:
        mask = torch.ones(fit.shape[0], dtype=torch.bool, device=fit.device)
    ideal = torch.where(mask[:, None], fit, torch.inf).amin(0)
    shifted = torch.where(mask[:, None], fit - ideal, 0)
    m = fit.shape[1]
    weights = torch.full((m, m), 1e-6, device=fit.device, dtype=fit.dtype)
    weights.fill_diagonal_(1)
    asf = (shifted[None, :, :] / weights[:, None, :]).amax(2)
    extreme_idx = torch.where(mask[None, :], asf, torch.inf).argmin(1)
    extreme = shifted[extreme_idx]
    solution, info = torch.linalg.solve_ex(extreme, fit.new_ones(m), check_errors=False)
    intercept = solution.reciprocal()
    valid = (info == 0) & torch.isfinite(intercept).all() & (intercept > 0).all()
    residual = (extreme @ solution - 1).abs().amax()
    valid = valid & torch.isfinite(residual) & (residual < 1e-4)
    span = torch.where(valid, intercept, shifted.amax(0))
    span = torch.where(torch.isfinite(span) & (span > 0), span, torch.ones_like(span))
    return shifted / span


def _associate(fit, ref):
    """Associate with reference lines and evaluate squared residuals directly.

    Maximum absolute projection identifies the nearest line. Computing the
    selected residual avoids catastrophic cancellation in ||f||^2 - projection^2
    when comparing candidates very close to the same reference line.
    """
    unit = ref / ref.norm(dim=1, keepdim=True).clamp_min(torch.finfo(fit.dtype).tiny)
    projection = fit @ unit.T
    group = projection.abs().argmax(1)
    chosen = unit[group]
    along = (fit * chosen).sum(1, keepdim=True)
    distance = (fit - along * chosen).square().sum(1)
    return distance, group


def _niching(group, distance, rho, count, *, valid=None, complete=None, random_order=None, tie_order=None, nearest_order=None):
    """Return a fixed number of survivors, retaining complete fronts before niching.

    ``valid`` marks cutoff-front candidates; ``complete`` marks mandatory survivors.
    All arrays keep the merged population's shape to support fullgraph compilation.
    Optional permutations make deterministic reference tests possible.
    """
    c = group.numel()
    r = rho.numel()
    device = group.device
    if valid is None:
        valid = torch.ones(c, dtype=torch.bool, device=device)
    if complete is None:
        complete = torch.zeros(c, dtype=torch.bool, device=device)
    order = torch.randperm(c, device=device) if random_order is None else random_order
    # A random permutation gives uniform within-direction sampling without replacement.
    best_dist = torch.full((r,), torch.inf, device=device, dtype=distance.dtype)
    best_dist.scatter_reduce_(0, group, torch.where(valid, distance, torch.inf), reduce="amin", include_self=True)
    # Nearest-point ties must use randomness independent of the remaining queue.
    # Reusing its permutation biases the later positions of other tied minima.
    nearest = torch.randperm(c, device=device) if nearest_order is None else nearest_order
    positions = torch.empty_like(order)
    positions[nearest] = torch.arange(c, device=device)
    best_pos = torch.full((r,), c, device=device, dtype=torch.long)
    best_pos.scatter_reduce_(
        0, group, torch.where(valid & (distance == best_dist[group]), positions, c), reduce="amin", include_self=True
    )
    first = valid & (rho[group] == 0) & (positions == best_pos[group])
    # Keep invalid padding after all candidates within each direction.
    priority = torch.where(valid, (~first).long(), 2)
    order = order[torch.argsort(priority.gather(0, order), stable=True)]
    order = order[torch.argsort(group[order], stable=True)]
    sizes = torch.zeros(r, dtype=torch.long, device=device).scatter_add(0, group, torch.ones_like(group))
    starts = sizes.cumsum(0) - sizes
    local = torch.arange(c, device=device) - starts[group[order]]
    # Candidate k in direction j becomes available at occupancy rho[j] + k.
    # Random ties between equal levels implement uniform direction selection.
    level = rho[group[order]] + local
    level = torch.where(valid[order], level, 2 * c)
    level = torch.where(complete[order], -1, level)
    tie = torch.randperm(c, device=device) if tie_order is None else tie_order
    events = tie[torch.argsort(level.gather(0, tie), stable=True)][:count]
    return order[events]


class NSGA3(Algorithm):
    """
    An implementation of the tensorized NSGA-III for many-objective optimization problems.

    Uses rank-based tournament mating and fixed-shape reference-direction selection.
    Initialize the workflow before compiling its step with ``fullgraph=True``.

    :references:
        [1] K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
            Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints," IEEE Transactions on Evolutionary
            Computation, vol. 18, no. 4, pp. 577-601, 2014. Available: https://ieeexplore.ieee.org/document/6600851

        [2] H. Li, Z. Liang, and R. Cheng, "GPU-accelerated Evolutionary Many-objective Optimization Using Tensorized
            NSGA-III," in 2025 IEEE Congress on Evolutionary Computation, 2025.
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
        data_type: Optional[torch.dtype] = None,
        device: torch.device | None = None,
    ):
        """Initializes the NSGA-III algorithm.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary` if not provided (optional).
        :param data_type: The data type for the decision variables (optional). Defaults to torch.float32.
        :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
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

        if data_type == torch.bool:
            population = torch.rand(self.pop_size, self.dim, device=device)
            population = population > 0.5
        else:
            length = self.ub - self.lb
            population = torch.rand(self.pop_size, self.dim, device=device)
            population = length * population + self.lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.full((self.pop_size, self.n_objs), torch.inf, device=device))
        self.rank = Mutable(torch.full((self.pop_size,), torch.inf, device=device))
        self.ref = uniform_sampling(self.pop_size, self.n_objs)[0].to(device=device)

    def init_step(self):
        """
        Perform the initialization step of the workflow.

        Calls the `init_step` of the algorithm if overwritten; otherwise, its `step` method will be invoked.
        """
        self.fit = self.evaluate(self.pop)
        self.rank = non_dominate_rank(self.fit)

    def step(self):
        """Generate offspring and select survivors while preserving their true ranks."""
        mating = self.selection(self.pop_size, [self.rank])
        offspring = self.crossover(self.pop[mating])
        offspring = self.mutation(offspring, self.lb, self.ub)
        offspring = offspring.clamp(self.lb, self.ub)
        merged_pop = torch.cat((self.pop, offspring))
        merged_fit = torch.cat((self.fit, self.evaluate(offspring)))
        selected = self._survive(merged_fit)
        self.pop = merged_pop[selected]
        self.fit = merged_fit[selected]

    def _survive(self, fit):
        """Return survivor indices and update ranks without using ranks as selection flags."""
        rank = _environmental_selection_rank(fit, self.pop_size)
        cutoff = rank.kthvalue(self.pop_size).values
        complete = rank < cutoff
        last = rank == cutoff
        # Fixed-size masks avoid nonzero, data-dependent slices and host branches.
        normalized = _normalize(fit, rank <= cutoff)
        ref_order = torch.randperm(self.ref.shape[0], device=fit.device)
        # gather avoids a randperm-index pattern-matcher bug in PyTorch 2.11.
        ref = self.ref.gather(0, ref_order[:, None].expand_as(self.ref)).to(fit.dtype)
        distance, group = _associate(normalized, ref)
        rho = torch.zeros(ref.shape[0], dtype=torch.long, device=fit.device).scatter_add(0, group, complete.long())
        selected = _niching(group, distance, rho, self.pop_size, valid=last, complete=complete)
        self.rank = rank[selected]
        return selected
