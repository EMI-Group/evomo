from typing import Callable, Optional

import torch
import torch.nn.functional as F
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, lexsort, nanmax, nanmin, randint

from evomo.operators.selection import non_dominate_rank, ref_vec_guided
from evomo.operators.selection.constraint_handling import total_violation
from evomo.utils import parse_evaluate, register_lazy_buffer


class RVEAa(Algorithm):
    """
    An implementation of the Reference Vector Guided Evolutionary Algorithm embedded with the reference vector
    regeneration strategy (RVEAa) for multi-objective optimization problems.

    This class is designed to solve multi-objective optimization problems using a reference vector guided evolutionary algorithm.

    :references:
        [1] R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, "A reference vector guided evolutionary algorithm
            for many-objective optimization," IEEE Transactions on Evolutionary Computation, vol. 20, no. 5,
            pp. 773-791, 2016. Available: https://ieeexplore.ieee.org/document/7386636
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
        """Initialize the RVEAa population and optimization state.

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
            tensors on the input device, preserving reference-vector slots and NaN padding where required. ``None``
            selects EvoMO's ``ref_vec_guided``. Constrained runs call ``selection_op(pop, fit, vectors, theta, cv)``
            and expect ``(pop, fit, cv)``.
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
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. Evaluation accepts objectives
            or ``(fitness, cv)`` with violations of shape ``(B,)`` or ``(B, C)``. First-front filtering jointly compares
            objectives and total violation. Each partition favors feasible APD winners, or minimum violation
            when no feasible solution exists. Final batch truncation also prioritizes violation. Selected violations
            and NaN padding are retained in ``self.cv``. Unconstrained selection and truncation remain unchanged.

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
        self.lb = lb.unsqueeze(0).to(device=device)
        self.ub = ub.unsqueeze(0).to(device=device)

        self.alpha = Parameter(alpha)
        self.fr = Parameter(fr)
        self.max_gen = Parameter(max_gen)

        self.rv_adapt_every = Mutable(torch.max(torch.round(1 / self.fr), torch.tensor(1.0)))

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

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
        v1 = torch.rand(self.pop_size, self.n_objs, device=device)
        v = torch.cat([v, v1], dim=0)

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        register_lazy_buffer(self, "cv", device_like="pop")
        self.reference_vector = Mutable(v)
        self.init_v = v0
        self.gen = Mutable(torch.tensor(0, dtype=int, device=device))

    def init_step(self):
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.rv_adapt_every = torch.max(torch.round(1 / self.fr), torch.tensor(1.0))
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))

    def _rv_adaptation(self, pop_obj: torch.Tensor):
        max_vals = nanmax(pop_obj, dim=0)[0]
        min_vals = nanmin(pop_obj, dim=0)[0]
        span = max_vals - min_vals
        if self.cv is not None:
            span = torch.where(span > 0, span, torch.ones_like(span))
        return self.init_v * span

    def _no_rv_adaptation(self, pop_obj: torch.Tensor):
        return self.reference_vector[: self.pop_size].clone()

    def _mating_pool(self):
        valid_mask = ~torch.isnan(self.pop).all(dim=1)
        num_valid = torch.sum(valid_mask, dtype=torch.int32)
        mating_pool = randint(0, num_valid, (self.pop_size,), device=self.pop.device)
        sorted_indices = torch.where(
            valid_mask, torch.arange(self.pop.size(0), device=self.pop.device), torch.iinfo(torch.int32).max
        )
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        pop = self.pop[sorted_indices[mating_pool]]
        return pop

    def _rv_regeneration(self, pop_obj: torch.Tensor, v: torch.Tensor):
        pop_obj = pop_obj - nanmin(pop_obj, dim=0).values
        cosine = F.cosine_similarity(pop_obj.unsqueeze(1), v.unsqueeze(0), dim=-1)

        mask = torch.isnan(cosine)
        input_tensor = torch.where(mask, -torch.inf, cosine)
        associate = input_tensor.max(dim=1, keepdim=False).indices
        associate = torch.where(input_tensor[:, 0] == -torch.inf, -1, associate)

        invalid = torch.sum((associate.unsqueeze(1) == torch.arange(v.size(0), device=pop_obj.device)), dim=0)
        span = nanmax(pop_obj, dim=0).values
        if self.cv is not None:
            span = torch.where(span > 0, span, torch.ones_like(span))
        rand = torch.rand((v.size(0), v.size(1)), device=pop_obj.device) * span
        new_v = torch.where((invalid == 0).unsqueeze(1), rand, v)

        return new_v

    def _batch_truncation(self, pop: torch.Tensor, obj: torch.Tensor):
        n = pop.size(0) // 2
        cosine = F.cosine_similarity(obj.unsqueeze(1), obj.unsqueeze(0), dim=-1)
        not_all_nan_rows = ~torch.isnan(cosine).all(dim=1)
        mask = torch.eye(cosine.size(0), dtype=torch.bool, device=pop.device) & not_all_nan_rows.unsqueeze(1)
        cosine = torch.where(mask, 0, cosine)

        sorted_values, _ = torch.sort(-cosine, dim=1)
        sorted_values = torch.where(torch.isnan(sorted_values[:, 0]), -torch.inf, sorted_values[:, 0])
        rank = torch.argsort(sorted_values)

        mask = torch.ones(rank.size(0), dtype=torch.bool, device=pop.device)
        mask = torch.where(
            torch.arange(rank.size(0), device=pop.device) < n, torch.tensor(0, dtype=torch.bool, device=pop.device), mask
        )
        mask = mask.unsqueeze(1)

        new_pop = torch.where(mask, pop, torch.nan)
        new_obj = torch.where(mask, obj, torch.nan)

        return new_pop, new_obj

    def _no_batch_truncation(self, pop: torch.Tensor, obj: torch.Tensor):
        return pop.clone(), obj.clone()

    def _constrained_keep_mask(self, obj, cv):
        """Keep up to N valid slots by violation, then angular batch diversity."""
        total = total_violation(cv)
        valid = torch.isfinite(obj).all(1) & torch.isfinite(total)
        cosine = F.cosine_similarity(obj[:, None, :], obj[None, :, :], dim=-1)
        pairs = valid[:, None] & valid[None, :] & ~torch.eye(len(obj), dtype=torch.bool, device=obj.device)
        crowding = torch.where(pairs, cosine, -torch.inf).amax(1)
        order = lexsort([crowding, torch.where(valid, total, torch.inf)])
        position = torch.empty_like(order).scatter(0, order, torch.arange(len(obj), device=obj.device))
        return valid & (position < self.pop_size)

    def _no_constrained_keep_mask(self, obj, cv):
        return torch.ones(len(obj), dtype=torch.bool, device=obj.device)

    def _constrained_batch_truncation(self, pop, obj, cv):
        keep = self._constrained_keep_mask(obj, cv)
        cv_mask = keep if cv.ndim == 1 else keep[:, None]
        return (
            torch.where(keep[:, None], pop, torch.nan),
            torch.where(keep[:, None], obj, torch.nan),
            torch.where(cv_mask, cv, torch.nan),
        )

    def _no_constrained_batch_truncation(self, pop, obj, cv):
        return pop.clone(), obj.clone(), cv.clone()

    def _update_pop_and_rv(self, survivor: torch.Tensor, survivor_fit: torch.Tensor, survivor_cv=None):
        v_regen = self._rv_regeneration(survivor_fit, self.reference_vector[self.pop_size :])
        if torch.compiler.is_compiling():
            v_adapt = torch.cond(
                self.gen % self.rv_adapt_every == 0, self._rv_adaptation, self._no_rv_adaptation, (survivor_fit,)
            )
            if self.cv is None:
                self.pop, self.fit = torch.cond(
                    self.gen == self.max_gen, self._batch_truncation, self._no_batch_truncation, (survivor, survivor_fit)
                )
            else:
                # A single mask output avoids a torch.cond/vmap tuple-unflatten bug.
                keep = torch.cond(
                    self.gen == self.max_gen,
                    self._constrained_keep_mask,
                    self._no_constrained_keep_mask,
                    (survivor_fit, survivor_cv),
                )
                cv_mask = keep if survivor_cv.ndim == 1 else keep[:, None]
                self.pop = torch.where(keep[:, None], survivor, torch.nan)
                self.fit = torch.where(keep[:, None], survivor_fit, torch.nan)
                self.cv = torch.where(cv_mask, survivor_cv, torch.nan)
        else:
            if self.gen % self.rv_adapt_every == 0:
                v_adapt = self._rv_adaptation(survivor_fit)
            else:
                v_adapt = self._no_rv_adaptation(survivor_fit)
            if self.cv is None:
                if self.gen == self.max_gen:
                    self.pop, self.fit = self._batch_truncation(survivor, survivor_fit)
                else:
                    self.pop, self.fit = self._no_batch_truncation(survivor, survivor_fit)
            else:
                if self.gen == self.max_gen:
                    self.pop, self.fit, self.cv = self._constrained_batch_truncation(survivor, survivor_fit, survivor_cv)
                else:
                    self.pop, self.fit, self.cv = self._no_constrained_batch_truncation(survivor, survivor_fit, survivor_cv)
        self.reference_vector = torch.cat([v_adapt, v_regen], dim=0)

    def step(self):
        """Perform a single optimization step."""

        self.gen = self.gen + 1
        pop = self._mating_pool()
        crossovered = self.crossover(pop)
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        off_fit, off_cv = parse_evaluate(self.evaluate(offspring))
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        if self.cv is None:
            rank = non_dominate_rank(merge_fit)
        else:
            merge_cv = torch.cat([self.cv, off_cv], dim=0)
            total = total_violation(merge_cv)
            valid = torch.isfinite(merge_fit).all(1) & torch.isfinite(total)
            # Treat CV as an extra objective: preserve objective/CV tradeoffs until
            # partition selection, rather than collapsing to the single best CV.
            joint = torch.cat([merge_fit, total[:, None]], dim=1)
            rank = non_dominate_rank(torch.where(valid[:, None], joint, torch.inf))
            rank = torch.where(valid, rank, -1)
        merge_fit = torch.where(rank.unsqueeze(1) == 0, merge_fit, torch.nan)
        merge_pop = torch.where(rank.unsqueeze(1) == 0, merge_pop, torch.nan)

        theta = (self.gen / self.max_gen) ** self.alpha
        if self.cv is None:
            survivor, survivor_fit = self.selection(merge_pop, merge_fit, self.reference_vector, theta)
            self._update_pop_and_rv(survivor, survivor_fit)
        else:
            cv_mask = rank == 0 if merge_cv.ndim == 1 else (rank == 0)[:, None]
            merge_cv = torch.where(cv_mask, merge_cv, torch.nan)
            survivor, survivor_fit, survivor_cv = self.selection(merge_pop, merge_fit, self.reference_vector, theta, merge_cv)
            self._update_pop_and_rv(survivor, survivor_fit, survivor_cv)
