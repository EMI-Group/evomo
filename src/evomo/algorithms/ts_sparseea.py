import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary_half
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    rank_with_constraint_objective,
)
from evomo.utils import register_lazy_buffer

from .sparseea import _evaluate_sparse, _select_sparse, _sparse_offspring_count


class TSSparseEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the TSSparseEA population and optimization state.

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
        :param kwargs: Optional ``max_fe`` bounds actual evaluations, including probes and initialization.
            Place bounds on the intended device; ``device=...`` does not move them.
            ``r_eva`` sets the first-stage evaluation fraction (default ``0.1``).
        :type kwargs: dict

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            accepts an objective tensor or a ``(fitness, constraint_violation)`` tuple.
            Constrained evaluations retain violations through selection and state updates.

            Tensor allocation uses ``lb.device``. There is no explicit device parameter; prepare both bounds on the intended
            device.
        """
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()
        self.dim = D

        # Stage 1 Grouping Parameters
        self.nGroup = 50
        group_size = (D + self.nGroup - 1) // self.nGroup
        sentinel = torch.iinfo(torch.int32).max

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.mask = Mutable(torch.ones((pop_size, D), dtype=torch.bool, device=device))
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.rank = Mutable(torch.zeros(pop_size, dtype=torch.int32, device=device))
        self.dis = Mutable(torch.zeros(pop_size, device=device))
        self.is_initialized = Mutable(torch.tensor(False, device=device))
        self.subcomponents = Mutable(torch.full((self.nGroup, group_size), sentinel, dtype=torch.int64, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        self.max_fe = kwargs.get("max_fe")
        self.stage1_fraction = kwargs.get("r_eva", 0.1)
        self.fe = Mutable(torch.tensor(0, dtype=torch.int64, device=device))
        self.var_fit = Mutable(self.pop.new_zeros(D))

    def _match_operator(self, off_mask: torch.Tensor, off_dec: torch.Tensor) -> torch.Tensor:
        # Match generated decisions to masks without reusing a decision, as in PlatEMO's Match.
        scale = (self.ub - self.lb).clamp_min(torch.finfo(off_dec.dtype).tiny)
        normalized = (off_dec - self.lb) / scale
        mask_f = off_mask.to(off_dec.dtype)
        dot = mask_f @ normalized.t()
        norm_m = torch.norm(mask_f, dim=1, keepdim=True)
        norm_p = torch.norm(normalized, dim=1, keepdim=True)
        sim = dot / (norm_m @ norm_p.t()).clamp_min(torch.finfo(off_dec.dtype).tiny)
        count = off_mask.shape[0]
        order = torch.randperm(count, device=off_dec.device)
        remaining = torch.ones(count, dtype=torch.bool, device=off_dec.device)
        selected = torch.empty(count, dtype=torch.long, device=off_dec.device)
        for i in range(count):
            row = order[i : i + 1]
            scores = sim.index_select(0, row).squeeze(0)
            best = torch.argmax(torch.where(remaining, scores, -torch.inf)).reshape(1)
            selected.scatter_(0, row, best)
            remaining.scatter_(0, best, False)
        return off_dec[selected]

    def init_step(self) -> None:
        self.mask = torch.rand_like(self.pop) < 0.5
        self.fit, self.cv = _evaluate_sparse(self, self.pop * self.mask)
        number = min(self.pop_size, self.n_objs + 1)
        _, reference_masks, _, _, _, _ = _select_sparse(self.pop, self.mask, self.fit, self.cv, number)
        references = reference_masks.to(self.pop.dtype)
        fitness = self.pop.new_zeros(self.dim)
        for _ in range(5):
            probe_dec = self.lb + torch.rand(self.dim, self.dim, dtype=self.pop.dtype, device=self.pop.device) * (
                self.ub - self.lb
            )
            probe_mask = torch.eye(self.dim, dtype=torch.bool, device=self.pop.device)
            probe_fit, probe_cv = _evaluate_sparse(self, probe_dec * probe_mask)
            fitness = fitness + rank_with_constraint_objective(probe_fit, probe_cv).to(fitness.dtype)
        self.var_fit = fitness
        order = fitness.argsort(stable=True)
        width = self.subcomponents.shape[1]
        padded = torch.full((self.nGroup * width,), torch.iinfo(torch.int32).max, dtype=torch.long, device=self.pop.device)
        padded[: self.dim] = order
        self.subcomponents = padded.reshape(self.nGroup, width)
        membership = (self.subcomponents[:, :, None] == torch.arange(self.dim, device=self.pop.device)[None, None]).any(1)
        horizon = 10000 if self.max_fe is None else self.max_fe
        iterations = max(0, int((self.stage1_fraction * horizon - int(self.fe)) / (self.pop_size * number)))
        for reference in references:
            groups = torch.rand(self.pop_size, self.nGroup, device=self.pop.device) < 0.5
            groups[0] = False
            groups[0, 0] = True
            mask = (groups[:, :, None] & membership[None]).any(1)
            fit, cv = _evaluate_sparse(self, reference[None] * mask)
            _, groups, fit, cv, rank, distance = _select_sparse(groups, groups, fit, cv, self.pop_size)
            for _ in range(iterations):
                mating = tournament_selection_multifit(self.pop_size, [-distance, rank.float()], tournament_size=2)
                off_groups = self._binary_operator(groups[mating])
                mask = (off_groups[:, :, None] & membership[None]).any(1)
                off_fit, off_cv = _evaluate_sparse(self, reference[None] * mask)
                combined = torch.cat([groups, off_groups])
                _, groups, fit, cv, rank, distance = _select_sparse(
                    combined, combined, torch.cat([fit, off_fit]), cat_violation(cv, off_cv), self.pop_size
                )
            mask = (groups[:, :, None] & membership[None]).any(1)
            normalized = (self.pop - self.lb) / (self.ub - self.lb).clamp_min(torch.finfo(self.pop.dtype).tiny)
            unit_dec = normalized / normalized.norm(dim=1, keepdim=True).clamp_min(torch.finfo(self.pop.dtype).tiny)
            unit_mask = mask.to(self.pop.dtype) / mask.to(self.pop.dtype).norm(dim=1, keepdim=True).clamp_min(
                torch.finfo(self.pop.dtype).tiny
            )
            decisions = self.pop[(unit_mask @ unit_dec.T).argmax(1)]
            fit, cv = _evaluate_sparse(self, decisions * mask)
            merged_cv = cat_violation(self.cv, cv, probe_cv)
            self.pop, self.mask, self.fit, self.cv, self.rank, self.dis = _select_sparse(
                torch.cat([self.pop, decisions, probe_dec]),
                torch.cat([self.mask, mask, probe_mask]),
                torch.cat([self.fit, fit, probe_fit]),
                merged_cv,
                self.pop_size,
            )
        self.is_initialized = torch.tensor(True, device=self.pop.device)

    def _binary_operator(self, parents):
        half, dim = len(parents) // 2, parents.shape[1]
        first, second = parents[:half], parents[half : 2 * half]
        point = torch.randint(dim, (half, 1), device=parents.device)
        off = torch.where(torch.arange(dim, device=parents.device)[None] > point, second, first)
        return off ^ (torch.rand(half, dim, device=parents.device) < 1 / dim)

    def _environmental_selection(self, pop, mask, fit, N, cv=None):
        pop, mask, fit, self.cv, rank, distance = _select_sparse(pop, mask, fit, cv, N)
        return pop, mask, fit, rank, distance

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        count = _sparse_offspring_count(self, half=True)
        if not count:
            return
        n_parents = self.pop_size if count == self.pop_size // 2 else 2 * count
        mating_pool = tournament_selection_multifit(n_parents, [-self.dis, self.rank], tournament_size=2)
        off_mask = self._binary_operator(self.mask[mating_pool])
        off_dec = simulated_binary_half(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        off_dec = clamp(polynomial_mutation(off_dec, self.lb, self.ub), self.lb, self.ub)
        off_pop = self._match_operator(off_mask, off_dec)

        # 5. Evaluation
        off_fit, off_cv = _evaluate_sparse(self, off_pop * off_mask)

        # 6. Merge & Environmental Selection
        combined_pop = torch.cat([self.pop, off_pop], dim=0)
        combined_mask = torch.cat([self.mask, off_mask], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.mask, self.fit, self.rank, self.dis = self._environmental_selection(
            combined_pop, combined_mask, combined_fit, self.pop_size, cat_violation(self.cv, off_cv)
        )


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = TSSparseEA(pop_size=100, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    jit_state_step()

    torch.cuda.synchronize()
    exec_start = time.perf_counter()

    for i in range(1, 50):
        jit_state_step()

        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            fit = fit[~torch.any(torch.isnan(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")
