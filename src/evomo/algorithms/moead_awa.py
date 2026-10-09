import math

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    constraint_dominance_matrix,
    constraint_improvement,
    prefer_by_constraint,
    take_violation,
    total_violation,
)
from evomo.operators.selection.distance_truncation import distance_truncation
from evomo.utils import parse_evaluate, register_lazy_buffer


class MOEADAWA(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        T: int = None,
        nr: int = None,
        nEP: int = None,
        max_evaluations: int | None = None,
        weight_update_interval: int = 100,
        rate_evol: float = 0.8,
        rate_update_weight: float = 0.05,
        **kwargs,
    ):
        """Initialize the MOEADAWA population and optimization state.

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
        :param T: Default: ``None``. Neighborhood size. ``None`` uses ``ceil(N / 10)`` for the sampled population size
            ``N``. An explicit value must satisfy ``1 <= T <= N``; this constructor does not cap it.
        :type T: int or None
        :param nr: Default: ``None``. Positive maximum replacements per offspring. ``None`` uses ``ceil(N / 100)`` for
            sampled population size ``N``.
        :type nr: int or None
        :param nEP: Default: ``None``. Positive external archive capacity for adaptive weights. ``None`` uses ``ceil(1.5
            * N)`` for sampled population size ``N``.
        :type nEP: int or None
        :param max_evaluations: Evaluation budget including initialization, used only for scheduling weight adaptation.
            ``None`` uses ``100 * actual_population_size``. Set this to the caller's actual budget; it does not stop the run.
        :param weight_update_interval: Default ``100``; weight adaptation interval in population-equivalent evaluations.
        :param rate_evol: Default ``0.8``; begin archive/weight adaptation after this fraction of the budget.
        :param rate_update_weight: Default ``0.05``; fraction of weights replaced per adaptation.
        :param kwargs: Default: ``{}``. Extra keyword arguments are accepted for constructor compatibility but are not
            read by this implementation. In particular, passing ``device=...`` here does not move tensors; place both
            bounds on the intended device before construction.
        :type kwargs: dict

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            accepts an objective tensor or a ``(fitness, constraint_violation)`` tuple.
            Constrained evaluations retain violations through selection and state updates.

            Reference-vector sampling can change the requested population size. Read ``self.pop.shape[0]`` for the actual size.

            Tensor allocation uses ``lb.device``. There is no explicit device parameter; prepare both bounds on the intended
            device.
        """
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # 1. Weight Init & Transformation
        W, actual_n = uniform_sampling(pop_size, n_objs)
        self.pop_size = int(actual_n)
        if self.pop_size < 2:
            raise ValueError("MOEADAWA requires at least two sampled solutions")
        self.max_evaluations = self.pop_size * 100 if max_evaluations is None else max_evaluations
        if (
            self.max_evaluations <= 0
            or weight_update_interval <= 0
            or not 0 <= rate_evol <= 1
            or not 0 < rate_update_weight <= 1
        ):
            raise ValueError("invalid adaptation budget, interval, or rate")
        self.weight_update_period = max(1, (weight_update_interval + 4) // 5)
        self.rate_evol = rate_evol
        self.rate_update_weight = rate_update_weight
        W = W.to(device)

        W_inv = W.clamp_min(1e-6).reciprocal()
        self.W = Mutable(W_inv / W_inv.sum(dim=1, keepdim=True))

        self.T = T if T is not None else (self.pop_size + 9) // 10
        self.nr = nr if nr is not None else (self.pop_size + 99) // 100
        self.nEP = nEP if nEP is not None else (self.pop_size * 3 + 1) // 2
        if not 1 <= self.T <= self.pop_size or self.nr <= 0 or self.nEP <= 0:
            raise ValueError("invalid neighborhood, replacement limit, or archive capacity")

        dist = torch.cdist(self.W, self.W)
        self.B = Mutable(torch.topk(dist, self.T, largest=False, dim=1).indices.to(torch.int32))

        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), 1e10, device=device))
        self.z = Mutable(torch.full((1, n_objs), 1e10, device=device))
        self.pi = Mutable(torch.ones((self.pop_size, 1), device=device))
        self.old_obj = Mutable(torch.full((self.pop_size, 1), 1e10, device=device))

        self.archive_pop = Mutable(torch.zeros((self.nEP, D), device=device))
        self.archive_fit = Mutable(torch.full((self.nEP, n_objs), 1e10, device=device))
        self.archive_size = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        self.evaluations = Mutable(torch.zeros((), dtype=torch.int64, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        register_lazy_buffer(self, "archive_cv", device_like="pop")
        register_lazy_buffer(self, "old_cv", device_like="pop")

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        self.z = torch.min(self.fit, dim=0, keepdim=True).values
        self.old_obj = torch.max(self.W * torch.abs(self.fit - self.z), dim=1, keepdim=True).values
        self.old_cv = None if self.cv is None else total_violation(self.cv)[:, None]
        self.archive_cv = None if self.cv is None else self.cv.new_full((self.nEP,) + self.cv.shape[1:], torch.inf)
        self.evaluations = torch.full_like(self.evaluations, self.pop_size)

    def _update_utility(self):
        g = torch.max(self.W * torch.abs(self.fit - self.z), dim=1, keepdim=True).values
        delta = (self.old_obj - g) / (self.old_obj + 1e-6)
        if self.cv is not None:
            new_cv = total_violation(self.cv)[:, None]
            delta = constraint_improvement(delta, new_cv if self.old_cv is None else self.old_cv, new_cv)
            self.old_cv = new_cv
        decay = 0.95 + 0.05 * delta / 0.001
        self.pi = torch.where(delta > 0.001, torch.ones_like(self.pi), self.pi * decay)
        if self.cv is not None:
            self.pi = self.pi.clamp(0, 1)
        self.old_obj = g

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        device = self.lb.device
        N = self.pop_size
        checkpoint = (self.evaluations + N - 1) // N
        if checkpoint % 10 == 0:
            self._update_utility()

        # 2.3 Variation & Neighborhood Update (Vectorized over chosen subproblems)
        for _ in range(5):
            # AWA uses transformed weights: boundary vectors have one small
            # component, including when there are more than two objectives.
            boundary_indices = torch.where((self.W < 1e-3).sum(dim=1) == 1)[0]
            num_candidates = max(0, N // 5 - boundary_indices.numel())
            mating_indices = tournament_selection_multifit(num_candidates, [-self.pi.squeeze()], tournament_size=10).reshape(-1)
            chosen_subproblems = torch.cat([boundary_indices, mating_indices])
            num_chosen = chosen_subproblems.numel()
            # Parent selection for all chosen subproblems
            # rand_mask: 1 for neighborhood, 0 for whole population
            rand_mask = torch.rand(num_chosen, device=device) < 0.9

            nb_indices = self.B[chosen_subproblems.long()]
            local_order = nb_indices.gather(1, torch.rand(num_chosen, self.T, device=device).argsort(dim=1)).long()
            global_order = torch.rand(num_chosen, N, device=device).argsort(dim=1)
            padded_local = torch.cat([local_order, global_order[:, self.T :]], dim=1)
            targets = torch.where(rand_mask[:, None], padded_local, global_order)
            slots = torch.arange(N, device=device)
            available = ~rand_mask[:, None] | (slots < self.T)
            p1_idx = targets[:, 0]
            p2_idx = (
                targets[:, 1]
                if self.T > 1
                else torch.where(p1_idx == global_order[:, 0], global_order[:, 1], global_order[:, 0])
            )

            # Variation
            parents = torch.cat([self.pop[p1_idx], self.pop[p2_idx]], dim=0)
            offspring = simulated_binary(parents)[:num_chosen]
            offspring = polynomial_mutation(offspring, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)

            off_fit, off_cv = parse_evaluate(self.evaluate(offspring))
            self.evaluations = self.evaluations + num_chosen
            self.z = torch.min(self.z, torch.min(off_fit, dim=0, keepdim=True).values)
            incumbent_cv = None if self.cv is None else total_violation(self.cv)
            offspring_cv = None if off_cv is None else total_violation(off_cv)

            # Update neighborhood for each chosen subproblem
            # This part is semi-vectorized to maintain MOEA/D logic
            for idx in range(num_chosen):
                proposals = targets[idx]
                g_old = torch.max(self.W[proposals] * torch.abs(self.fit[proposals] - self.z), dim=1).values
                g_new = torch.max(self.W[proposals] * torch.abs(off_fit[idx] - self.z), dim=1).values
                mask = available[idx] & (g_new <= g_old)
                if self.cv is not None:
                    mask = available[idx] & prefer_by_constraint(mask, offspring_cv[idx], incumbent_cv[proposals])
                mask = mask & (mask.long().cumsum(0) <= self.nr)
                replace = (
                    torch.zeros(N, dtype=torch.long, device=device).scatter_reduce(0, proposals, mask.long(), reduce="amax") > 0
                )
                self.pop = torch.where(replace[:, None], offspring[idx], self.pop)
                self.fit = torch.where(replace[:, None], off_fit[idx], self.fit)
                if self.cv is not None:
                    self.cv = torch.where(replace.reshape((-1,) + (1,) * (self.cv.ndim - 1)), off_cv[idx], self.cv)
                    incumbent_cv = torch.where(replace, offspring_cv[idx], incumbent_cv)

        # 2.4 Adaptive Weight Adjustment (AWA)
        if self.evaluations >= self.rate_evol * self.max_evaluations:
            if self.archive_size == 0:
                self.archive_pop, self.archive_fit, self.archive_size, self.archive_cv = _update_ep(
                    self.archive_pop,
                    self.archive_fit,
                    self.archive_size,
                    self.pop,
                    self.fit,
                    self.nEP,
                    self.archive_cv,
                    self.cv,
                )
            self.archive_pop, self.archive_fit, self.archive_size, self.archive_cv = _update_ep(
                self.archive_pop, self.archive_fit, self.archive_size, offspring, off_fit, self.nEP, self.archive_cv, off_cv
            )
        checkpoint = (self.evaluations + N - 1) // N
        if self.evaluations >= self.rate_evol * self.max_evaluations and checkpoint % self.weight_update_period == 0:
            self.W, self.B, self.pop, self.fit, self.cv = _awa_logic(
                self.W,
                self.pop,
                self.fit,
                self.archive_pop,
                self.archive_fit,
                self.archive_size,
                self.z,
                self.T,
                self.cv,
                self.archive_cv,
                self.rate_update_weight,
            )


def _update_ep(archive_pop, archive_fit, archive_size, new_pop, new_fit, max_size, archive_cv=None, new_cv=None):
    device = archive_pop.device
    # Merge
    combined_pop = torch.cat([archive_pop[:archive_size], new_pop], dim=0)
    combined_fit = torch.cat([archive_fit[:archive_size], new_fit], dim=0)

    # Non-dominated filtering
    # f1 dominates f2 if f1 <= f2 and f1 < f2
    dom_mat = (combined_fit.unsqueeze(1) <= combined_fit.unsqueeze(0)).all(dim=-1) & (
        combined_fit.unsqueeze(1) < combined_fit.unsqueeze(0)
    ).any(dim=-1)
    combined_cv = cat_violation(take_violation(archive_cv, slice(None, archive_size)), new_cv)
    dom_mat = constraint_dominance_matrix(dom_mat, combined_cv)
    is_nondominated = ~dom_mat.any(dim=0)

    curr_pop = combined_pop[is_nondominated]
    curr_fit = combined_fit[is_nondominated]
    curr_cv = take_violation(combined_cv, is_nondominated)

    # Pruning using product of K-nearest neighbors
    if curr_fit.shape[0] > max_size:
        M = curr_fit.shape[1]
        keep_idx = distance_truncation(torch.cdist(curr_fit, curr_fit), max_size, neighbors=M)
        curr_pop = curr_pop[keep_idx]
        curr_fit = curr_fit[keep_idx]
        curr_cv = take_violation(curr_cv, keep_idx)

    new_size_val = curr_fit.shape[0]
    res_pop = torch.zeros_like(archive_pop)
    res_fit = torch.full_like(archive_fit, 1e10)
    res_pop[:new_size_val] = curr_pop
    res_fit[:new_size_val] = curr_fit
    res_cv = None
    if archive_cv is not None:
        res_cv = torch.full_like(archive_cv, torch.inf)
        res_cv[:new_size_val] = curr_cv
    return res_pop, res_fit, torch.tensor(new_size_val, dtype=torch.int32, device=device), res_cv


def _awa_logic(W, pop, fit, ep_pop, ep_fit, ep_size, z, T, cv=None, ep_cv=None, rate=0.05):
    device = W.device
    N, M = W.shape
    valid_ep_fit = ep_fit[:ep_size]
    valid_ep_pop = ep_pop[:ep_size]
    valid_ep_cv = take_violation(ep_cv, slice(None, ep_size))

    # 1. Re-assign
    all_fit = torch.cat([fit, valid_ep_fit], dim=0)
    all_pop = torch.cat([pop, valid_ep_pop], dim=0)

    # Tchebycheff for all solutions against all weights
    g_all = torch.max(W.unsqueeze(1) * torch.abs(all_fit.unsqueeze(0) - z), dim=2).values
    all_cv = cat_violation(cv, valid_ep_cv)
    if all_cv is not None:
        totals = total_violation(all_cv)
        g_all = torch.where((totals == totals.amin())[None], g_all, torch.inf)
        eligible_ep = total_violation(valid_ep_cv) == totals.amin()
        valid_ep_fit = valid_ep_fit[eligible_ep]
        valid_ep_pop = valid_ep_pop[eligible_ep]
        valid_ep_cv = valid_ep_cv[eligible_ep]
    best_idx = torch.argmin(g_all, dim=1)

    new_pop = all_pop[best_idx]
    new_fit = all_fit[best_idx]
    new_cv = take_violation(all_cv, best_idx)

    # 2. Delete Overcrowded & Add New
    num_change = min(math.ceil(N * rate), valid_ep_fit.shape[0])
    keep = distance_truncation(torch.cdist(new_fit, new_fit), N - num_change, neighbors=M)
    combined_fit = torch.cat([new_fit, valid_ep_fit])
    distances = torch.cdist(valid_ep_fit, combined_fit)
    selected = torch.cat([keep, torch.zeros(valid_ep_fit.shape[0], device=device, dtype=torch.bool)])
    ep_selected = torch.zeros(valid_ep_fit.shape[0], device=device, dtype=torch.bool)
    for added in range(num_change):
        nearest = (
            distances.masked_fill(~selected[None], torch.inf).topk(min(M, N - num_change + added), largest=False, dim=1).values
        )
        density = nearest.prod(dim=1)
        chosen = torch.where(ep_selected, -torch.inf, density).argmax()
        ep_selected[chosen] = True
        selected[N + chosen] = True
    added_fit = valid_ep_fit[ep_selected]
    inverse = (added_fit - z).clamp_min(1e-6).reciprocal()
    added_weights = inverse / inverse.sum(dim=1, keepdim=True)
    W_new = torch.cat([W[keep], added_weights])
    new_pop = torch.cat([new_pop[keep], valid_ep_pop[ep_selected]])
    new_fit = torch.cat([new_fit[keep], added_fit])
    if new_cv is not None:
        new_cv = torch.cat([new_cv[keep], valid_ep_cv[ep_selected]])

    dist = torch.cdist(W_new, W_new)
    B_new = torch.topk(dist, T, largest=False, dim=1).indices.to(torch.int32)

    return W_new, B_new, new_pop, new_fit, new_cv


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEADAWA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
            fit = fit[~torch.any(torch.isinf(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")
