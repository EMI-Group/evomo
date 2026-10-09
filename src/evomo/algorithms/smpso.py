import torch
from evox.core import Algorithm, Mutable
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import lexsort

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    constrained_dominates,
    rank_with_constraints,
    take_violation,
)
from evomo.utils import parse_evaluate, register_lazy_buffer, unique_rows_sorted


class SMPSO(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the SMPSO population and optimization state.

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
        :param kwargs: Default: ``{}``. Extra keyword arguments are accepted for constructor compatibility but are not
            read by this implementation. In particular, passing ``device=...`` here does not move tensors; place both
            bounds on the intended device before construction.
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

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.vel = Mutable(torch.zeros((pop_size, D), device=device))

        # Personal Best
        self.pbest_pop = Mutable(torch.zeros((pop_size, D), device=device))
        self.pbest_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # External Archive (Gbest)
        self.archive_pop = Mutable(torch.zeros((pop_size, D), device=device))
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.archive_size = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        register_lazy_buffer(self, "archive_cv", device_like="pop")
        register_lazy_buffer(self, "pbest_cv", device_like="pop")

    def _update_archive(self, off_pop, off_fit, off_cv=None) -> None:
        device = self.lb.device
        # Combine current archive and new candidates
        valid_archive_mask = torch.arange(self.pop_size, device=device) < self.archive_size

        combined_pop = torch.cat([self.archive_pop[valid_archive_mask], off_pop], dim=0)
        combined_fit = torch.cat([self.archive_fit[valid_archive_mask], off_fit], dim=0)

        # Remove duplicates
        u_pop, u_idx = unique_rows_sorted(combined_pop)
        u_fit = combined_fit[u_idx]

        # Non-dominated sort
        u_cv = take_violation(cat_violation(take_violation(self.archive_cv, valid_archive_mask), off_cv), u_idx)
        rank = rank_with_constraints(u_fit, u_cv)
        mask_rank1 = rank == 0
        rank1_fit = u_fit[mask_rank1]
        rank1_pop = u_pop[mask_rank1]

        num_rank1 = rank1_fit.shape[0]

        if num_rank1 > self.pop_size:
            # Truncate using crowding distance
            cd_mask = torch.ones(num_rank1, dtype=torch.bool, device=device)
            cd = crowding_distance(rank1_fit, cd_mask)
            # Sort by CD descending
            indices = lexsort(torch.stack([-cd]))
            selected_indices = indices[: self.pop_size]

            self.archive_pop = rank1_pop[selected_indices]
            self.archive_fit = rank1_fit[selected_indices]
            self.archive_cv = take_violation(take_violation(u_cv, mask_rank1), selected_indices)
            self.archive_size = torch.tensor(self.pop_size, dtype=torch.int32, device=device)
        else:
            # Fill archive and update size
            new_archive_pop = torch.zeros((self.pop_size, self.lb.numel()), device=device)
            new_archive_fit = torch.full((self.pop_size, self.n_objs), torch.inf, device=device)

            new_archive_pop[:num_rank1] = rank1_pop
            new_archive_fit[:num_rank1] = rank1_fit

            if u_cv is not None:
                self.archive_cv = torch.full_like(self.archive_cv, torch.inf)
                self.archive_cv[:num_rank1] = u_cv[mask_rank1]
            self.archive_pop = new_archive_pop
            self.archive_fit = new_archive_fit
            self.archive_size = torch.tensor(num_rank1, dtype=torch.int32, device=device)

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        self.pbest_pop = self.pop.clone()
        self.pbest_fit = self.fit.clone()
        self.pbest_cv = None if self.cv is None else self.cv.clone()
        self.archive_cv = None if self.cv is None else torch.full_like(self.cv, torch.inf)
        self._update_archive(self.pop, self.fit, self.cv)

    def _select_leaders(self):
        valid = torch.arange(self.pop_size, device=self.pop.device) < self.archive_size
        fits = self.archive_fit[valid]
        distance = crowding_distance(fits, torch.ones(fits.shape[0], dtype=torch.bool, device=fits.device))
        selected = tournament_selection_multifit(self.pop_size, [-distance], tournament_size=2)
        return self.archive_pop[valid][selected]

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        N = self.pop_size
        D = self.lb.numel()
        device = self.lb.device

        # 1. Leader Selection (Gbest)
        gbest_pop = self._select_leaders()

        # 2. Stochastic Parameters
        W = torch.rand((N, 1), device=device) * 0.4 + 0.1
        C1 = torch.rand((N, 1), device=device) * 1.0 + 1.5
        C2 = torch.rand((N, 1), device=device) * 1.0 + 1.5

        phi = torch.clamp(C1 + C2, min=4.0)
        chi = 2 / (torch.abs(2 - phi - torch.sqrt(phi**2 - 4 * phi)) + 1e-6)

        # 3. Velocity Update
        r1 = torch.rand((N, D), device=device)
        r2 = torch.rand((N, D), device=device)

        new_vel = chi * (W * self.vel + C1 * r1 * (self.pbest_pop - self.pop) + C2 * r2 * (gbest_pop - self.pop))

        # 4. Boundary & Deterministic Back
        delta = (self.ub - self.lb) / 2
        new_vel = torch.clamp(new_vel, -delta, delta)

        off_pop = self.pop + new_vel
        out_mask = (off_pop < self.lb) | (off_pop > self.ub)
        new_vel = torch.where(out_mask, new_vel * 0.001, new_vel)
        off_pop = torch.clamp(off_pop, self.lb, self.ub)

        # 5. Polynomial Mutation
        ind_mask = torch.rand(N, device=device) < 0.15
        # The operator already divides pro_m by D; gate particles only here.
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub, pro_m=ind_mask[:, None].to(off_pop.dtype))

        # 6. Evaluation
        off_fit, off_cv = parse_evaluate(self.evaluate(off_pop))

        # 7. Pbest Update
        pbest_dom_off = (self.pbest_fit <= off_fit).all(dim=-1) & (self.pbest_fit < off_fit).any(dim=-1)
        if off_cv is not None:
            pbest_dom_off = constrained_dominates(self.pbest_fit, self.pbest_cv, off_fit, off_cv)
        replace_mask = ~pbest_dom_off

        self.pbest_pop = torch.where(replace_mask.unsqueeze(1), off_pop, self.pbest_pop)
        self.pbest_fit = torch.where(replace_mask.unsqueeze(1), off_fit, self.pbest_fit)

        # 8. Archive Update
        if off_cv is not None:
            self.pbest_cv = torch.where(replace_mask.reshape((-1,) + (1,) * (off_cv.ndim - 1)), off_cv, self.pbest_cv)
        self._update_archive(off_pop, off_fit, off_cv)

        # Update current swarm state
        self.pop = off_pop
        self.fit = off_fit
        self.cv = off_cv
        self.vel = new_vel


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SMPSO(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
