import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    rank_with_constraints,
)
from evomo.utils import parse_evaluate, register_lazy_buffer


class KnEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the KnEA population and optimization state.

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
        self.knee_points = Mutable(torch.zeros(pop_size, dtype=torch.bool, device=device))

        # Adaptive neighborhood ratio and knee point ratio per front
        # Using 2*pop_size to handle combined population in step
        self.r = Mutable(torch.full((2 * pop_size,), -1.0, device=device))
        self.t = Mutable(torch.full((2 * pop_size,), -1.0, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        # PlatEMO starts without identified knee points.

    def _get_knee_points(self, fit: torch.Tensor, radius: torch.Tensor):
        knees, distance, _ = self._find_knee_points(fit, radius)
        return knees, distance

    def _find_knee_points(self, fit: torch.Tensor, radius: torch.Tensor):
        size, objectives = fit.shape
        if size <= objectives:
            return torch.ones(size, device=fit.device, dtype=torch.bool), fit.new_zeros(size), fit.new_ones(())
        order = fit.argsort(dim=0, descending=True, stable=True)
        used = torch.zeros(size, device=fit.device, dtype=torch.bool)
        extremes = []
        for objective in range(objectives):
            candidate = order[:, objective]
            chosen = candidate[used[candidate].to(torch.int32).argmin()]
            extremes.append(chosen)
            used[chosen] = True
        extreme_fit = fit[torch.stack(extremes)]
        # A pseudoinverse also supports duplicate/constant objective columns.
        hyperplane = torch.linalg.pinv(extreme_fit) @ fit.new_ones((objectives, 1))
        distance = -(fit @ hyperplane - 1).squeeze(1) / torch.linalg.vector_norm(hyperplane).clamp_min(1e-12)
        span = (fit.amax(0) - fit.amin(0)) * radius
        remaining = torch.ones(size, device=fit.device, dtype=torch.bool)
        knees = torch.zeros_like(remaining)
        ordered = distance.argsort(descending=True, stable=True)
        for position in range(size):
            index = ordered[position]
            choose = remaining[index]
            knees[index] = choose
            nearby = ((fit - fit[index]).abs() <= span).all(dim=1)
            remaining = torch.where(choose, remaining & ~nearby, remaining)
        fraction = knees.to(fit.dtype).mean()
        positions = torch.arange(size, device=fit.device)
        last = torch.where(knees[ordered], positions, -1).amax()
        knees[ordered[last]] = False
        return knees, distance, fraction

    def _update_adaptive_params(self, front: torch.Tensor, previous_fraction: torch.Tensor):
        factor = torch.exp((1 - previous_fraction / 0.5) / self.n_objs)
        self.r[front] = torch.where(previous_fraction < 0, torch.ones_like(previous_fraction), self.r[front] / factor)

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        device = self.pop.device
        N = self.pop_size

        # 1. Mating Selection
        # Weighted Distance for Crowding
        D_mat = torch.cdist(self.fit, self.fit, p=2)
        D_mat = D_mat.masked_fill(torch.eye(N, device=device, dtype=torch.bool), torch.inf)
        k = min(3, N - 1)
        nearest = D_mat.topk(k, largest=False).values
        crowd = (nearest * torch.arange(3, 3 - k, -1, device=device)).sum(dim=1)

        # Tournament
        rank = rank_with_constraints(self.fit, self.cv)
        mating_pool = tournament_selection_multifit(N, [-crowd, -self.knee_points.float(), rank], tournament_size=2)

        # Variation
        crossovered = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(crossovered, self.lb, self.ub, pro_m=1.0, dis_m=20.0)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit, off_cv = parse_evaluate(self.evaluate(offspring))

        # 3. Environmental Selection
        combined_pop = torch.cat([self.pop, offspring], dim=0)
        combined_fit = torch.cat([self.fit, off_fit], dim=0)

        # NDSort
        combined_cv = cat_violation(self.cv, off_cv)
        fronts = rank_with_constraints(combined_fit, combined_cv)

        # Peeling logic
        new_pop = torch.zeros_like(self.pop)
        new_fit = torch.zeros_like(self.fit)
        new_cv = None if self.cv is None else torch.zeros_like(self.cv)
        new_knee = torch.zeros(N, dtype=torch.bool, device=device)

        current_count = 0

        # We iterate through fronts to fill the population
        # JIT-safe: iterate up to 2*N (worst case)
        for f_no in range(2 * N):
            mask = fronts == f_no
            num_in_front = torch.sum(mask.int())

            # If front is empty or we are already full, skip
            is_active = (num_in_front > 0) & (current_count < N)

            if is_active:
                f_fit = combined_fit[mask]
                f_pop = combined_pop[mask]

                # Calculate knee points for this front
                if f_fit.shape[0] > self.n_objs:
                    self._update_adaptive_params(f_no, self.t[f_no])
                is_knee_f, dist_f, fraction = self._find_knee_points(f_fit, self.r[f_no])
                if f_fit.shape[0] > self.n_objs:
                    self.t[f_no] = fraction

                if current_count + num_in_front <= N:
                    # Select all
                    indices = torch.arange(current_count, current_count + num_in_front, device=device)
                    new_pop[indices] = f_pop
                    new_fit[indices] = f_fit
                    if new_cv is not None:
                        new_cv[indices] = combined_cv[mask]
                    new_knee[indices] = is_knee_f
                    current_count += num_in_front
                else:
                    # Partial selection from the last front
                    num_needed = N - current_count
                    # Priority: Knee points first, then hyperplane distance
                    # lexsort: primary key last. Primary: is_knee, Secondary: dist
                    sel_indices = lexsort(torch.stack([-dist_f, -is_knee_f.float()]))

                    final_indices = sel_indices[:num_needed]
                    fill_indices = torch.arange(current_count, N, device=device)

                    new_pop[fill_indices] = f_pop[final_indices]
                    new_fit[fill_indices] = f_fit[final_indices]
                    if new_cv is not None:
                        new_cv[fill_indices] = combined_cv[mask][final_indices]
                    new_knee[fill_indices] = is_knee_f[final_indices]
                    current_count = N

        self.pop = new_pop
        self.fit = new_fit
        self.cv = new_cv
        self.knee_points = new_knee


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = KnEA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    # 1. Trigger JIT compilation (First step)
    jit_state_step()

    # 2. Pure execution (Remaining 49 steps)
    torch.cuda.synchronize()
    exec_start = time.perf_counter()

    for i in range(1, 50):
        jit_state_step()

        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            # Simple NaN filtering for metric calculation
            fit = fit[~torch.any(torch.isnan(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")
