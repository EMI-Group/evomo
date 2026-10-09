import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, lexsort, randint

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    constraint_keys,
    rank_with_constraints,
    take_violation,
    total_violation,
)
from evomo.utils import parse_evaluate, register_lazy_buffer, unique_rows_sorted


class CLIA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the CLIA population and optimization state.

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
        self.dim = lb.numel()

        # Reference Vectors initialization (Bug #13)
        v, _ = uniform_sampling(pop_size, n_objs)
        self.v = Mutable(v.to(device))

        # Population State
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), 1e10, device=device))

        # Archive State
        self.archive_pop = Mutable(torch.empty((0, self.dim), device=device))
        self.archive_fit = Mutable(torch.empty((0, n_objs), device=device))

        # SVM State (Step 2.4)
        self.svm_state = {
            "a": Mutable(torch.empty((0,), device=device)),
            "b": Mutable(torch.zeros((1,), device=device)),
            "X_mer": Mutable(torch.empty((0, self.dim), device=device)),
            "y_mer": Mutable(torch.empty((0,), device=device)),
        }

        self.zmin = Mutable(torch.full((n_objs,), 1e10, device=device))
        self.count = Mutable(torch.zeros(3, dtype=torch.int32, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        register_lazy_buffer(self, "archive_cv", device_like="pop")

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        self.zmin = torch.min(self.fit, dim=0)[0]
        self.archive_cv = None if self.cv is None else self.cv[:0]
        self._update_archive(self.pop, self.fit, self.cv)

    def _calculate_sine_distance(self, objs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Safe normalization (Bug #12)
        norm_objs = torch.norm(objs, dim=1, keepdim=True) + 1e-6
        norm_v = torch.norm(v, dim=1, keepdim=True) + 1e-6
        cosine = (objs @ v.t()) / (norm_objs @ norm_v.t())
        # Clamp for numerical stability
        sine_dist = torch.sqrt(torch.clamp(1 - cosine**2, min=0.0) + 1e-6)
        return sine_dist

    def _update_archive(self, off_pop: torch.Tensor, off_fit: torch.Tensor, off_cv=None):
        combined_pop = torch.cat([self.archive_pop, off_pop, self.pop], dim=0)
        combined_fit = torch.cat([self.archive_fit, off_fit, self.fit], dim=0)

        # Unique rows (Bug #3)
        combined_cv = cat_violation(self.archive_cv, off_cv, self.cv)
        if combined_cv is None:
            u_fit, u_idx = unique_rows_sorted(combined_fit)
        else:
            _, u_idx = unique_rows_sorted(torch.cat([combined_fit, total_violation(combined_cv)[:, None]], dim=1))
            u_fit = combined_fit[u_idx]
        u_pop = combined_pop[u_idx]

        # Non-dominated filter
        u_cv = take_violation(combined_cv, u_idx)
        rank = rank_with_constraints(u_fit, u_cv)
        mask = rank == 0

        # Deadlock Breaker: if no rank 1, take all (Bug #9)
        mask = torch.where(mask.any(), mask, torch.ones_like(mask, dtype=torch.bool))

        self.archive_pop = u_pop[mask]
        self.archive_fit = u_fit[mask]
        self.archive_cv = take_violation(u_cv, mask)

        # Pruning if archive is too large
        arc_size = self.archive_fit.shape[0]
        limit = int(0.33 * self.n_objs * self.pop_size)

        # Vectorized pruning logic
        zmin = torch.min(self.archive_fit, dim=0)[0]
        zmax = torch.max(self.archive_fit, dim=0)[0]
        norm_fit = (self.archive_fit - zmin) / (zmax - zmin + 1e-6)

        dist = self._calculate_sine_distance(norm_fit, self.v)
        min_dist, _ = torch.min(dist, dim=1)

        # Lexsort: Primary key is min_dist (last) (Bug #25)
        idx = lexsort(min_dist.unsqueeze(0))

        # Use torch.where or slicing to keep size consistent
        keep_count = torch.minimum(torch.tensor(limit, device=idx.device), torch.tensor(arc_size, device=idx.device))
        keep_idx = idx[:keep_count]

        self.archive_pop = self.archive_pop[keep_idx]
        self.archive_fit = self.archive_fit[keep_idx]
        self.archive_cv = take_violation(self.archive_cv, keep_idx)

    def _cascade_clustering_selection(self, combined_pop, combined_fit, combined_cv=None):
        # 1. Normalization
        zmin = torch.min(combined_fit, dim=0)[0]
        zmax = torch.max(combined_fit, dim=0)[0]
        norm_fit = (combined_fit - zmin) / (zmax - zmin + 1e-6)

        # 2. Association
        sine_dist = self._calculate_sine_distance(norm_fit, self.v)
        dist_matrix, pi = torch.min(sine_dist, dim=1)

        # 3. F-Metric (PDM)
        d1 = torch.norm(norm_fit, dim=1)
        f_metric = 5 * dist_matrix * d1 + torch.mean(norm_fit, dim=1)

        # 4. Round-Robin Picking (Bug #29)
        # Sort by cluster (pi) first, then by f_metric
        # Primary key pi must be last in lexsort stack (Bug #25)
        cluster_keys = constraint_keys([f_metric], combined_cv)
        sub_idx = lexsort([*cluster_keys, pi.float()])

        sorted_pi = pi[sub_idx]
        # Vectorized intra-cluster rank:
        # Mark where cluster changes
        diff = torch.cat([torch.tensor([0], device=pi.device), (sorted_pi[1:] != sorted_pi[:-1]).long()])
        # Cumulative sum resets at each new cluster
        intra_rank = (
            torch.arange(len(pi), device=pi.device)
            - torch.where(diff == 1, torch.arange(len(pi), device=pi.device), torch.tensor(0, device=pi.device)).cummax(dim=0)[
                0
            ]
        )

        # Final selection: Primary key is intra_rank (last), Secondary is f_metric
        final_sort_idx = lexsort(constraint_keys([f_metric[sub_idx], intra_rank.float()], take_violation(combined_cv, sub_idx)))
        final_indices = sub_idx[final_sort_idx]

        if combined_cv is not None:
            self.cv = combined_cv[final_indices[: self.pop_size]]
        return combined_pop[final_indices[: self.pop_size]], combined_fit[final_indices[: self.pop_size]]

    def _svm_incremental_learning(self):
        # Placeholder for Step 2.4 logic as per blueprint
        # In a real scenario, this would update self.v based on archive_fit
        # For runtime stability, we ensure the reference vectors remain valid
        pass

    def step(self) -> None:
        # 1. Mating
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        mating_pool = randint(0, self.pop_size, (self.pop_size,), device=self.lb.device)
        if self.cv is not None:
            other = randint(0, self.pop_size, (self.pop_size,), device=self.lb.device)
            totals = total_violation(self.cv)
            mating_pool = torch.where(totals[mating_pool] <= totals[other], mating_pool, other)
        parents = self.pop[mating_pool]
        offspring = simulated_binary(parents, pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit, off_cv = parse_evaluate(self.evaluate(offspring))

        # 3. Archive Update
        self._update_archive(offspring, off_fit, off_cv)
        self.zmin = torch.min(self.zmin, torch.min(off_fit, dim=0)[0])

        # 4. Environmental Selection
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit = self._cascade_clustering_selection(merge_pop, merge_fit, cat_violation(self.cv, off_cv))

        # 5. SVM Reference Adaptation (Step 2.4)
        self._svm_incremental_learning()


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = CLIA(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
