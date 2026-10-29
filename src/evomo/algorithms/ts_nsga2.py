import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp, lexsort

from evomo.operators.selection import non_dominate_rank


class TSNSGAII(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, max_fe: int = 10000):
        """Initialize the TSNSGAII population and optimization state.

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
        :param max_fe: Default: ``10000``. Positive evaluation horizon setting the phase boundary at
            ``(0.8 + 0.2 * (1 - 3 / n_objs)) * max_fe`` evaluations, as in the reference implementation. The evaluation
            counter includes the initial population. This parameter controls the phase switch and does not stop
            subsequent calls to ``step()``.
        :type max_fe: int

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            expects an objective tensor of shape ``(B, n_objs)``. It does not consume a ``(fitness,
            constraint_violation)`` tuple.

            Tensor allocation uses ``lb.device``. There is no explicit device parameter; prepare both bounds on the intended
            device.
        """
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.max_fe = max_fe
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # Reference Weights
        w, _ = uniform_sampling(pop_size, n_objs)
        self.w = Mutable(w.to(device))

        # Selection Keys
        self.front_no = Mutable(torch.zeros(pop_size, dtype=torch.int32, device=device))
        self.d2 = Mutable(torch.zeros(pop_size, device=device))
        self.fe = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit = self.evaluate(self.pop)
        self.fe = self.fe + self.pop_size
        # Initial Selection to populate keys
        self.pop, self.fit, self.front_no, self.d2 = self._environmental_selection(self.pop, self.fit)

    def step(self) -> None:
        # 1. Mating
        # Lexsort keys: Primary (front_no) must be last. Secondary (d2) first.
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        mating_pool = tournament_selection_multifit(self.pop_size, [self.d2, self.front_no.float()], tournament_size=2)
        offspring = simulated_binary(self.pop[mating_pool], pro_c=1.0, dis_c=20.0)
        offspring = polynomial_mutation(offspring, self.lb, self.ub, dis_m=20.0)
        offspring = clamp(offspring, self.lb, self.ub)

        # 2. Evaluation
        off_fit = self.evaluate(offspring)
        self.fe = self.fe + self.pop_size

        # 3. Selection
        merged_pop = torch.cat([self.pop, offspring], dim=0)
        merged_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit, self.front_no, self.d2 = self._environmental_selection(merged_pop, merged_fit)

    def _environmental_selection(self, pop, fit):
        # Stage 1 (SPD sorting) until (0.8 + 0.2 * (1 - 3 / M)) of the evaluation budget, then stage 2.
        if self.fe < (0.8 + 0.2 * (1 - 3 / self.n_objs)) * self.max_fe:
            return self._spd_selection(pop, fit)
        return self._decomposition_selection(pop, fit)

    def _spd_selection(self, pop, fit):
        """Stage 1: keep the first SPD fronts, then the lowest-d2 solutions of the last one."""
        N = self.pop_size
        norm_fit = _normalize(fit)
        cosine = _cosine(norm_fit, self.w)
        norm_val = torch.norm(norm_fit, dim=1, keepdim=True)
        d1_all = norm_val * cosine
        d2_all = norm_val * torch.sqrt(clamp(1 - cosine**2, 0.0, 1.0))
        d2, rp = torch.min(d2_all, dim=1)
        d1 = d1_all.gather(1, rp.unsqueeze(1)).squeeze(1)

        # The extreme non-dominated solution of each objective gets d1 = d2 = 0.
        nd = torch.where(non_dominate_rank(norm_fit) == 0)[0]
        extreme = nd[torch.argmax(norm_fit[nd], dim=0)]
        d1 = d1.index_fill(0, extreme, 0.0)
        d2 = d2.index_fill(0, extreme, 0.0)

        front_no = self._spd_sort(norm_fit, d1 + 5 * d2, rp, N)
        max_front = torch.max(torch.where(front_no < torch.iinfo(front_no.dtype).max, front_no, -1))
        # Earlier fronts first, then the last front by d2; unsorted solutions sort last.
        survivor_idx = lexsort([d2, front_no])[:N]
        return pop[survivor_idx], fit[survivor_idx], torch.clamp(front_no[survivor_idx], max=max_front), d2[survivor_idx]

    def _spd_sort(self, fit, spd_score, rp, n_sort):
        """Assign SPD fronts in the reference's sequential elimination order.

        A solution SPD-dominates another when it Pareto-dominates it, or when neither Pareto-dominates the other,
        they share a reference vector, and it has the smaller d1 + 5 * d2. This relation is not transitive:
        candidates eliminated during a front's scan must not eliminate later candidates. Unsorted solutions
        keep the dtype maximum.
        """
        n = fit.shape[0]
        better_any = (fit.unsqueeze(1) < fit.unsqueeze(0)).any(-1)
        pareto_dom = (fit.unsqueeze(1) <= fit.unsqueeze(0)).all(-1) & better_any
        incomparable = ~pareto_dom & ~pareto_dom.T
        same_rp = rp.unsqueeze(1) == rp.unsqueeze(0)
        dominates = pareto_dom | (incomparable & same_rp & (spd_score.unsqueeze(1) < spd_score.unsqueeze(0)))

        unsorted_value = torch.iinfo(torch.int32).max
        front_no = torch.full((n,), unsorted_value, dtype=torch.int32, device=fit.device)
        positions = torch.arange(n, device=fit.device)
        front = 0
        while int(torch.sum(front_no < unsorted_value)) < min(n_sort, n):
            front += 1
            deleted = front_no != unsorted_value
            for i in range(n):
                active = ~deleted[i]
                later = (positions > i) & ~deleted
                # Stop at the first remaining candidate that defeats i. Comparisons before that point
                # may delete candidates, but comparisons after it must have no effect.
                first_defeat = torch.where(later & dominates[:, i], positions, n).amin()
                deleted = deleted | (active & later & (positions < first_defeat) & dominates[i])
                deleted[i] = deleted[i] | (active & (first_defeat < n))
                front_no[i] = torch.where(~deleted[i], front, front_no[i])
        return front_no

    def _decomposition_selection(self, pop, fit):
        """Stage 2: among the first non-dominated fronts holding at least N solutions, take for each reference
        vector in turn the remaining solution with the largest cosine to it, after normalizing the remaining ones.

        The reference's level sort does not narrow this set (its early exit compares ``size(..., 1)`` of a row
        of solutions, which is always 1), so it is not applied. With fewer reference vectors than ``pop_size``,
        the vectors are reused in order until ``pop_size`` solutions are chosen.
        """
        N = self.pop_size
        rank = non_dominate_rank(fit)
        sorted_rank, _ = torch.sort(rank)
        candidate = rank <= sorted_rank[N - 1]

        chosen = []
        remaining = candidate.clone()
        n_w = self.w.shape[0]
        for t in range(N):
            f_min = _masked_min(fit, remaining)
            f_max = _masked_max(fit, remaining)
            norm_fit = (fit - f_min) / torch.clamp(f_max - f_min, min=torch.finfo(fit.dtype).tiny)
            cosine = _cosine(norm_fit, self.w[t % n_w].unsqueeze(0)).squeeze(1)
            best = torch.argmax(torch.where(remaining, cosine, -torch.inf))
            chosen.append(best)
            remaining = remaining.clone()
            remaining[best] = False
        survivor_idx = torch.stack(chosen)

        new_fit = fit[survivor_idx]
        new_front = non_dominate_rank(new_fit)
        # Density of each solution's region (number of selected solutions closest in cosine to the same vector).
        region = torch.argmax(_cosine(_normalize(new_fit), self.w), dim=1)
        density = torch.bincount(region, minlength=n_w)[region].to(new_fit.dtype)
        return pop[survivor_idx], new_fit, new_front, density


def _normalize(fit):
    f_min = torch.min(fit, dim=0)[0]
    f_max = torch.max(fit, dim=0)[0]
    return (fit - f_min) / torch.clamp(f_max - f_min, min=torch.finfo(fit.dtype).tiny)


def _masked_min(fit, mask):
    return torch.min(torch.where(mask.unsqueeze(1), fit, torch.inf), dim=0)[0]


def _masked_max(fit, mask):
    return torch.max(torch.where(mask.unsqueeze(1), fit, -torch.inf), dim=0)[0]


def _cosine(norm_fit, w):
    """Cosine between each normalized solution and each reference vector; a zero vector has cosine 0."""
    tiny = torch.finfo(norm_fit.dtype).tiny
    norm_val = torch.clamp(torch.norm(norm_fit, dim=1, keepdim=True), min=tiny)
    w_norm = torch.clamp(torch.norm(w, dim=1, keepdim=True), min=tiny)
    return (norm_fit @ w.T) / (norm_val @ w_norm.T)


# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = TSNSGAII(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
