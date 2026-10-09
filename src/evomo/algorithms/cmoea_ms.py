import torch
import torch.nn.functional as F
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    constraint_dominance_matrix,
    take_violation,
)
from evomo.operators.selection.distance_truncation import distance_truncation
from evomo.utils import parse_evaluate, register_lazy_buffer


class CMOEA_MS(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        type: int = 1,
        max_gen: int = 100,
        lambda_: float = 0.5,
        **kwargs,
    ):
        """Initialize the CMOEA_MS population and optimization state.

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
        :param type: Default: ``1``. Variation selector: ``1`` uses simulated binary crossover and polynomial mutation.
            Any other value uses the current differential-variation branch with fixed scale ``0.5``.
        :type type: int
        :param max_gen: Positive generation budget used only for the scoring-stage transition.
            The caller still controls termination. Defaults to ``100``.
        :param lambda_: Feasible fraction required to enter the objective optimization stage; defaults to ``0.5``.
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
        if max_gen <= 0 or not 0 <= lambda_ <= 1:
            raise ValueError("max_gen must be positive and lambda_ must be in [0, 1]")
        self.stage_switch_gen = 0.1 * max_gen
        self.stage_fraction = lambda_
        self.lb = lb
        self.ub = ub
        # Rename 'type' to 'op_type' to avoid collision with torch.nn.Module.type
        self.op_type = Parameter(torch.tensor(type, dtype=torch.int32, device=device))
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.fitness = Mutable(torch.zeros(pop_size, device=device))
        self.iter = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")

    def _cal_sde(self, norm_fit: torch.Tensor) -> torch.Tensor:
        shifted = torch.maximum(norm_fit[:, None], norm_fit[None])
        distance = torch.linalg.vector_norm(norm_fit[:, None] - shifted, dim=-1)
        diagonal = torch.eye(len(norm_fit), device=norm_fit.device, dtype=torch.bool)
        k = max(1, int(len(norm_fit) ** 0.5))
        return 1 / (distance.masked_fill(diagonal, torch.inf).sort(dim=1).values[:, k - 1] + 2)

    def _cal_fitness(self, objectives: torch.Tensor, cv=None) -> torch.Tensor:
        dominate = (objectives[:, None] <= objectives[None]).all(-1) & (objectives[:, None] < objectives[None]).any(-1)
        dominate = constraint_dominance_matrix(dominate, cv, objective_ties=True)
        raw = dominate.T.to(objectives.dtype) @ dominate.sum(dim=1).to(objectives.dtype)
        distance = (1 - F.cosine_similarity(objectives[:, None], objectives[None], dim=-1)).nan_to_num(nan=1)
        diagonal = torch.eye(len(objectives), device=objectives.device, dtype=torch.bool)
        k = max(1, int(len(objectives) ** 0.5))
        density = 1 / (distance.masked_fill(diagonal, torch.inf).sort(dim=1).values[:, k - 1] + 2)
        return raw + density

    def _constrained_fitness(self, fit, cv, initial=False):
        """PlatEMO MS fitness; objective-only input is the zero-CV case."""
        if cv is None or (cv.ndim == 2 and cv.shape[1] == 0):
            violation = fit.new_zeros(fit.shape[0])
        else:
            positive = cv.clamp_min(0).nan_to_num(nan=torch.inf, posinf=torch.inf)
            maximum = torch.where(torch.isfinite(positive), positive, 0).amax(dim=0)
            normalized = positive / torch.where(maximum > 0, maximum, torch.ones_like(maximum))
            violation = normalized.mean(dim=1) if cv.ndim == 2 else normalized
        minimum, maximum = fit.amin(dim=0), fit.amax(dim=0)
        normalized_fit = (fit - minimum) / (maximum - minimum).clamp_min(1e-12)
        surrogate = torch.stack([self._cal_sde(normalized_fit), violation], dim=1)
        exploratory = self._cal_fitness(surrogate)
        convergent = self._cal_fitness(fit, violation)
        stage = ((violation == 0).float().mean() > self.stage_fraction) & (self.iter + 1 >= self.stage_switch_gen)
        fitness = exploratory if initial else torch.where(stage, convergent, exploratory)
        return torch.where(torch.isfinite(violation), fitness, torch.inf)

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit, self.cv = parse_evaluate(self.evaluate(self.pop))
        self.fitness = self._constrained_fitness(self.fit, self.cv, initial=True)

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.iter = self.iter + 1
        device = self.pop.device
        N = self.pop_size

        mating_idx = tournament_selection_multifit(N, [self.fitness], tournament_size=2)
        parents = self.pop[mating_idx]

        if self.op_type == 1:
            off_pop = simulated_binary(parents, pro_c=1.0, dis_c=20.0)
            off_pop = polynomial_mutation(off_pop, self.lb, self.ub, pro_m=1.0, dis_m=20.0)
        else:
            # DE logic: current-to-best style or similar
            # Using parents as a base for variation
            off_pop = parents + 0.5 * (
                self.pop[torch.randint(0, N, (N,), device=device)] - self.pop[torch.randint(0, N, (N,), device=device)]
            )

        off_pop = clamp(off_pop, self.lb, self.ub)
        off_fit, off_cv = parse_evaluate(self.evaluate(off_pop))

        Q_pop = torch.cat([self.pop, off_pop], dim=0)
        Q_fit = torch.cat([self.fit, off_fit], dim=0)

        Q_cv = cat_violation(self.cv, off_cv)
        fitness_total = self._constrained_fitness(Q_fit, Q_cv)
        nondominated = torch.where(fitness_total < 1)[0]
        if nondominated.numel() > N:
            objectives = Q_fit[nondominated]
            distance = (1 - F.cosine_similarity(objectives[:, None], objectives[None], dim=-1)).nan_to_num(nan=1)
            survivor_idx = nondominated[distance_truncation(distance, N)]
        else:
            survivor_idx = fitness_total.argsort(stable=True)[:N]

        self.pop = Q_pop[survivor_idx]
        self.fit = Q_fit[survivor_idx]
        self.cv = take_violation(Q_cv, survivor_idx)
        self.fitness = fitness_total[survivor_idx]


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    # CMOEA_MS must be replaced by your actual class name
    algo = CMOEA_MS(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
