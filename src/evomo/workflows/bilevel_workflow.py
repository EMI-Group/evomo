"""Compose existing multiobjective algorithms into an optimistic bilevel search."""

import math
from collections.abc import Callable

import torch
from evox.core import Algorithm, Mutable, Problem, Workflow, use_state

from evomo.operators.selection import nd_environmental_selection, non_dominate_rank
from evomo.problems.bilevel import BLMOP
from evomo.utils import parse_evaluate

from .unified_workflow import UnifiedWorkflow


def _positive_integer(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _make_algorithm(factory, lb, ub, n_objs):
    algorithm = factory(lb=lb.clone(), ub=ub.clone(), n_objs=n_objs, device=lb.device)
    if not isinstance(algorithm, Algorithm):
        raise ValueError("Algorithm factories must return an EvoX Algorithm.")
    algorithm.to(device=lb.device, dtype=lb.dtype)
    if not isinstance(getattr(algorithm, "pop", None), torch.Tensor) or algorithm.pop.ndim != 2:
        raise ValueError("An algorithm must expose a floating-point population tensor 'pop' of shape (n, d).")
    if algorithm.pop.shape[0] < 1 or algorithm.pop.shape[1] != lb.numel() or not algorithm.pop.is_floating_point():
        raise ValueError("Algorithm population dimensions must match the supplied bounds.")
    if getattr(algorithm, "n_objs", n_objs) != n_objs:
        raise ValueError("Algorithm objective count must match the requested level.")
    return algorithm


def _constraint_support(algorithm, declared, needed, level):
    if declared is not None and not isinstance(declared, bool):
        raise ValueError(f"{level}_supports_constraints must be boolean or None.")
    supports = hasattr(algorithm, "cv") if declared is None else declared
    if needed and not supports:
        raise ValueError(
            f"The {level} algorithm must handle (fitness, cv) for this problem. "
            f"Use a constraint-aware algorithm, or declare {level}_supports_constraints=True for a custom adapter."
        )
    return supports


class _LowerRun(Workflow):
    """One independently functionalized lower algorithm, parameterized by xu."""

    def __init__(self, problem, algorithm, generations, supports_constraints):
        super().__init__()
        self.problem = problem
        self.generations = generations
        self.supports_constraints = supports_constraints
        self.xu = Mutable((problem.lb_u + problem.ub_u) / 2)
        self.evaluations = Mutable(torch.zeros((), device=problem.device, dtype=torch.int64))
        owner = self

        class BoundAlgorithm(type(algorithm)):
            def __init__(self):
                super(Algorithm, self).__init__()
                self.__dict__.update(algorithm.__dict__)

            def evaluate(self, population):
                owner.evaluations = owner.evaluations + population.shape[0]
                result = owner.problem.evaluate_lower(owner.xu, population)
                if owner.supports_constraints:
                    return result
                fit, _ = parse_evaluate(result)
                return fit

        self.algorithm = BoundAlgorithm()

    def solve(self):
        self.algorithm.init_step()
        for _ in range(self.generations):
            self.algorithm.step()
        return self.algorithm.pop


class _LowerSolver(torch.nn.Module):
    def __init__(self, problem, factory, generations, restarts, supports_constraints, batch_mode, chunk_size):
        super().__init__()
        algorithm = _make_algorithm(factory, problem.lb_l, problem.ub_l, problem.m_l)
        needed = problem.n_iq_l + problem.n_eq_l > 0
        supports = _constraint_support(algorithm, supports_constraints, needed, "lower")
        self.runner = _LowerRun(problem, algorithm, generations, supports)
        self.restarts = restarts
        self.batch_mode = batch_mode
        self.pop_size = algorithm.pop.shape[0]
        # Materialize lazy state once, retaining pre-initialization values for
        # all existing buffers. Count the actual setup evaluations separately.
        cold = {k: v.detach().clone() for k, v in self.runner.state_dict().items()}
        self.runner.algorithm.init_step()
        self.register_buffer("setup_evaluations", self.runner.evaluations.detach().clone())
        prepared = self.runner.state_dict()
        initial = {k: cold.get(k, v).detach().clone() for k, v in prepared.items()}
        if "algorithm.pop" not in initial:
            raise ValueError("The lower algorithm must store pop as a registered mutable buffer.")
        self.keys = tuple(initial)
        self.initial = torch.nn.Module()
        for i, value in enumerate(initial.values()):
            self.initial.register_buffer(f"s{i}", value)
        self.single = use_state(self.runner.solve)
        self.batched = torch.vmap(self.single, randomness="different", chunk_size=chunk_size)

    def solve(self, xu):
        p = self.runner.problem
        count = xu.shape[0] * self.restarts
        state = {
            k: getattr(self.initial, f"s{i}").unsqueeze(0).expand(count, *getattr(self.initial, f"s{i}").shape).clone()
            for i, k in enumerate(self.keys)
        }
        state["xu"] = xu[:, None, :].expand(-1, self.restarts, -1).reshape(count, p.d_u)
        state["algorithm.pop"] = p.lb_l + (p.ub_l - p.lb_l) * torch.rand(
            count, self.pop_size, p.d_l, device=xu.device, dtype=xu.dtype
        )
        if self.batch_mode == "vmap":
            final, pop = self.batched(state)
        else:
            outputs = [self.single({k: v[i] for k, v in state.items()}) for i in range(count)]
            final = {k: torch.stack([out[0][k] for out in outputs]) for k in outputs[0][0]}
            pop = torch.stack([out[1] for out in outputs])
        pop = pop.reshape(xu.shape[0], self.restarts * self.pop_size, p.d_l)
        fit, cv = parse_evaluate(p.evaluate_lower(xu[:, None, :], pop))
        if cv is None:
            cv = fit.new_zeros(*fit.shape[:-1], 0)
        # Re-evaluate returned populations: adapters need not expose fit, and
        # merged restarts must be ranked together for the same leader.
        evaluations = final["evaluations"].sum() + pop.shape[0] * pop.shape[1]
        finite = torch.isfinite(fit).all(-1) & torch.isfinite(cv).all(-1) & torch.isfinite(pop).all(-1)
        finite = finite & ((pop >= p.lb_l) & (pop <= p.ub_l)).all(-1)
        rank_cv = torch.cat((torch.where(torch.isfinite(cv), cv, torch.inf), (~finite).to(fit.dtype).unsqueeze(-1)), -1)
        rank = torch.vmap(non_dominate_rank)(torch.where(finite.unsqueeze(-1), fit, torch.inf), rank_cv)
        eligible = finite & (cv.sum(-1) == 0) & (rank == 0)
        return pop, fit, cv, eligible, evaluations


class _BilevelEvaluator(Problem):
    def __init__(self, problem, solver, archive_size, repair, validator, accuracy_tol):
        super().__init__()
        self.problem = problem
        self.solver = solver
        self.archive_size = archive_size
        self.repair = repair
        self.validator = validator
        self.accuracy_tol = accuracy_tol
        self.upper_supports_constraints = True
        n_cv = 1 + problem.n_iq_u + problem.n_eq_u + problem.n_iq_l + problem.n_eq_l
        self.pop = Mutable(problem.lb.new_zeros(archive_size, problem.d))
        self.fit = Mutable(problem.lb.new_full((archive_size, problem.m_u), torch.inf))
        self.lower_fit = Mutable(problem.lb.new_full((archive_size, problem.m_l), torch.inf))
        self.cv = Mutable(problem.lb.new_full((archive_size, n_cv), torch.inf))
        self.rank = Mutable(torch.zeros(archive_size, device=problem.device, dtype=torch.int32))
        self.upper_evaluations = Mutable(torch.zeros((), device=problem.device, dtype=torch.int64))
        self.lower_evaluations = Mutable(solver.setup_evaluations.clone())

    def evaluate(self, population):
        p = self.problem
        xu = population[:, : p.d_u].clamp(p.lb_u, p.ub_u)
        if self.repair is not None:
            xu = self.repair(xu).clamp(p.lb_u, p.ub_u)
        xl, lower_fit, lower_cv, eligible, evaluations = self.solver.solve(xu)
        if self.accuracy_tol is not None:
            reference = p.lower_pf(xu)
            distance = torch.cdist(lower_fit, reference, compute_mode="donot_use_mm_for_euclid_dist").amin(-1)
            eligible = eligible & (distance <= self.accuracy_tol)
        if self.validator is not None:
            accepted = self.validator(xu, xl, lower_fit, lower_cv)
            if accepted.shape != eligible.shape or accepted.dtype != torch.bool or accepted.device != eligible.device:
                raise ValueError("response_validator must return a boolean mask of shape (n_leaders, n_responses).")
            eligible = eligible & accepted
        leaders = xu[:, None, :].expand(-1, xl.shape[1], -1)
        upper_fit, upper_cv = parse_evaluate(p.evaluate_upper(leaders, xl))
        if upper_cv is None:
            upper_cv = upper_fit.new_zeros(*upper_fit.shape[:-1], 0)
        finite = torch.isfinite(upper_fit).all(-1) & torch.isfinite(upper_cv).all(-1)
        eligible = eligible & finite & torch.isfinite(population).all(-1)[:, None]
        excluded = (~eligible).to(upper_fit.dtype).unsqueeze(-1)
        cv = torch.cat((excluded, upper_cv, lower_cv), -1)
        cv = torch.where(torch.isfinite(cv), cv, torch.inf)
        upper_fit = torch.where(torch.isfinite(upper_fit), upper_fit, torch.inf)
        joint = torch.cat((leaders, xl), -1)
        self._archive(joint.flatten(0, 1), upper_fit.flatten(0, 1), lower_fit.flatten(0, 1), cv.flatten(0, 1))
        self.lower_evaluations = self.lower_evaluations + evaluations
        self.upper_evaluations = self.upper_evaluations + upper_fit.shape[0] * upper_fit.shape[1]
        # Preferences choose a response AFTER a multiobjective lower search.
        # Every eligible response, including unchosen ones, enters the archive.
        safe_lower = torch.where(torch.isfinite(lower_fit), lower_fit, 0)
        ideal = torch.where(eligible.unsqueeze(-1), safe_lower, torch.inf).amin(1, keepdim=True)
        nadir = torch.where(eligible.unsqueeze(-1), safe_lower, -torch.inf).amax(1, keepdim=True)
        has_response = eligible.any(-1)
        ideal = torch.where(has_response[:, None, None], ideal, 0)
        span = torch.where(has_response[:, None, None], nadir - ideal, 1).clamp_min(torch.finfo(xu.dtype).eps)
        normalized = (safe_lower - ideal) / span
        preference = population[:, p.d_u :].clamp(1e-6, 1).unsqueeze(1)
        weighted = normalized / preference
        score = weighted.amax(-1) + 1e-6 * weighted.sum(-1)
        score = torch.where(eligible, score, torch.inf)
        fallback = cv.sum(-1).argmin(-1)
        index = torch.where(has_response, score.argmin(-1), fallback)
        selected_fit = upper_fit.gather(1, index[:, None, None].expand(-1, 1, p.m_u)).squeeze(1)
        selected_cv = cv.gather(1, index[:, None, None].expand(-1, 1, cv.shape[-1])).squeeze(1)
        if self.upper_supports_constraints:
            return selected_fit, selected_cv
        # Only allowed when BOTH levels have no physical constraints. An
        # excluded response still cannot acquire a finite outer fitness.
        return torch.where((selected_cv.sum(-1) == 0)[:, None], selected_fit, torch.inf)

    def _archive(self, pop, fit, lower_fit, cv):
        payload = torch.cat((torch.cat((self.pop, pop)), torch.cat((self.lower_fit, lower_fit))), -1)
        payload, self.fit, self.rank, _, self.cv = nd_environmental_selection(
            payload, torch.cat((self.fit, fit)), self.archive_size, torch.cat((self.cv, cv))
        )
        self.pop = payload[:, : self.problem.d]
        self.lower_fit = payload[:, self.problem.d :]


class BilevelWorkflow(Workflow):
    """Plug two existing Algorithm factories into an optimistic BLMOP search.

    Factories receive keyword arguments ``lb, ub, n_objs, device``; for example
    ``partial(NSGA2, pop_size=32)``. The upper algorithm searches ``[xu, w]``
    with ``m_l`` response-preference coordinates w in [0,1], NOT joint [xu,xl]
    variables. Each evaluation solves the multiobjective follower using all
    independent restarts, merges their feasible nondominated responses, and
    selects a response by a normalized augmented Chebyshev score using w.
    An external nondominated archive retains ALL eligible responses and their
    actual joint [xu,xl] variables, upper/lower fitness and constraint values.

    ``pop, fit, lower_fit, cv, rank`` refer to this bounded archive; use
    ``solution_mask`` to exclude infeasible/padding/dominated entries.
    ``upper_algorithm.pop`` contains latent [xu,w] search coordinates.
    Its stored fitness always belongs to the response evaluated at that time;
    re-solving a leader can find a different approximation. The archive keeps
    the actual evaluated pairs, so no stochastic re-decoding is needed.

    The lower factory must expose pop and keep changing state in registered
    tensors (EvoX Mutable/Parameter). Initialization materializes lazy tensors
    once; these setup evaluations are included in lower_evaluations.
    Counters include all lower restarts and final response re-evaluation,
    and all upper response evaluations, not just the chosen response. A lower
    algorithm's nominal population size may differ from its actual size.

    By default, algorithms exposing cv are assumed constraint-aware. Other
    custom adapters can declare *_supports_constraints explicitly; declaring
    True is a contract to handle (fitness,cv), not an automatic penalty scheme.
    Algorithms without constraint support are rejected on constrained levels.
    The upper level must also handle lower-level physical constraints.

    ``lower_batch_mode='vmap'`` batches independent solver states and RNG;
    ``'sequential'`` is an explicit compatibility option. Optional lower_chunk_size
    reduces vmap memory. Algorithms must themselves support the chosen batching
    and compilation mode: no silent fallback or algorithm rewrite is performed.
    Call init_step() before compiling step(). Fixed lower budgets are unrolled.

    Restarts and larger budgets improve approximation, but do not certify
    global follower optimality. response_validator(xu,xl,lower_fit,lower_cv)
    can apply an independent problem-specific accuracy check and returns a
    fixed-shape boolean mask. Rejected responses cannot become solutions.
    Optional lower_accuracy_tol rejects responses whose objective distance to
    problem.lower_pf(xu) exceeds this tolerance. This requires implemented
    analytic reference fronts and depends on their sampling resolution; it
    is an absolute, unnormalized per-response tolerance, not aggregate LGD.
    upper_repair handles discrete leader variables before lower solving.
    """

    def __init__(
        self,
        problem: BLMOP,
        upper_algorithm: Callable,
        lower_algorithm: Callable,
        *,
        lower_generations: int = 30,
        lower_restarts: int = 2,
        archive_size: int = 128,
        lower_batch_mode: str = "vmap",
        lower_chunk_size: int | None = None,
        upper_supports_constraints: bool | None = None,
        lower_supports_constraints: bool | None = None,
        upper_repair: Callable | None = None,
        response_validator: Callable | None = None,
        lower_accuracy_tol: float | None = None,
    ):
        super().__init__()
        if not isinstance(problem, BLMOP) or min(problem.m_u, problem.m_l) < 2:
            raise ValueError("BilevelWorkflow requires a BLMOP with multiple objectives at both levels.")
        for name, value in (
            ("lower_generations", lower_generations),
            ("lower_restarts", lower_restarts),
            ("archive_size", archive_size),
        ):
            _positive_integer(name, value)
        if lower_batch_mode not in ("vmap", "sequential"):
            raise ValueError("lower_batch_mode must be 'vmap' or 'sequential'.")
        if lower_chunk_size is not None:
            _positive_integer("lower_chunk_size", lower_chunk_size)
        if lower_accuracy_tol is not None:
            if not math.isfinite(lower_accuracy_tol) or lower_accuracy_tol < 0:
                raise ValueError("lower_accuracy_tol must be finite and nonnegative.")
            try:
                problem.lower_pf((problem.lb_u + problem.ub_u) / 2)
            except NotImplementedError as error:
                raise ValueError("lower_accuracy_tol requires problem.lower_pf(); use response_validator instead.") from error
        solver = _LowerSolver(
            problem,
            lower_algorithm,
            lower_generations,
            lower_restarts,
            lower_supports_constraints,
            lower_batch_mode,
            lower_chunk_size,
        )
        evaluator = _BilevelEvaluator(problem, solver, archive_size, upper_repair, response_validator, lower_accuracy_tol)
        self.register_buffer("upper_lb", torch.cat((problem.lb_u, problem.lb_u.new_zeros(problem.m_l))))
        self.register_buffer("upper_ub", torch.cat((problem.ub_u, problem.ub_u.new_ones(problem.m_l))))
        upper = _make_algorithm(upper_algorithm, self.upper_lb, self.upper_ub, problem.m_u)
        evaluator.upper_supports_constraints = _constraint_support(
            upper, upper_supports_constraints, problem.n_iq_u + problem.n_eq_u + problem.n_iq_l + problem.n_eq_l > 0, "upper"
        )
        self.upper_workflow = UnifiedWorkflow(upper, evaluator, device=problem.device)
        self.generation = Mutable(torch.zeros((), device=problem.device, dtype=torch.int64))
        self._initialized = False

    @property
    def evaluator(self):
        return self.upper_workflow.problem

    @property
    def problem(self):
        return self.evaluator.problem

    @property
    def upper_algorithm(self):
        return self.upper_workflow.algorithm

    @property
    def pop(self):
        return self.evaluator.pop

    @property
    def fit(self):
        return self.evaluator.fit

    @property
    def lower_fit(self):
        return self.evaluator.lower_fit

    @property
    def cv(self):
        return self.evaluator.cv

    @property
    def rank(self):
        return self.evaluator.rank

    @property
    def solution_mask(self):
        return (self.cv.sum(-1) == 0) & (self.rank == 0)

    @property
    def upper_evaluations(self):
        return self.evaluator.upper_evaluations

    @property
    def lower_evaluations(self):
        return self.evaluator.lower_evaluations

    def init_step(self):
        if self._initialized:
            raise RuntimeError("BilevelWorkflow is already initialized.")
        self.upper_workflow.init_step()
        self._initialized = True

    def get_extra_state(self):
        return {"initialized": self._initialized}

    def set_extra_state(self, state):
        self._initialized = state["initialized"]

    def step(self):
        if not self._initialized:
            raise RuntimeError("Call init_step() before step().")
        self.upper_workflow.step()
        self.generation = self.generation + 1
