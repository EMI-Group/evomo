"""A small optimistic bilevel baseline with batched nested NSGA-II searches."""

from collections.abc import Callable

import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit

from evomo.operators.selection import nd_environmental_selection
from evomo.problems.bilevel import BLMOP
from evomo.utils import parse_evaluate


class NestedNSGA2(Algorithm):
    """Optimistic nested NSGA-II for a BLMOP with multiple objectives at both levels.

    This self-contained driver owns its problem: call ``init_step()`` and then
    ``step()`` directly, without UnifiedWorkflow. Each upper offspring gets a
    fresh lower NSGA-II search; searches for all leaders are batched with vmap.
    Upper survival considers every feasible rank-zero lower response. It does
    not scalarize the follower or reduce its response set to a single point.

    ``pop`` contains joint [xu, xl] pairs, ``fit`` their upper objectives, and
    ``lower_fit`` their lower objectives. ``cv`` contains a response exclusion
    column (infinity for lower dominated/infeasible candidates), followed by
    the original upper and lower constraint violations. Insufficient eligible
    responses leave excluded fillers in the fixed-size population; their cv
    is infinite and they must not be reported as solutions.

    Finite lower searches approximate Pareto optimality, not certify it. In
    particular, deceptive problems require adequate lower search budgets.
    Optional ``upper_repair`` handles discrete leader variables explicitly.
    Both levels use SBX, polynomial mutation and NSGA-II environmental selection.
    """

    def __init__(
        self,
        problem: BLMOP,
        pop_size: int = 32,
        lower_pop_size: int = 32,
        lower_generations: int = 30,
        upper_repair: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        if not isinstance(problem, BLMOP) or min(problem.m_u, problem.m_l) < 2:
            raise ValueError("NestedNSGA2 requires a BLMOP with multiple objectives at both levels.")
        for name, value in (("pop_size", pop_size), ("lower_pop_size", lower_pop_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 4 or value % 2:
                raise ValueError(f"{name} must be an even integer >= 4.")
        if not isinstance(lower_generations, int) or isinstance(lower_generations, bool) or lower_generations < 1:
            raise ValueError("lower_generations must be a positive integer.")
        self.problem = problem
        self.pop_size, self.lower_pop_size = pop_size, lower_pop_size
        self.lower_generations, self.upper_repair = lower_generations, upper_repair
        self.n_objs, self.dim = problem.m_u, problem.d
        self.n_cv_u = problem.n_iq_u + problem.n_eq_u
        self.n_cv_l = problem.n_iq_l + problem.n_eq_l
        xu = self._repair(self._sample((pop_size,), problem.lb_u, problem.ub_u))
        self.pop = Mutable(torch.cat((xu, xu.new_zeros(pop_size, problem.d_l)), -1))
        self.fit = Mutable(xu.new_full((pop_size, problem.m_u), torch.inf))
        self.lower_fit = Mutable(xu.new_full((pop_size, problem.m_l), torch.inf))
        self.cv = Mutable(xu.new_full((pop_size, 1 + self.n_cv_u + self.n_cv_l), torch.inf))
        self.rank = Mutable(torch.zeros(pop_size, dtype=torch.int32, device=xu.device))
        self.dis = Mutable(xu.new_full((pop_size,), -torch.inf))
        self.generation = Mutable(torch.zeros((), dtype=torch.int64, device=xu.device))
        self.upper_evaluations = Mutable(torch.zeros((), dtype=torch.int64, device=xu.device))
        self.lower_evaluations = Mutable(torch.zeros((), dtype=torch.int64, device=xu.device))
        self._initialized = False

    @staticmethod
    def _sample(batch, lb, ub):
        return lb + (ub - lb) * torch.rand(*batch, lb.numel(), device=lb.device, dtype=lb.dtype)

    def _repair(self, xu):
        xu = xu.clamp(self.problem.lb_u, self.problem.ub_u)
        if self.upper_repair is not None:
            xu = self.upper_repair(xu)
        return xu.clamp(self.problem.lb_u, self.problem.ub_u)

    @staticmethod
    def _vary(pop, rank, dis, cv, lb, ub):
        parents = tournament_selection_multifit(pop.shape[0], [-dis, rank, cv.sum(-1)])
        return polynomial_mutation(simulated_binary(pop[parents]), lb, ub).clamp(lb, ub)

    def _lower_evaluate(self, xu, xl):
        fit, cv = parse_evaluate(self.problem.evaluate_lower(xu[:, None, :], xl))
        if cv is None:
            cv = fit.new_zeros(*fit.shape[:-1], 0)
        return fit, cv

    def _solve_lower(self, xu):
        p = self.problem
        pop = self._sample((xu.shape[0], self.lower_pop_size), p.lb_l, p.ub_l)
        fit, cv = self._lower_evaluate(xu, pop)
        select = torch.vmap(nd_environmental_selection, in_dims=(0, 0, None, 0))
        pop, fit, rank, dis, cv = select(pop, fit, self.lower_pop_size, cv)
        vary = torch.vmap(self._vary, in_dims=(0, 0, 0, 0, None, None), randomness="different")
        for _ in range(self.lower_generations):
            off = vary(pop, rank, dis, cv, p.lb_l, p.ub_l)
            off_fit, off_cv = self._lower_evaluate(xu, off)
            pop, fit, rank, dis, cv = select(
                torch.cat((pop, off), 1), torch.cat((fit, off_fit), 1), self.lower_pop_size, torch.cat((cv, off_cv), 1)
            )
        return pop, fit, rank, cv

    def _response_candidates(self, xu, xl, lower_fit, lower_rank, lower_cv):
        """Fixed-shape candidate pool: only feasible lower rank zero is eligible."""
        leaders = xu[:, None, :].expand(*xl.shape[:-1], self.problem.d_u)
        upper_fit, upper_cv = parse_evaluate(self.problem.evaluate_upper(leaders, xl))
        if upper_cv is None:
            upper_cv = upper_fit.new_zeros(*upper_fit.shape[:-1], 0)
        eligible = (lower_rank == 0) & (lower_cv.sum(-1) == 0)
        excluded = torch.where(eligible, 0.0, torch.inf).to(upper_fit.dtype).unsqueeze(-1)
        cv = torch.cat((excluded, upper_cv, lower_cv), -1).flatten(0, 1)
        joint = torch.cat((leaders, xl), -1).flatten(0, 1)
        return joint, upper_fit.flatten(0, 1), lower_fit.flatten(0, 1), cv

    def _nested_candidates(self, xu):
        candidates = self._response_candidates(xu, *self._solve_lower(xu))
        n = xu.shape[0] * self.lower_pop_size
        self.upper_evaluations = self.upper_evaluations + n
        self.lower_evaluations = self.lower_evaluations + n * (self.lower_generations + 1)
        return candidates

    def _survive(self, pop, fit, lower_fit, cv):
        payload = torch.cat((pop, lower_fit), -1)
        payload, self.fit, self.rank, self.dis, self.cv = nd_environmental_selection(payload, fit, self.pop_size, cv)
        self.pop, self.lower_fit = payload[:, : self.dim], payload[:, self.dim :]

    def init_step(self):
        if self._initialized:
            raise RuntimeError("NestedNSGA2 is already initialized.")
        self._survive(*self._nested_candidates(self.pop[:, : self.problem.d_u]))
        self._initialized = True

    def step(self):
        if not self._initialized:
            raise RuntimeError("Call init_step() before step().")
        p = self.problem
        xu = self._repair(self._vary(self.pop[:, : p.d_u], self.rank, self.dis, self.cv, p.lb_u, p.ub_u))
        pop, fit, lower_fit, cv = self._nested_candidates(xu)
        self._survive(
            torch.cat((self.pop, pop)),
            torch.cat((self.fit, fit)),
            torch.cat((self.lower_fit, lower_fit)),
            torch.cat((self.cv, cv)),
        )
        self.generation = self.generation + 1
