"""Tensor interfaces for bilevel benchmarks; evaluation never solves the follower."""

import math
from typing import NamedTuple

import torch
from evox.core import Problem


class BilevelEvaluation(NamedTuple):
    upper_fitness: torch.Tensor
    lower_fitness: torch.Tensor
    upper_cv: torch.Tensor
    lower_cv: torch.Tensor


class BLMOP(Problem):
    """Base for bilevel multiobjective problems.

    All objectives are minimized. Inequalities use ``G <= 0`` and equalities
    ``H == 0``. CV contains one nonnegative value per constraint, with shape
    ``(*batch, 0)`` for an unconstrained level. Bounds are not added to CV.
    ``evaluate(z)`` evaluates the upper level at ``z = [xu, xl]``; it does not
    certify lower-level optimality. Inputs preserve their device and dtype and
    may have broadcastable leading dimensions (including an empty batch).
    """

    def __init__(
        self,
        d_u: int,
        d_l: int,
        m_u: int,
        m_l: int,
        lb_u,
        ub_u,
        lb_l,
        ub_l,
        *,
        n_iq_u: int = 0,
        n_iq_l: int = 0,
        n_eq_u: int = 0,
        n_eq_l: int = 0,
        ref_num: int = 1000,
        constr_eq_eps: float = 1e-4,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (d_u, d_l, m_u, m_l, ref_num)):
            raise ValueError("Dimensions, objective counts and ref_num must be positive integers.")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (n_iq_u, n_iq_l, n_eq_u, n_eq_l)):
            raise ValueError("Constraint counts must be nonnegative integers.")
        if not math.isfinite(constr_eq_eps) or constr_eq_eps < 0:
            raise ValueError("constr_eq_eps must be finite and nonnegative.")
        self.d_u, self.d_l, self.m_u, self.m_l = d_u, d_l, m_u, m_l
        self.d, self.dim, self.m = d_u + d_l, d_u + d_l, m_u
        self.n_iq_u, self.n_iq_l = n_iq_u, n_iq_l
        self.n_eq_u, self.n_eq_l = n_eq_u, n_eq_l
        self.ref_num, self.constr_eq_eps = ref_num, constr_eq_eps
        device = torch.get_default_device() if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("Bounds require a floating-point dtype.")
        for name, values, dim in (("lb_u", lb_u, d_u), ("ub_u", ub_u, d_u), ("lb_l", lb_l, d_l), ("ub_l", ub_l, d_l)):
            bound = torch.as_tensor(values, device=device, dtype=dtype).clone()
            if bound.shape != (dim,) or not torch.isfinite(bound).all():
                raise ValueError(f"{name} must contain {dim} finite bounds.")
            self.register_buffer(name, bound)
        if not ((self.lb_u < self.ub_u).all() and (self.lb_l < self.ub_l).all()):
            raise ValueError("Every lower bound must be strictly below its upper bound.")

    @property
    def device(self):
        return self.lb_u.device

    @property
    def dtype(self):
        return self.lb_u.dtype

    @property
    def lb(self):
        return torch.cat((self.lb_u, self.lb_l))

    @property
    def ub(self):
        return torch.cat((self.ub_u, self.ub_l))

    def bounds(self):
        return self.lb, self.ub

    def name(self):
        return type(self).__name__

    def split(self, z: torch.Tensor):
        if z.ndim < 1 or z.shape[-1] != self.d:
            raise ValueError(f"Expected last dimension {self.d}, got {tuple(z.shape)}.")
        return z[..., : self.d_u], z[..., self.d_u :]

    def _inputs(self, xu, xl):
        if xu.ndim < 1 or xl.ndim < 1 or xu.shape[-1] != self.d_u or xl.shape[-1] != self.d_l:
            raise ValueError(f"Expected xu[..., {self.d_u}] and xl[..., {self.d_l}].")
        if xu.device != xl.device or xu.dtype != xl.dtype or not xu.is_floating_point():
            raise ValueError("xu and xl must have the same floating-point dtype and device.")
        batch = torch.broadcast_shapes(xu.shape[:-1], xl.shape[:-1])
        return xu.expand(*batch, self.d_u), xl.expand(*batch, self.d_l)

    def _empty_constraints(self, xu):
        empty = xu.new_empty((*xu.shape[:-1], 0))
        return empty, empty

    def _constraints_upper(self, xu, xl):
        return self._empty_constraints(xu)

    def _constraints_lower(self, xu, xl):
        return self._empty_constraints(xu)

    def constraints_upper(self, xu, xl):
        """Return raw inequality and equality residuals ``(G_u, H_u)``."""
        return self._constraints_upper(*self._inputs(xu, xl))

    def constraints_lower(self, xu, xl):
        """Return raw inequality and equality residuals ``(G_l, H_l)``."""
        return self._constraints_lower(*self._inputs(xu, xl))

    def _cv(self, g, h):
        return torch.cat((g.clamp_min(0), (h.abs() - self.constr_eq_eps).clamp_min(0)), dim=-1)

    def _objectives_upper(self, xu, xl):
        raise NotImplementedError

    def _objectives_lower(self, xu, xl):
        raise NotImplementedError

    def evaluate_upper(self, xu, xl):
        xu, xl = self._inputs(xu, xl)
        fitness = self._objectives_upper(xu, xl)
        if self.n_iq_u + self.n_eq_u:
            return fitness, self._cv(*self._constraints_upper(xu, xl))
        return fitness

    def evaluate_lower(self, xu, xl):
        xu, xl = self._inputs(xu, xl)
        fitness = self._objectives_lower(xu, xl)
        if self.n_iq_l + self.n_eq_l:
            return fitness, self._cv(*self._constraints_lower(xu, xl))
        return fitness

    def evaluate(self, z):
        return self.evaluate_upper(*self.split(z))

    def evaluate_all(self, z):
        """Return named upper/lower objectives and CV, including empty CV tensors."""
        xu, xl = self._inputs(*self.split(z))
        return BilevelEvaluation(
            self._objectives_upper(xu, xl),
            self._objectives_lower(xu, xl),
            self._cv(*self._constraints_upper(xu, xl)),
            self._cv(*self._constraints_lower(xu, xl)),
        )

    def lower_problem(self, xu):
        """Bind one leader vector to a standard EvoX follower Problem."""
        return LowerLevelProblem(self, xu)

    def pf(self):
        """Return a verified leader reference front, if available."""
        raise NotImplementedError(f"No verified upper front for {self.name()} and these parameters.")

    def lower_ps(self, xu):
        """Return follower reference decisions with shape ``(*batch, ref_num, d_l)``.

        Scalar follower problems may return one representative optimum. This
        method is optional and never runs an optimizer or certifies input points.
        """
        raise NotImplementedError(f"No analytic follower reference set for {self.name()}.")

    def lower_pf(self, xu):
        xl = self.lower_ps(xu)
        return self._objectives_lower(*self._inputs(xu.unsqueeze(-2), xl))


class LowerLevelProblem(Problem):
    """A follower problem conditioned on a fixed leader; supports standard workflows."""

    def __init__(self, problem: BLMOP, xu: torch.Tensor):
        super().__init__()
        if xu.shape != (problem.d_u,) or not xu.is_floating_point():
            raise ValueError(f"Expected one floating-point leader vector of shape ({problem.d_u},).")
        self.problem = problem
        self.register_buffer("xu", xu.detach().clone())
        self.d, self.m = problem.d_l, problem.m_l

    @property
    def lb(self):
        return self.problem.lb_l

    @property
    def ub(self):
        return self.problem.ub_l

    def evaluate(self, xl):
        return self.problem.evaluate_lower(self.xu, xl)

    def pf(self):
        return self.problem.lower_pf(self.xu)
