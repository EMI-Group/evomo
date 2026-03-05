from typing import Dict, Optional, Tuple

import torch
from evox.core import Problem

from evomo.utils import at_least_2d


def default_shape(problem, n: int) -> Dict[str, Tuple[int, ...]]:
    d = problem.d
    return dict(
        F=(n, problem.m),
        G=(n, problem.n_iq),
        H=(n, problem.n_eq),
        CV=(n,),
        dF=(n, problem.m, d),
        dG=(n, problem.n_iq, d),
        dH=(n, problem.n_eq, d),
    )


class CMOP(Problem):
    """
    Base class for Constrained Multi-objective Optimization Problems (CMOP).
    """

    def __init__(
        self,
        d: int,
        m: int,
        n_iq: int,
        n_eq: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        *,
        scale_d: bool = False,
        constr_eq_eps: float = 1e-4,
        return_cv: bool = False,
        requires_kwargs: bool = False,
        callback=None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.device = device if device is not None else torch.get_default_device()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.d = d
        self.m = m
        self.n_iq = n_iq
        self.n_eq = n_eq
        self.n_constraints = self.n_iq + self.n_eq
        self.constr_eq_eps = constr_eq_eps

        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # write to self
        self.lb = lb.to(device=self.device)
        self.ub = ub.to(device=self.device)

        self.scale_d = scale_d
        self.return_cv = return_cv

        self.requires_kwargs = requires_kwargs
        self.callback = callback

    def name(self) -> str:
        return self.__class__.__name__

    def has_bounds(self) -> bool:
        return self.lb is not None and self.ub is not None

    def has_constraints(self) -> bool:
        return self.n_constraints > 0

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lb, self.ub

    def __str__(self):
        return f"# name: {self.name()}\n# d: {self.d}\n# m: {self.m}\n# n_ieq_constr: {self.n_iq}\n# n_eq: {self.n_eq}\n"

    def _scale_d(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.lb) / (self.ub - self.lb)

    def _unscale_d(self, X: torch.Tensor) -> torch.Tensor:
        return self.lb + X * (self.ub - self.lb)

    def evaluate(self, X: torch.Tensor):
        """
        Evaluate the population X.
        Returns a tuple (fitness, constraint_violation).
        """
        X2d, only_single_value = at_least_2d(X)
        X2d = torch.as_tensor(X2d, device=self.device, dtype=self.dtype)
        assert X2d.shape[1] == self.d

        X_eval = self._unscale_d(X2d) if self.scale_d else X2d
        Y = self.fn(X_eval)  # (N, m + n_iq + n_eq)

        F = Y[:, : self.m]
        G = Y[:, self.m : self.m + self.n_iq] if self.n_iq > 0 else None
        H = Y[:, self.m + self.n_iq :] if self.n_eq > 0 else None

        cv = self._compute_cv(G, H)

        if only_single_value:
            return F[0], cv[0]
        return F, cv

    def _compute_cv(self, G: Optional[torch.Tensor], H: Optional[torch.Tensor]) -> torch.Tensor:
        if G is None and H is None:
            return torch.zeros((0, 0), device=self.device)

        cvs = []
        if G is not None and G.numel() > 0:
            cvs.append(torch.clamp(G, min=0))
        if H is not None and H.numel() > 0:
            cvs.append(torch.clamp(torch.abs(H) - self.constr_eq_eps, min=0))

        if not cvs:
            N = G.shape[0] if G is not None else H.shape[0]
            return torch.zeros((N, 0), device=self.device)

        return torch.cat(cvs, dim=1)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def pf(self):
        raise NotImplementedError()
