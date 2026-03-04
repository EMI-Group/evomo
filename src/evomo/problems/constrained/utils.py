from evox.core import Problem
from typing import Optional, Tuple, Iterable, Dict, Any, List
import torch
import os
import numpy as np

def at_least_2d(X: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if X.ndim == 1:
        return X.unsqueeze(0), True
    elif X.ndim == 2:
        return X, False
    else:
        raise ValueError(f"Expected 1D/2D torch.Tensor, got {tuple(X.shape)}")

def get_pareto_front(f: torch.Tensor) -> torch.Tensor:
    """
    Return the non-dominated set (Pareto front) of the given objectives.
    """
    # f shape: (N, M)
    # Simple O(N^2) implementation
    x_expanded = f.unsqueeze(1)  # (N, 1, M)
    y_expanded = f.unsqueeze(0)  # (1, N, M)
    
    # x dominates y if x <= y and at least one x < y
    less_equal = (x_expanded <= y_expanded).all(dim=2)
    strictly_less = (x_expanded < y_expanded).any(dim=2)
    domination = less_equal & strictly_less  # (N, N)
    
    # an element j is dominated if any i dominates it
    is_dominated = domination.any(dim=0)
    return f[~is_dominated]


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
        return (f"# name: {self.name()}\n"
                f"# d: {self.d}\n"
                f"# m: {self.m}\n"
                f"# n_ieq_constr: {self.n_iq}\n"
                f"# n_eq: {self.n_eq}\n")


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
        
        F = Y[:, :self.m]
        G = Y[:, self.m:self.m + self.n_iq] if self.n_iq > 0 else None
        H = Y[:, self.m + self.n_iq:] if self.n_eq > 0 else None
        
        cv = self._compute_cv(G, H)
        
        if only_single_value:
            return F[0], cv[0]
        return F, cv

    # Calculate total constraint violation
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
        """
        Return the Pareto front for the problem.

        :return: A tensor representing the Pareto front.
        """
        raise NotImplementedError()

def load_pareto_front_from_file(fname, device=None, dtype=torch.float32):
    """
    Load Pareto front from pf/<fname>.

    Parameters
    ----------
    fname : str
        File name inside pf directory.
    device : torch.device or str, optional
        Target device (e.g. "cuda", torch.device("cuda")).
        If None, tensor stays on CPU.
    dtype : torch.dtype
        Tensor dtype (default: torch.float32).

    Returns
    -------
    torch.Tensor
        Pareto front tensor sorted by the first column.
    """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(current_dir, "pf", fname)

    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Pareto front file not found: {fname}")

    # load using numpy
    pf_np = np.loadtxt(fname)

    # convert to torch tensor
    pf = torch.as_tensor(pf_np, dtype=dtype)

    # sort by first column
    pf = pf[pf[:, 0].argsort()]

    # move to target device if specified
    if device is not None:
        pf = pf.to(device)

    return pf

