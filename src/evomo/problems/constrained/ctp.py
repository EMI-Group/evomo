import numpy as np
from .utils import CMOP, load_pareto_front_from_file
import torch


__all__ = ['CTP1', 'CTP2', 'CTP3', 'CTP4', 'CTP5', 'CTP6', 'CTP7', 'CTP8']


def ctp(X, theta, a, b, c, d, e):
    """
    Generator for constrained multiobjective problems CTP.
    """

    X = torch.atleast_2d(X) # default expand first dimension
    theta = torch.as_tensor(theta, dtype=X.dtype, device=X.device)

    # Distance values
    h = X[:, 1:] ** 2 - 10 * torch.cos(2 * torch.pi * X[:, 1:])
    g = 1 + 10 * (X.shape[1] - 1) + torch.sum(h, dim=1, keepdim=True)

    # Objective values
    f1 = X[:, [0]]
    f2 = g * (1 - torch.sqrt(f1 / g))

    # Constraint values
    h1 = torch.cos(theta) * (f2 - e) - torch.sin(theta) * f1
    h2 = torch.sin(theta) * (f2 - e) + torch.cos(theta) * f1
    h3 = a * torch.abs(torch.sin(b * torch.pi * (h2 ** c))) ** d
    c  = h3 - h1

    return torch.cat([f1, f2, c],dim=1)


def ctp1(X):
    """
    CTP1 constrained multiobjective optimization problem.
    """

    X = torch.atleast_2d(X)
 
    # Distance values
    h = X[:, 1:].square() - 10 * torch.cos(2 * torch.pi * X[:, 1:])
    g = 1 + 10 * (X.shape[1] - 1) + torch.sum(h, dim=1, keepdim=True)

    # Objective values
    f1 = X[:, [0]]
    f2 = g * torch.exp(-f1 / g)

    # Constraint values
    c1 = f2 - 0.858 * torch.exp(-0.541 * f1)
    c2 = f2 - 0.728 * torch.exp(-0.295 * f1)

    return torch.cat([f1, f2, -c1, -c2],dim=1)

def ctp2(X):
    """
    CTP2 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.2 * torch.pi, 0.2, 10, 1, 6, 1
    return ctp(X, theta, a, b, c, d, e)


def ctp3(X):
    """
    CTP3 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.2 * torch.pi, 0.1, 10, 1, 0.5, 1
    return ctp(X, theta, a, b, c, d, e)


def ctp4(X):
    """
    CTP4 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.2 * torch.pi, 0.75, 10, 1, 0.5, 1
    return ctp(X, theta, a, b, c, d, e)


def ctp5(X):
    """
    CTP5 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.2 * torch.pi, 0.1, 10, 2, 0.5, 1
    return ctp(X, theta, a, b, c, d, e)


def ctp6(X):
    """
    CTP6 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = 0.1 * torch.pi, 40, 0.5, 1, 2, -2
    return ctp(X, theta, a, b, c, d, e)


def ctp7(X):
    """
    CTP7 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.05 * torch.pi, 40, 5, 1, 6, 0
    return ctp(X, theta, a, b, c, d, e)


def ctp8(X):
    """
    CTP8 constrained multiobjective optimization problem.
    """
    theta1, a1, b1, c1, d1, e1 = -0.05 * torch.pi, 40, 2, 1, 6, 0
    theta2, a2, b2, c2, d2, e2 = 0.1 * torch.pi, 40, 0.5, 1, 2, -2
    Fk = ctp(X, theta1, a1, b1, c1, d1, e1)
    C = ctp(X, theta2, a2, b2, c2, d2, e2)
    return torch.cat([Fk, C[:, [2]]],dim=1)


# Generator #


class CTP(CMOP):
    """
    The CPT test suite generator of constrained multibjecitve problems CTP1-CTP8.

    Parameters
    ----------
    :param prob_id (int): CTP problem id.
    :param d (int): Dimension of the decision space.
    :param m (int): Number of the objectives.

    Raise
    -----
    :raise ValueError: If prob_id is not in {1, ..., 8} or d is smaller than 2.

    References
    ----------
    [Deb2001] K. Deb, A. Pratap, T. Meyarivan, "Constrained Test Problems for Multi-objective
    Evolutionary Optimization," Evolutionary Multi-Criterion Optimization (EMO 2001), pp. 284-298,
    doi: 10.1007/3-540-44719-9_20.
    """

    _prob = {
        1: ctp1,
        2: ctp2,
        3: ctp3,
        4: ctp4,
        5: ctp5,
        6: ctp6,
        7: ctp7,
        8: ctp8
    }

    def __init__(self, prob_id, m=2, d=2,**kwargs):

        if prob_id not in set(range(1, 9)):
            raise ValueError("Please select a valid prob id.")

        # Set fn
        self.fn = self._prob[prob_id]
        self.device = kwargs.pop("device", torch.get_default_device())

        # Set dim
        if d < 2:
            raise ValueError("Please select a larger value for d (>= 2).")

        # Set obj
        if m != 2:
            raise ValueError("CTP suite can only be instantiated with 2 objectives.")

        n_iq = 2 if prob_id in {1, 8} else 1

        # Set lb, ub
        lb = torch.tensor([0.] + [-5.] * (d - 1),device=self.device)
        ub = torch.tensor([1.] + [5.] * (d - 1),device=self.device)

        super(CTP, self).__init__(d=d,
                                  m=m,
                                  n_iq=n_iq,
                                  n_eq=0,
                                  lb=lb,
                                  ub=ub,
                                  **kwargs)

    def fn(self, X):
        return self.fn(X)

# Pymoo convention #

class CTP1(CTP):
    """
    CTP1 constrained multiobjective optimization problem.
    """

    def __init__(self, m=2, d=5, **kwargs):
        super(CTP1, self).__init__(prob_id=1, m=m, d=d, **kwargs)

    def pf(self):
        return load_pareto_front_from_file(f"CTP/{self.fn.__name__.lower()}_M{self.m}_D{self.d}.pf", 
                                            device=self.device)


class CTP2(CTP):
    """
    CTP2 constrained multiobjective optimization problem.
    """

    def __init__(self, m=2, d=5, **kwargs):
        super(CTP2, self).__init__(prob_id=2, m=m, d=d, **kwargs)

    def pf(self):
        return load_pareto_front_from_file(f"CTP/{self.fn.__name__.lower()}_M{self.m}_D{self.d}.pf",
                                            device=self.device)


class CTP3(CTP):
    """
    CTP3 constrained multiobjective optimization problem.
    """

    def __init__(self, m=2, d=5, **kwargs):
        super(CTP3, self).__init__(prob_id=3, m=m, d=d, **kwargs)

    def pf(self):
        return load_pareto_front_from_file(f"CTP/{self.fn.__name__.lower()}_M{self.m}_D{self.d}.pf",
                                            device=self.device)


class CTP4(CTP):
    """
    CTP4 constrained multiobjective optimization problem.
    """

    def __init__(self, m=2, d=5, **kwargs):
        super(CTP4, self).__init__(prob_id=4, m=m, d=d, **kwargs)

    def pf(self):
        return load_pareto_front_from_file(f"CTP/{self.fn.__name__.lower()}_M{self.m}_D{self.d}.pf",
                                            device=self.device)


class CTP5(CTP):
    """
    CTP5 constrained multiobjective optimization problem.
    """

    def __init__(self, m=2, d=5, **kwargs):
        super(CTP5, self).__init__(prob_id=5, m=m, d=d, **kwargs)

    def pf(self):
        return load_pareto_front_from_file(f"CTP/{self.fn.__name__.lower()}_M{self.m}_D{self.d}.pf",
                                            device=self.device)


class CTP6(CTP):
    """
    CTP6 constrained multiobjective optimization problem.
    """

    def __init__(self, m=2, d=5, **kwargs):
        super(CTP6, self).__init__(prob_id=6, m=m, d=d, **kwargs)

    def pf(self):
        return load_pareto_front_from_file(f"CTP/{self.fn.__name__.lower()}_M{self.m}_D{self.d}.pf",
                                            device=self.device)


class CTP7(CTP):
    """
    CTP7 constrained multiobjective optimization problem.
    """

    def __init__(self, m=2, d=5, **kwargs):
        super(CTP7, self).__init__(prob_id=7, m=m, d=d, **kwargs)

    def pf(self):
        return load_pareto_front_from_file(f"CTP/{self.fn.__name__.lower()}_M{self.m}_D{self.d}.pf",
                                            device=self.device)


class CTP8(CTP):
    """
    CTP8 constrained multiobjective optimization problem.
    """

    def __init__(self, m=2, d=5, **kwargs):
        super(CTP8, self).__init__(prob_id=8, m=m, d=d, **kwargs)

    def pf(self):
        return load_pareto_front_from_file(f"CTP/{self.fn.__name__.lower()}_M{self.m}_D{self.d}.pf",
                                            device=self.device)
