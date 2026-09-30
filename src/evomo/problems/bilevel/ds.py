"""DS1--DS5, using the equations and domains in Deb and Sinha (2010).

The 2008 construction paper numbers different problems; it must not be used to
assign the later DS1--DS5 names. DS3's leader x1 has lattice spacing 0.1.
DS4/DS5 retain the printed [-1, 1] domain for the first follower variable;
the commonly distributed [0, 1] implementation is a different domain.
"""

import math

import torch

from .base import BLMOP


def _front_2d(points):
    """Reference-point filtering in O(N log N), without a pairwise matrix."""
    order = torch.argsort(points[:, 1], stable=True)
    order = order[torch.argsort(points[order, 0], stable=True)]
    points = points[order]
    best = torch.cummin(points[:, 1], dim=0).values
    keep = torch.cat((torch.ones(1, device=points.device, dtype=torch.bool), points[1:, 1] < best[:-1]))
    return points[keep]


class DS(BLMOP):
    def __init__(
        self,
        prob_id: int,
        k: int | None = None,
        ell: int = 4,
        *,
        r: float | None = None,
        alpha: float | None = None,
        gamma: float | None = None,
        tau: float = 1.0,
        **kwargs,
    ):
        if not isinstance(prob_id, int) or isinstance(prob_id, bool) or prob_id not in (1, 2, 3, 4, 5):
            raise ValueError("DS prob_id must be in 1..5.")
        k = (10 if prob_id <= 3 else 5) if k is None else k
        minimum = {1: 4, 2: 2, 3: 3, 4: 1, 5: 1}[prob_id]
        if not isinstance(k, int) or isinstance(k, bool) or k < minimum:
            raise ValueError(f"DS{prob_id} requires integer k >= {minimum}.")
        if not isinstance(ell, int) or isinstance(ell, bool) or ell < 1:
            raise ValueError("ell must be a positive integer.")
        self.prob_id, self.k, self.ell = prob_id, k, ell
        self.r = {1: 0.1, 2: 0.25, 3: 0.2, 4: 0.1, 5: 0.1}[prob_id] if r is None else r
        self.alpha = (5.0 if prob_id == 5 else 1.0) if alpha is None else alpha
        self.gamma = (4.0 if prob_id == 2 else 1.0) if gamma is None else gamma
        self.tau = tau
        if not all(math.isfinite(v) for v in (self.r, self.alpha, self.gamma, tau)):
            raise ValueError("DS parameters must be finite.")
        if self.r <= 0 or self.alpha <= 0 or self.gamma <= 0:
            raise ValueError("r, alpha and gamma must be positive.")
        if prob_id <= 3:
            du, dl = k, k
            lower_u = {1: [1] + [-k] * (k - 1), 2: [0.001] + [-k] * (k - 1), 3: [0] * k}[prob_id]
            upper_u = [4] + [k] * (k - 1) if prob_id == 1 else [k] * k
            lower_l, upper_l = [-k] * k, [k] * k
        else:
            du, dl = 1, k + ell
            lower_u, upper_u = [1], [2]
            lower_l, upper_l = [-1] + [-dl] * (dl - 1), [1] + [dl] * (dl - 1)
        super().__init__(
            du,
            dl,
            2,
            2,
            lower_u,
            upper_u,
            lower_l,
            upper_l,
            n_iq_u=int(prob_id >= 3),
            n_iq_l=int(prob_id == 3),
            n_eq_u=int(prob_id == 3),
            **kwargs,
        )

    def _objectives_upper(self, y, x):
        y1, x1 = y[..., 0], x[..., 0]
        if self.prob_id in (1, 2):
            linked = self.tau * (x[..., 1:] - y[..., 1:]).square().sum(-1)
            angle = self.gamma * torch.pi / 2 * x1 / y1
            if self.prob_id == 1:
                target = torch.arange(1, self.k, device=y.device, dtype=y.dtype) / 2
                penalty = (y[..., 1:] - target).square().sum(-1) + linked
                a = 1 + self.r - torch.cos(self.alpha * torch.pi * y1) + penalty
                b = 1 + self.r - torch.sin(self.alpha * torch.pi * y1) + penalty
            else:
                c, s = math.cos(0.2 * math.pi), math.sin(0.2 * math.pi)
                bump = torch.sqrt(torch.abs(0.02 * torch.sin(5 * torch.pi * y1)))
                a = torch.where(y1 > 1, y1 - (1 - c), c * y1 + s * bump)
                b = torch.where(y1 > 1, 0.1 * (y1 - 1) - s, -s * y1 + c * bump)
                tail = y[..., 1:]
                penalty = (tail.square() + 10 * (1 - torch.cos(torch.pi / self.k * tail))).sum(-1) + linked
                a, b = a + penalty, b + penalty
            a, b = a - self.r * torch.cos(angle), b - self.r * torch.sin(angle)
        elif self.prob_id == 3:
            target = torch.arange(3, self.k + 1, device=y.device, dtype=y.dtype) / 2
            penalty = (y[..., 2:] - target).square().sum(-1) + self.tau * (x[..., 2:] - y[..., 2:]).square().sum(-1)
            radius = 0.1 + 0.15 * torch.abs(torch.sin(2 * torch.pi * (y1 - 0.1)))
            # 4*atan2 is equivalent to the printed 4*atan ratio modulo 2*pi,
            # and avoids dividing by zero on the circle's coordinate axes.
            angle = 4 * torch.atan2(y[..., 1] - x[..., 1], y1 - x1)
            a = y1 + penalty - radius * torch.cos(angle)
            b = y[..., 1] + penalty - radius * torch.sin(angle)
        else:
            factor = (1 + x[..., 1 : self.k].square().sum(-1)) * y1
            a, b = (1 - x1) * factor, x1 * factor
        return torch.stack((a, b), -1)

    def _objectives_lower(self, y, x):
        x1 = x[..., 0]
        if self.prob_id == 1:
            delta = x[..., 1:] - y[..., 1:]
            a = x1.square() + (delta.square() + 10 * (1 - torch.cos(torch.pi / self.k * delta))).sum(-1)
            b = (x - y).square().sum(-1) + 10 * torch.abs(torch.sin(torch.pi / self.k * delta)).sum(-1)
        elif self.prob_id == 2:
            a = x1.square() + (x[..., 1:] - y[..., 1:]).square().sum(-1)
            weight = torch.arange(1, self.k + 1, device=x.device, dtype=x.dtype)
            b = ((x - y).square() * weight).sum(-1)
        elif self.prob_id == 3:
            penalty = (x[..., 2:] - y[..., 2:]).square().sum(-1)
            a, b = x1 + penalty, x[..., 1] + penalty
        else:
            factor = (1 + x[..., self.k :].square().sum(-1)) * y[..., 0]
            a, b = (1 - x1) * factor, x1 * factor
        return torch.stack((a, b), -1)

    def _constraints_upper(self, y, x):
        _, h = self._empty_constraints(y)
        if self.prob_id == 3:
            g = (1 - y[..., 0].square() - y[..., 1]).unsqueeze(-1)
            h = (y[..., 0] - torch.round(10 * y[..., 0]) / 10).unsqueeze(-1)
        elif self.prob_id in (4, 5):
            a, b = (1 - x[..., 0]) * y[..., 0], x[..., 0] * y[..., 0]
            residual = 1 - a - b / 2
            if self.prob_id == 5:
                residual = 2 - a - b / 2 - torch.floor(self.alpha * a + 0.2) / self.alpha
            g = residual.unsqueeze(-1)
        else:
            return self._empty_constraints(y)
        return g, h

    def _constraints_lower(self, y, x):
        if self.prob_id != 3:
            return self._empty_constraints(y)
        _, h = self._empty_constraints(y)
        return ((x[..., :2] - y[..., :2]).square().sum(-1) - self.r**2).unsqueeze(-1), h

    def lower_ps(self, xu):
        if xu.ndim < 1 or xu.shape[-1] != self.d_u:
            raise ValueError(f"Expected xu[..., {self.d_u}].")
        if self.prob_id >= 4:
            # With the printed negative x1 domain, nonzero tail penalties can
            # also be efficient. A zero-tail line is not the entire response set.
            return super().lower_ps(xu)
        t = torch.linspace(0, 1, self.ref_num, device=xu.device, dtype=xu.dtype)
        if self.prob_id <= 2:
            x1 = xu[..., 0, None] * t
            tail = xu[..., None, 1:].expand(*x1.shape, self.d_l - 1)
            return torch.cat((x1.unsqueeze(-1), tail), -1)
        angle = t * torch.pi / 2
        first = xu[..., None, :2] - self.r * torch.stack((torch.cos(angle), torch.sin(angle)), -1)
        tail = xu[..., None, 2:].expand(*first.shape[:-1], self.d_l - 2)
        return torch.cat((first, tail), -1)

    def pf(self):
        angle = torch.linspace(0, torch.pi / 2, self.ref_num, device=self.device, dtype=self.dtype)
        if self.prob_id == 1 and self.alpha == self.gamma == self.tau == 1:
            return (1 + self.r) * (1 - torch.stack((torch.cos(angle), torch.sin(angle)), -1))
        if self.prob_id == 2 and self.gamma == 4 and self.tau == 1:
            y = torch.tensor([0.001, 0.2, 0.4, 0.6, 0.8, 1], device=self.device, dtype=self.dtype)
            c, s = math.cos(0.2 * math.pi), math.sin(0.2 * math.pi)
            bump = torch.sqrt(torch.abs(0.02 * torch.sin(5 * torch.pi * y)))
            centers = torch.stack((c * y + s * bump, -s * y + c * bump), -1)
            points = centers[:, None, :] - self.r * torch.stack((torch.cos(angle), torch.sin(angle)), -1)
            return _front_2d(points.reshape(-1, 2))
        if self.prob_id == 3 and self.tau == 1:
            y1 = torch.arange(10 * self.k + 1, device=self.device, dtype=self.dtype) / 10
            y2 = (1 - y1.square()).clamp_min(0)
            centers = torch.stack((y1, y2), -1)
            radius = 0.1 + 0.15 * torch.abs(torch.sin(2 * torch.pi * (y1 - 0.1)))
            points = centers[:, None, :] - radius[:, None, None] * torch.stack((torch.cos(angle), torch.sin(angle)), -1)
            return _front_2d(points.reshape(-1, 2))
        # Published DS4/DS5 PF formulas contradict their printed domain/constraint.
        return super().pf()


class DS1(DS):
    def __init__(self, k: int = 10, **kwargs):
        super().__init__(1, k, **kwargs)


class DS2(DS):
    def __init__(self, k: int = 10, **kwargs):
        super().__init__(2, k, **kwargs)


class DS3(DS):
    def __init__(self, k: int = 10, **kwargs):
        super().__init__(3, k, **kwargs)


class DS4(DS):
    def __init__(self, k: int = 5, ell: int = 4, **kwargs):
        super().__init__(4, k, ell, **kwargs)


class DS5(DS):
    def __init__(self, k: int = 5, ell: int = 4, **kwargs):
        super().__init__(5, k, ell, **kwargs)
