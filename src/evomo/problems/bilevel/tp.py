"""TP1--TP4 from Deb and Sinha's bilevel multiobjective benchmark study.

Reference: An Efficient and Accurate Solution Methodology for Bilevel
Multi-objective Programming Problems Using a Hybrid Evolutionary-Local-Search
Algorithm (2010),
also reproduced in Sinha's thesis, pp. 80--82. These are NOT PlatEMO's robust TP
suite or BLEAQ2's single-objective TP suite. The printed definitions use
(x2 - 1)**2 in TP3 and minimization at the lower level in TP4.
"""

import math

import torch

from .base import BLMOP


class TP(BLMOP):
    def __init__(self, prob_id: int, d_l: int | None = None, *, bound: float = 150.0, **kwargs):
        if not isinstance(prob_id, int) or isinstance(prob_id, bool) or prob_id not in (1, 2, 3, 4):
            raise ValueError("TP prob_id must be in 1..4.")
        dimensions = {1: (1, 2), 2: (1, 14), 3: (1, 2), 4: (2, 3)}
        du, dl = dimensions[prob_id]
        if d_l is not None:
            if prob_id != 2 and d_l != dl:
                raise ValueError("Only TP2 has a configurable follower dimension.")
            dl = d_l
        if not isinstance(dl, int) or isinstance(dl, bool) or dl < 1:
            raise ValueError("d_l must be a positive integer.")
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("TP4's computational bound must be finite and positive.")
        low_u, high_u, low_l, high_l = {
            1: (0, 1, -1, 1),
            2: (-1, 2, -1, 2),
            3: (0, 10, 0, 10),
            4: (0, bound, 0, bound),
        }[prob_id]
        self.prob_id = prob_id
        super().__init__(
            du,
            dl,
            2,
            2,
            [low_u] * du,
            [high_u] * du,
            [low_l] * dl,
            [high_l] * dl,
            n_iq_u={1: 1, 2: 0, 3: 1, 4: 2}[prob_id],
            n_iq_l={1: 1, 2: 0, 3: 4, 4: 3}[prob_id],
            **kwargs,
        )

    def _objectives_upper(self, y, x):
        y1, x1, x2 = y[..., 0], x[..., 0], x[..., min(1, self.d_l - 1)]
        if self.prob_id == 1:
            a, b = x1 - y1, x2
        elif self.prob_id == 2:
            g = (x1 - 1).square() + x[..., 1:].square().sum(-1)
            a, b = g + y1.square(), g + (y1 - 1).square()
        elif self.prob_id == 3:
            a = x1 + x2.square() + y1 + torch.sin(x1 + y1).square()
            b = torch.cos(x2) * (0.1 + y1) * torch.exp(-x1 / (0.1 + x2))
        else:
            y2, x3 = y[..., 1], x[..., 2]
            a = -(y1 + 9 * y2 + 10 * x1 + x2 + 3 * x3)
            b = -(9 * y1 + 2 * y2 + 2 * x1 + 7 * x2 + 4 * x3)
        return torch.stack((a, b), -1)

    def _objectives_lower(self, y, x):
        y1, x1, x2 = y[..., 0], x[..., 0], x[..., min(1, self.d_l - 1)]
        if self.prob_id == 1:
            a, b = x1, x2
        elif self.prob_id == 2:
            g = x[..., 1:].square().sum(-1)
            a, b = x1.square() + g, (x1 - y1).square() + g
        elif self.prob_id == 3:
            a = ((x1 - 2).square() + (x2 - 1).square()) / 4 + (x2 * y1 + (5 - y1).square()) / 16 + torch.sin(x2 / 10)
            b = (x1.square() + (x2 - 6).pow(4) - 2 * x1 * y1 - (5 - y1).square()) / 80
        else:
            # The source defines argmin at the lower level and maximization above.
            y2, x3 = y[..., 1], x[..., 2]
            a = 4 * y1 + 6 * y2 + 7 * x1 + 4 * x2 + 8 * x3
            b = 6 * y1 + 4 * y2 + 8 * x1 + 7 * x2 + 4 * x3
        return torch.stack((a, b), -1)

    def _constraints_upper(self, y, x):
        _, h = self._empty_constraints(y)
        y1, x1 = y[..., 0], x[..., 0]
        if self.prob_id == 1:
            g = -(1 + x1 + x[..., 1]).unsqueeze(-1)
        elif self.prob_id == 3:
            g = ((x1 - 0.5).square() + (x[..., 1] - 5).square() + (y1 - 5).square() - 16).unsqueeze(-1)
        elif self.prob_id == 4:
            y2, x2, x3 = y[..., 1], x[..., 1], x[..., 2]
            g = torch.stack(
                (3 * y1 + 9 * y2 + 9 * x1 + 5 * x2 + 3 * x3 - 1039, -4 * y1 - y2 + 3 * x1 - 3 * x2 + 2 * x3 - 94), -1
            )
        else:
            return self._empty_constraints(y)
        return g, h

    def _constraints_lower(self, y, x):
        _, h = self._empty_constraints(y)
        y1, x1 = y[..., 0], x[..., 0]
        if self.prob_id == 1:
            g = (x.square().sum(-1) - y1.square()).unsqueeze(-1)
        elif self.prob_id == 3:
            x2 = x[..., 1]
            g = torch.stack((x1.square() - x2, 5 * x1.square() + x2 - 10, x2 + y1 / 6 - 5, -x1), -1)
        elif self.prob_id == 4:
            y2, x2, x3 = y[..., 1], x[..., 1], x[..., 2]
            g = torch.stack(
                (
                    3 * y1 - 9 * y2 - 9 * x1 - 4 * x2 - 61,
                    5 * y1 + 9 * y2 + 10 * x1 - x2 - 2 * x3 - 924,
                    3 * y1 - 3 * y2 + x2 + 5 * x3 - 420,
                ),
                -1,
            )
        else:
            return self._empty_constraints(y)
        return g, h

    def lower_ps(self, xu):
        if xu.ndim < 1 or xu.shape[-1] != self.d_u:
            raise ValueError(f"Expected xu[..., {self.d_u}].")
        t = torch.linspace(0, 1, self.ref_num, device=xu.device, dtype=xu.dtype)
        y = xu[..., 0, None]
        if self.prob_id == 1:
            return torch.stack((-y * torch.cos(t * torch.pi / 2), -y * torch.sin(t * torch.pi / 2)), -1)
        if self.prob_id == 2:
            x1 = y * t
            zeros = x1.new_zeros((*x1.shape, self.d_l - 1))
            return torch.cat((x1.unsqueeze(-1), zeros), -1)
        return super().lower_ps(xu)

    def pf(self):
        if self.prob_id == 1:
            x2 = torch.linspace(-1, 0, self.ref_num, device=self.device, dtype=self.dtype)
            x1 = -1 - x2
            y = torch.sqrt(x1.square() + x2.square())
            return torch.stack((x1 - y, x2), -1)
        if self.prob_id == 2:
            y = torch.linspace(0.5, 1, self.ref_num, device=self.device, dtype=self.dtype)
            return torch.stack(((y - 1).square() + y.square(), 2 * (y - 1).square()), -1)
        return super().pf()


class TP1(TP):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)


class TP2(TP):
    def __init__(self, d_l: int = 14, **kwargs):
        super().__init__(2, d_l, **kwargs)


class TP3(TP):
    def __init__(self, **kwargs):
        super().__init__(3, **kwargs)


class TP4(TP):
    def __init__(self, **kwargs):
        super().__init__(4, **kwargs)
