import torch
from evox.operators.sampling import grid_sampling, uniform_sampling

from evomo.problems.constrained.base import CMOP
from evomo.utils import get_pareto_front


class MW(CMOP):
    _n_iqs = {
        **dict.fromkeys([1, 2, 4, 6, 8, 9, 14], 1),
        **dict.fromkeys([3, 7, 12, 13], 2),
        **dict.fromkeys([5, 10], 3),
        **dict.fromkeys([11], 4),
    }

    _ubs = {
        **dict.fromkeys([1, 2, 3, 4, 5, 7, 8, 9, 10, 12], 1),
        **dict.fromkeys([6], 1.1),
        **dict.fromkeys([13, 14], 1.5),
        **dict.fromkeys([11], 2**0.5),
    }

    def __init__(self, prob_id: int = None, d: int = None, m: int = None, ref_num: int = 1000, **kwargs):

        if prob_id not in range(1, 15):
            raise ValueError("Please select a valid prob id.")

        # Set dim
        n_iq = self._n_iqs[prob_id]

        # Set lb, ub
        self.device = kwargs.get("device", torch.get_default_device())
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        lb = torch.zeros(d, device=self.device, dtype=dtype)
        ub = torch.full((d,), self._ubs[prob_id], device=self.device, dtype=dtype)
        self.ref_num = ref_num
        self.m = m

        super(MW, self).__init__(d=d, m=m, n_iq=n_iq, n_eq=0, lb=lb, ub=ub, **kwargs)
        self.register_buffer("sqrt2", torch.sqrt(torch.tensor(2.0, dtype=dtype)))

    @staticmethod
    def LA1(A, B, C, D, theta):
        return A * torch.pow(torch.sin(B * torch.pi * torch.pow(theta, C)), D)

    @staticmethod
    def LA2(A, B, C, D, theta):
        return A * torch.pow(torch.sin(B * torch.pow(theta, C)), D)

    @staticmethod
    def LA3(A, B, C, D, theta):
        return A * torch.pow(torch.cos(B * torch.pow(theta, C)), D)

    def g1(self, X):
        d = self.d
        n = d - self.m
        z = X[:, self.m - 1 :].pow(n)
        i = torch.arange(self.m - 1, d, device=X.device, dtype=X.dtype)
        exp = 1 - torch.exp(-10.0 * (z - 0.5 - i / (2 * d)) * (z - 0.5 - i / (2 * d)))
        distance = 1 + exp.sum(dim=1, keepdim=True)
        return distance

    def g2(self, X):
        d = self.d
        n = d
        i = torch.arange(self.m - 1, d, device=X.device, dtype=X.dtype)
        z = 1 - torch.exp(-10.0 * (X[:, self.m - 1 :] - i / n) * (X[:, self.m - 1 :] - i / n))
        contrib = (0.1 / n) * z * z + 1.5 - 1.5 * torch.cos(2 * torch.pi * z)
        distance = 1 + contrib.sum(dim=1, keepdims=True)
        return distance

    def g3(self, X):
        contrib = 2.0 * (X[:, self.m - 1 :] + (X[:, self.m - 2 : -1] - 0.5) * (X[:, self.m - 2 : -1] - 0.5) - 1.0) ** 2
        distance = 1 + contrib.sum(dim=1, keepdims=True)
        return distance


class MW1(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")
        if not d > 2:
            raise ValueError("Number of variables must be greater than two.")

        super(MW1, self).__init__(prob_id=1, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        f0 = X[:, [0]]
        f1 = g - 0.85 * f0
        F = torch.column_stack([f0, f1])
        C = f0 + f1 - 1 - self.LA1(0.5, 2.0, 1.0, 8.0, self.sqrt2 * (f1 - f0))
        return torch.column_stack([F, C])

    def pf(self):
        r1 = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device)
        r2 = 1.0 - 0.85 * r1
        l = self.sqrt2 * (r2 - r1)  # noqa: E741
        c = 1.0 - r1 - r2 + 0.5 * torch.sin(2.0 * torch.pi * l).pow(8)  # (N,1)
        R = torch.column_stack([r1, r2])
        return R[c >= 0]


class MW2(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")
        if not d > 2:
            raise ValueError("Number of variables must be greater than two.")

        super(MW2, self).__init__(prob_id=2, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)
        f0 = X[:, [0]]
        f1 = g - f0
        F = torch.column_stack([f0, f1])
        C = f0 + f1 - 1 - self.LA1(0.5, 3.0, 1.0, 8.0, self.sqrt2 * (f1 - f0))
        return torch.column_stack([F, C])

    def pf(self):
        t = torch.linspace(0, 1, self.ref_num, device=self.device).unsqueeze(1)
        return torch.cat((t, 1 - t), dim=1)


class MW3(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")

        super(MW3, self).__init__(prob_id=3, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g3(X)
        f0 = X[:, [0]]
        f1 = g - f0
        g0 = f0 + f1 - 1.05 - self.LA1(0.45, 0.75, 1.0, 6.0, self.sqrt2 * (f1 - f0))
        g1 = 0.85 - f0 - f1 + self.LA1(0.3, 0.75, 1.0, 2.0, self.sqrt2 * (f1 - f0))
        F = torch.column_stack([f0, f1])
        C = torch.column_stack([g0, g1])
        return torch.column_stack([F, C])

    def pf(self):
        t = torch.linspace(0, 1, self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t], dim=1)
        while True:
            C = 0.85 - pf[:, 0] - pf[:, 1] + 0.3 * torch.sin(0.75 * torch.pi * self.sqrt2 * (pf[:, 1] - pf[:, 0])) ** 2
            invalid = C > 0
            if not invalid.any():
                break
            pf[invalid] *= 1.001
        return pf


class MW4(MW):
    def __init__(self, d=10, m=3, ref_num: int = 1000, **kwargs):
        if not d > 2:
            raise ValueError("Number of variables must be greater than two.")

        super(MW4, self).__init__(prob_id=4, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        F = g * torch.ones((X.size(0), self.m), device=X.device, dtype=X.dtype)
        F[:, 1:] *= X[:, torch.arange(self.m - 2, -1, -1, device=X.device)]
        F[:, :-1] *= torch.flip(torch.cumprod(1 - X[:, : self.m - 1], dim=1), dims=[1])
        C = F.sum(dim=1, keepdim=True) - 1.0 - self.LA1(0.4, 2.5, 1.0, 8.0, F[:, [-1]] - F[:, :-1].sum(dim=1, keepdim=True))
        return torch.cat([F, C], dim=1)

    def pf(self):
        pf, _ = uniform_sampling(self.ref_num * self.m, self.m)
        pf = pf.to(self.device)
        l = pf[:, [-1]] - pf[:, :-1].sum(dim=1, keepdim=True)  # noqa: E741
        c = (1.0 + 0.4 * torch.sin(2.5 * torch.pi * l).pow(8)) - pf.sum(dim=1, keepdim=True)
        return pf[c.squeeze() >= 0]


class MW5(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")
        if not d > 2:
            raise ValueError("Number of variables must be greater than two.")

        super(MW5, self).__init__(prob_id=5, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        f0 = g * X[:, [0]]
        f1 = g * torch.sqrt(1.0 - X[:, [0]] ** 2 + 1e-6)

        atan = torch.atan2(f1, f0)

        g0 = f0**2 + f1**2 - (1.7 - self.LA2(0.2, 2.0, 1.0, 1.0, atan)) ** 2.0
        t = 0.5 * torch.pi - 2 * torch.abs(atan - 0.25 * torch.pi)
        g1 = (1 + self.LA2(0.5, 6.0, 3.0, 1.0, t)) ** 2 - f0**2 - f1**2
        g2 = (1 - self.LA2(0.45, 6.0, 3.0, 1.0, t)) ** 2 - f0**2 - f1**2
        F = torch.column_stack([f0, f1])
        C = torch.column_stack([g0, g1, g2])
        return torch.column_stack([F, C])

    def pf(self):
        pf = torch.tensor(
            [
                [0.0000, 1.0000],
                [0.3922, 0.9199],
                [0.4862, 0.8739],
                [0.5490, 0.8358],
                [0.5970, 0.8023],
                [0.6359, 0.7719],
                [0.6686, 0.7436],
                [0.6969, 0.7174],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        pf = torch.cat([pf, torch.flip(pf, dims=[1])], dim=0)

        return pf


class MW6(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")

        super(MW6, self).__init__(prob_id=6, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)  # (N,1)
        x0 = X[:, [0]]  # (N,1)
        f0 = g * x0
        f1 = g * torch.sqrt(1.1**2 - x0**2 + 1e-6)
        theta = torch.atan2(f1, f0)
        C = (
            (f0**2) / (1.0 + self.LA3(0.15, 6.0, 4.0, 10.0, theta)) ** 2
            + (f1**2) / (1.0 + self.LA3(0.75, 6.0, 4.0, 10.0, theta)) ** 2
            - 1.0
        )
        return torch.cat([f0, f1, C], dim=1)

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t], dim=1)
        pf = pf / torch.sqrt((pf**2).sum(dim=1, keepdim=True) / 1.21)
        l = torch.cos(6.0 * torch.atan(pf[:, [1]] / pf[:, [0]]) ** 4.0) ** 10.0  # noqa: E741
        c = 1.0 - (pf[:, [0]] / (1.0 + 0.15 * l)) ** 2 - (pf[:, [1]] / (1.0 + 0.75 * l)) ** 2
        return pf[c.squeeze() >= 0]


class MW7(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")

        super(MW7, self).__init__(prob_id=7, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g3(X)
        f0 = g * X[:, [0]]
        f1 = g * torch.sqrt(1 - X[:, [0]] ** 2 + 1e-6)

        atan = torch.atan2(f1, f0)
        g0 = f0**2 + f1**2 - (1.2 + torch.abs(self.LA2(0.4, 4.0, 1.0, 16.0, atan))) ** 2
        g1 = (1.15 - self.LA2(0.2, 4.0, 1.0, 8.0, atan)) ** 2 - f0**2 - f1**2
        return torch.column_stack([f0, f1, g0, g1])

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t], dim=1)
        pf = pf / torch.sqrt((pf**2).sum(dim=1, keepdim=True))
        while True:
            ang = torch.atan2(pf[:, 1], pf[:, 0])
            c = (1.15 - 0.2 * torch.sin(4.0 * ang) ** 8) ** 2 - (pf**2).sum(dim=1)
            invalid = c > 0
            if not invalid.any():
                break
            pf[invalid] *= 1.001

        pf = get_pareto_front(pf)
        return pf


class MW8(MW):
    def __init__(self, d=10, m=3, ref_num: int = 1000, **kwargs):
        super(MW8, self).__init__(prob_id=8, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)
        f = g.repeat(1, self.m)
        f[:, 1:] *= torch.sin(0.5 * torch.pi * X[:, : (self.m - 1)].flip(1))
        cos = torch.cos(0.5 * torch.pi * X[:, : (self.m - 1)])
        f[:, :-1] *= torch.flip(torch.cumprod(cos, dim=1), dims=[1])

        f_squared = (f**2).sum(dim=1, keepdim=True)
        g0 = f_squared - (1.25 - self.LA2(0.5, 6.0, 1.0, 2.0, torch.asin(f[:, [-1]] / torch.sqrt(f_squared)))) ** 2
        return torch.column_stack([f, g0])

    def pf(self):
        pf, _ = uniform_sampling(self.ref_num * self.m, self.m)
        pf = pf.to(self.device)
        pf = pf / torch.sqrt((pf**2).sum(dim=1, keepdim=True))
        mask = 1.0 - (1.25 - 0.5 * torch.sin(6.0 * torch.asin(pf[:, -1])) ** 2) ** 2 <= 0
        return pf[mask]


class MW9(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")
        if not d > 2:
            raise ValueError("Number of variables must be greater than two.")

        super(MW9, self).__init__(prob_id=9, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        f0 = g * X[:, [0]]
        f1 = g * (1.0 - torch.pow(f0 / g, 0.6))
        t1 = (1 - 0.64 * f0 * f0 - f1) * (1 - 0.36 * f0 * f0 - f1)
        t2 = (1.35 * 1.35 - (f0 + 0.35) * (f0 + 0.35) - f1) * (1.15 * 1.15 - (f0 + 0.15) * (f0 + 0.15) - f1)
        C = torch.minimum(t1, t2)
        return torch.column_stack([f0, f1, C])

    def pf(self):
        x = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        R = torch.column_stack([x, 1.0 - x**0.6])  # (N,2)

        while True:
            T1 = (1 - 0.64 * R[:, 0] ** 2 - R[:, 1]) * (1 - 0.36 * R[:, 0] ** 2 - R[:, 1])
            T2 = 1.35**2 - (R[:, 0] + 0.35) ** 2 - R[:, 1]
            T3 = 1.15**2 - (R[:, 0] + 0.15) ** 2 - R[:, 1]
            invalid = torch.minimum(T1, T2 * T3) > 0
            if not invalid.any():
                break
            R[invalid] = R[invalid] * 1.001
        return get_pareto_front(R)


class MW10(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")

        super(MW10, self).__init__(prob_id=10, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)
        f0 = g * torch.pow(X[:, [0]], self.d)
        f1 = g * (1.0 - (f0 / g) ** 2)

        g0 = -1.0 * (2.0 - 4.0 * f0 * f0 - f1) * (2.0 - 8.0 * f0 * f0 - f1)
        g1 = (2.0 - 2.0 * f0 * f0 - f1) * (2.0 - 16.0 * f0 * f0 - f1)
        g2 = (1.0 - f0 * f0 - f1) * (1.2 - 1.2 * f0 * f0 - f1)
        return torch.column_stack([f0, f1, g0, g1, g2])

    def pf(self):
        x = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        R = torch.column_stack([x, 1.0 - x**2])  # (N,2)

        while True:
            x1, x2 = R[:, 0], R[:, 1]
            c1 = (2 - 4 * x1**2 - x2) * (2 - 8 * x1**2 - x2)
            c2 = (2 - 2 * x1**2 - x2) * (2 - 16 * x1**2 - x2)
            c3 = (1 - x1**2 - x2) * (1.2 - 1.2 * x1**2 - x2)
            invalid = (c1 < 0) | (c2 > 0) | (c3 > 0)
            if not invalid.any():
                break
            R[invalid] = R[invalid] * 1.001
            R = R[~(R > 1.3).any(dim=1)]
        return get_pareto_front(R)


class MW11(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")

        super(MW11, self).__init__(prob_id=11, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g3(X)
        f0 = g * X[:, [0]]
        f1 = g * torch.sqrt(2.0 - X[:, [0]] ** 2)

        g0 = -1.0 * (3.0 - f0 * f0 - f1) * (3.0 - 2.0 * f0 * f0 - f1)
        g1 = (3.0 - 0.625 * f0 * f0 - f1) * (3.0 - 7.0 * f0 * f0 - f1)
        g2 = -1.0 * (1.62 - 0.18 * f0 * f0 - f1) * (1.125 - 0.125 * f0 * f0 - f1)
        g3 = (2.07 - 0.23 * f0 * f0 - f1) * (0.63 - 0.07 * f0 * f0 - f1)
        return torch.column_stack([f0, f1, g0, g1, g2, g3])

    def pf(self):
        x = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        R = torch.column_stack([x, 1.0 - x])  # (N,2)
        R = R / torch.sqrt((R**2).sum(dim=1, keepdim=True) / 2)  # Normalized to radius sqrt(2)

        while True:
            x1, x2 = R[:, 0], R[:, 1]
            c1 = (3 - x1**2 - x2) * (3 - 2 * x1**2 - x2)
            c2 = (3 - 0.625 * x1**2 - x2) * (3 - 7 * x1**2 - x2)
            c3 = (1.62 - 0.18 * x1**2 - x2) * (1.125 - 0.125 * x1**2 - x2)
            c4 = (2.07 - 0.23 * x1**2 - x2) * (0.63 - 0.07 * x1**2 - x2)
            invalid = (c1 < 0) | (c2 > 0) | (c3 < 0) | (c4 > 0)
            if not invalid.any():
                break
            R[invalid] = R[invalid] * 1.001
            R = R[~(R > 2.2).any(dim=1)]

        R = torch.cat([R, torch.tensor([[1.0, 1.0]], device=self.device, dtype=R.dtype)], dim=0)
        return get_pareto_front(R)


class MW12(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")
        if not d > 2:
            raise ValueError("Number of variables must be greater than two.")

        super(MW12, self).__init__(prob_id=12, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        f0 = g * X[:, [0]]
        f1 = g * (0.85 - 0.8 * (f0 / g) - 0.08 * torch.abs(torch.sin(3.2 * torch.pi * (f0 / g))))

        g0 = (
            -1.0
            * (1 - 0.625 * f0 - f1 + 0.08 * torch.sin(2 * torch.pi * (f1 - f0 / 1.6)))
            * (1.4 - 0.875 * f0 - f1 + 0.08 * torch.sin(2 * torch.pi * (f1 / 1.4 - f0 / 1.6)))
        )

        g1 = (1 - 0.8 * f0 - f1 + 0.08 * torch.sin(2 * torch.pi * (f1 - f0 / 1.5))) * (
            1.8 - 1.125 * f0 - f1 + 0.08 * torch.sin(2 * torch.pi * (f1 / 1.8 - f0 / 1.6))
        )
        return torch.column_stack([f0, f1, g0, g1])

    def pf(self):
        x = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        R = torch.column_stack([x, 0.85 - 0.8 * x - 0.08 * torch.abs(torch.sin(3.2 * torch.pi * x))])

        while True:
            x1, x2 = R[:, 0], R[:, 1]
            c1 = (1 - 0.8 * x1 - x2 + 0.08 * torch.sin(2 * torch.pi * (x2 - x1 / 1.5))) * (
                1.8 - 1.125 * x1 - x2 + 0.08 * torch.sin(2 * torch.pi * (x2 / 1.8 - x1 / 1.6))
            )
            invalid = c1 > 0
            if not invalid.any():
                break
            R[invalid] = R[invalid] * 1.001

        return R


class MW13(MW):
    def __init__(self, m=2, d=10, ref_num: int = 1000, **kwargs):
        if m != 2:
            raise ValueError("Number of objectives must equal two.")

        super(MW13, self).__init__(prob_id=13, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)
        f0 = g * X[:, [0]]
        f1 = g * (5.0 - torch.exp(f0 / g) - torch.abs(0.5 * torch.sin(3 * torch.pi * (f0 / g))))

        g0 = (
            -1.0
            * (5.0 - (1 + f0 + 0.5 * (f0**2)) - 0.5 * torch.sin(3 * torch.pi * f0) - f1)
            * (5.0 - (1 + 0.7 * f0) - 0.5 * torch.sin(3 * torch.pi * f0) - f1)
        )

        g1 = (5.0 - torch.exp(f0) - 0.5 * torch.sin(3 * torch.pi * f0) - f1) * (
            5.0 - (1 + 0.4 * f0) - 0.5 * torch.sin(3 * torch.pi * f0) - f1
        )
        return torch.column_stack([f0, f1, g0, g1])

    def pf(self):
        x = torch.linspace(0.0, 1.5, steps=self.ref_num, device=self.device).unsqueeze(1)
        R = torch.column_stack([x, 5.0 - torch.exp(x) - 0.5 * torch.abs(torch.sin(3.0 * torch.pi * x))])

        while True:
            x1, x2 = R[:, 0], R[:, 1]
            c1 = (5.0 - torch.exp(x1) - 0.5 * torch.sin(3.0 * torch.pi * x1) - x2) * (
                5.0 - (1.0 + 0.4 * x1) - 0.5 * torch.sin(3.0 * torch.pi * x1) - x2
            )
            invalid = c1 > 0
            if not invalid.any():
                break
            R[invalid] = R[invalid] * 1.001
        return get_pareto_front(R)


class MW14(MW):
    def __init__(self, d=10, m=3, ref_num: int = 1000, **kwargs):
        super(MW14, self).__init__(prob_id=14, d=d, m=m, ref_num=ref_num, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g3(X)
        f = torch.zeros((X.size(0), self.m), device=X.device, dtype=X.dtype)
        f[:, :-1] = X[:, : (self.m - 1)]

        LA1 = self.LA1(1.5, 1.1, 2.0, 1.0, f[:, :-1])
        inter = (6.0 - torch.exp(f[:, :-1]) - LA1).sum(dim=1, keepdim=True)
        f[:, [-1]] = g / (self.m - 1) * inter

        alpha = 6.1 - 1.0 - f[:, :-1] - 0.5 * (f[:, :-1] ** 2) - LA1
        C = f[:, [-1]] - (alpha.sum(dim=1, keepdim=True) / (self.m - 1))
        return torch.column_stack([f, C])

    def pf(self):
        # intervals & median (scalars)
        i0, i1, i2, i3 = 0.0, 0.731000, 1.331000, 1.500000
        median = (i1 - i0) / ((i3 - i2) + (i1 - i0))

        # grid points in [0,1] for M-1 dims
        X, _ = grid_sampling(self.ref_num, self.m - 1)
        X = X.to(self.device)

        # piecewise linear remap by median
        X = torch.where(
            X <= median,
            X * ((i1 - i0) / median) + i0,
            (X - median) * ((i3 - i2) / (1.0 - median)) + i2,
        )

        # last column
        last = (6.0 - torch.exp(X) - 1.5 * torch.sin(1.1 * torch.pi * (X**2))).sum(dim=1, keepdim=True) / (self.m - 1)

        return torch.cat([X, last], dim=1)
