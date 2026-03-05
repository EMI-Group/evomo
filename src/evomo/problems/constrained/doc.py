import torch
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp

from evomo.problems.constrained.base import CMOP


class DOC1(CMOP):
    """
    Z. Liu and Y. Wang, Handling constrained multiobjective optimization
    problems with constraints in both the decision and objective spaces. IEEE
    Transactions on Evolutionary Computation, 2019, 23(5): 870-884.
    """

    def __init__(self, ref_num: int = 1000, **kwargs):
        self.m = 2
        self.d = 6
        self.lower = torch.tensor([0, 78, 33, 27, 27, 27])
        self.upper = torch.tensor([1, 102, 45, 45, 45, 45])
        self.ref_num = ref_num

        kwargs.pop("m", None)
        kwargs.pop("d", None)
        super().__init__(d=self.d, m=self.m, n_iq=7, n_eq=0, lb=self.lower, ub=self.upper, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        X = clamp(X, self.lb, self.ub)
        g = 5.3578547 * X[:, 3] ** 2 + 0.8356891 * X[:, 1] * X[:, 5] + 37.293239 * X[:, 1] - 40792.141 + 30665.5386717834 + 1

        f1 = X[:, 0]
        f2 = g * (1 - torch.sqrt(f1) / g)

        # Constraints in objective space
        c1 = torch.relu(-(f1**2 + f2**2 - 1))

        # Constraints in decision space
        c2 = 85.334407 + 0.0056858 * X[:, 2] * X[:, 5] + 0.0006262 * X[:, 1] * X[:, 4] - 0.0022053 * X[:, 3] * X[:, 5] - 92
        c3 = -85.334407 - 0.0056858 * X[:, 2] * X[:, 5] - 0.0006262 * X[:, 1] * X[:, 4] + 0.0022053 * X[:, 3] * X[:, 5]
        c4 = 80.51249 + 0.0071317 * X[:, 2] * X[:, 5] + 0.0029955 * X[:, 1] * X[:, 2] + 0.0021813 * X[:, 3] ** 2 - 110
        c5 = -80.51249 - 0.0071317 * X[:, 2] * X[:, 5] - 0.0029955 * X[:, 1] * X[:, 2] - 0.0021813 * X[:, 3] ** 2 + 90
        c6 = 9.300961 + 0.0047026 * X[:, 3] * X[:, 5] + 0.0012547 * X[:, 1] * X[:, 3] + 0.0019085 * X[:, 3] * X[:, 4] - 25
        c7 = -9.300961 - 0.0047026 * X[:, 3] * X[:, 5] - 0.0012547 * X[:, 1] * X[:, 3] - 0.0019085 * X[:, 3] * X[:, 4] + 20

        return torch.column_stack([f1, f2, c1, c2, c3, c4, c5, c6, c7])

    def pf(self):
        pf, _ = uniform_sampling(self.ref_num * self.m, self.m)
        pf = pf.to(self.device)
        pf = pf / (torch.sqrt(torch.sum(pf**2, dim=1, keepdim=True)) + 1e-12)
        # return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.m}_D{self.d}.pf"))
        return pf


class DOC2(CMOP):
    def __init__(self, ref_num: int = 1000, **kwargs):
        self.m = 2
        self.d = 16
        self.lower = torch.tensor([0] + [0] * 15)
        self.upper = torch.tensor([1] + [10] * 15)
        self.ref_num = ref_num

        kwargs.pop("m", None)
        kwargs.pop("d", None)
        super().__init__(d=self.d, m=self.m, n_iq=7, n_eq=0, lb=self.lower, ub=self.upper, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        X = clamp(X, self.lb, self.ub)

        popsize, _ = X.shape

        device, dtype = X.device, X.dtype

        b = torch.tensor([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1], dtype=dtype, device=device)
        c1 = torch.tensor(
            [
                [30, -20, -10, 32, -10],
                [-20, 39, -6, -31, 32],
                [-10, -6, 10, -6, -10],
                [32, -31, -6, 39, -20],
                [-10, 32, -10, -20, 30],
            ],
            dtype=dtype,
            device=device,
        )
        d = torch.tensor([4, 8, 10, 6, 2], dtype=dtype, device=device)
        g_temp = (
            torch.sum((X[:, 11:16] @ c1[:5, :]) * X[:, 11:16], dim=1)
            + 2 * torch.sum(d * X[:, 11:16] ** 3, dim=1)
            - torch.sum(b * X[:, 1:11], dim=1)
        )

        g = (g_temp - 32.6555929502) + 1

        f1 = X[:, 0]
        f2 = g * (1 - (f1) ** (1 / 3) / g)

        # Constraints in objective space
        g1 = torch.relu(-(torch.sqrt(f1) + f2 - 1))
        d1 = torch.zeros((popsize, 3), dtype=dtype, device=device)

        d1[:, 0] = torch.relu(
            (f1 - 1 / 8) ** 2 + (f2 - 1 + torch.sqrt(torch.tensor(1 / 8, device=device, dtype=dtype))) ** 2 - 0.15**2
        )
        d1[:, 1] = torch.relu(
            (f1 - 1 / 2) ** 2 + (f2 - 1 + torch.sqrt(torch.tensor(1 / 2, device=device, dtype=dtype))) ** 2 - 0.15**2
        )
        d1[:, 2] = torch.relu(
            (f1 - 7 / 8) ** 2 + (f2 - 1 + torch.sqrt(torch.tensor(7 / 8, device=device, dtype=dtype))) ** 2 - 0.15**2
        )

        g2 = torch.min(d1, dim=1).values

        a = torch.tensor(
            [
                [-16, 2, 0, 1, 0],
                [0, -2, 0, 0.4, 2],
                [-3.5, 0, 2, 0, 0],
                [0, -2, 0, -4, -1],
                [0, -9, -2, 1, -2.8],
                [2, 0, -4, 0, 0],
                [-1, -1, -1, -1, -1],
                [-1, -2, -3, -2, -1],
                [1, 2, 3, 4, 5],
                [1, 1, 1, 1, 1],
            ],
            dtype=dtype,
            device=device,
        )

        c1 = torch.tensor(
            [
                [30, -20, -10, 32, -10],
                [-20, 39, -6, -31, 32],
                [-10, -6, 10, -6, -10],
                [32, -31, -6, 39, -20],
                [-10, 32, -10, -20, 30],
            ],
            dtype=dtype,
            device=device,
        )
        d = torch.tensor([4, 8, 10, 6, 2], dtype=dtype, device=device)
        e = torch.tensor([-15, -27, -36, -18, -12], dtype=dtype, device=device)

        # Constraints in decision space
        G = -2 * (X[:, 11:16] @ c1[:5, :]) - 3 * (X[:, 11:16] ** 2) * d - e + (X[:, 1:11] @ a[:10, :])

        return torch.column_stack([f1, f2, g1, g2, G])

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)  # (N,1)
        pf = torch.cat([t, 1.0 - torch.sqrt(t)], dim=1)
        mask = (pf[:, 0] < 0.05) | ((pf[:, 0] > 0.2202) & (pf[:, 0] < 0.3830)) | ((pf[:, 0] > 0.6247) & (pf[:, 0] < 0.7440))
        pf = pf[~mask]
        return pf

        # return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.m}_D{self.d}.pf"))


class DOC3(CMOP):
    def __init__(self, ref_num: int = 1000, **kwargs):
        self.m = 2
        self.d = 10
        self.lower = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01])
        self.upper = torch.tensor([1, 1, 300, 100, 200, 100, 1, 100, 200, 0.03])
        self.ref_num = ref_num

        kwargs.pop("m", None)
        kwargs.pop("d", None)
        super().__init__(d=self.d, m=self.m, n_iq=10, n_eq=0, lb=self.lower, ub=self.upper, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        X = clamp(X, self.lb, self.ub)

        g_temp = -9 * X[:, 5] - 15 * X[:, 8] + 6 * X[:, 1] + 16 * X[:, 2] + 10 * (X[:, 6] + X[:, 7])
        g = (g_temp + 400.0551) + 1

        f1 = X[:, 0]
        f2 = g * (1 - f1 / g)

        # Constraints in objective space
        c1 = torch.relu(-(f1**2 + f2**2 - 1))
        sqrt2 = 2**0.5
        c2 = torch.relu(-(torch.abs((-f1 + f2 - 0.5) / sqrt2) - 0.1 / sqrt2))
        c3 = torch.relu(-(torch.abs((-f1 + f2 - 0.0) / sqrt2) - 0.1 / sqrt2))
        c4 = torch.relu(-(torch.abs((-f1 + f2 + 0.5) / sqrt2) - 0.1 / sqrt2))

        # Constraints in decision space
        c5 = X[:, 9] * X[:, 3] + 0.02 * X[:, 6] - 0.025 * X[:, 5]
        c6 = X[:, 9] * X[:, 4] + 0.02 * X[:, 7] - 0.015 * X[:, 8]
        c7 = torch.abs(X[:, 1] + X[:, 2] - X[:, 3] - X[:, 4]) - 0.0001
        c8 = torch.abs(0.03 * X[:, 1] + 0.01 * X[:, 2] - X[:, 9] * (X[:, 3] + X[:, 4])) - 0.0001
        c9 = torch.abs(X[:, 3] + X[:, 6] - X[:, 5]) - 0.0001
        c10 = torch.abs(X[:, 4] + X[:, 7] - X[:, 8]) - 0.0001

        return torch.column_stack([f1, f2, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])

    def pf(self):
        pf, _ = uniform_sampling(self.ref_num * self.m, self.m)
        pf = pf / (torch.sqrt(torch.sum(pf**2, dim=1, keepdim=True)) + 1e-12)
        mask = (
            ((pf[:, 0] > 0.3403) & (pf[:, 0] < 0.4782))
            | ((pf[:, 0] > 0.6553) & (pf[:, 0] < 0.7553))
            | ((pf[:, 0] > 0.8782) & (pf[:, 0] < 0.9403))
        )
        pf = pf[~mask]
        # return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.m}_D{self.d}.pf"))
        return pf


class DOC4(CMOP):
    def __init__(self, ref_num: int = 1000, **kwargs):
        self.m = 2
        self.d = 8
        self.lower = torch.tensor([0, -10, -10, -10, -10, -10, -10, -10])
        self.upper = torch.tensor([1, 10, 10, 10, 10, 10, 10, 10])
        self.ref_num = ref_num

        kwargs.pop("m", None)
        kwargs.pop("d", None)
        super().__init__(d=self.d, m=self.m, n_iq=6, n_eq=0, lb=self.lower, ub=self.upper, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        X = clamp(X, self.lb, self.ub)

        g_temp = (
            (X[:, 1] - 10) ** 2
            + 5 * (X[:, 2] - 12) ** 2
            + X[:, 3] ** 4
            + 3 * (X[:, 4] - 11) ** 2
            + 10 * X[:, 5] ** 6
            + 7 * X[:, 6] ** 2
            + X[:, 7] ** 4
            - 4 * X[:, 6] * X[:, 7]
            - 10 * X[:, 6]
            - 8 * X[:, 7]
        )
        g = g_temp - 680.6300573745 + 1

        f1 = X[:, 0]
        f2 = g * (1 - torch.sqrt(f1) / g)

        # Constraints in objective space
        c1 = torch.relu(-(f1 + f2 - 1))
        c2 = torch.relu(-(f1 + f2 - 1 - torch.abs(torch.sin(10 * torch.pi * (f1 - f2 + 1)))))

        # Constraints in decision space
        c3 = -127 + 2 * X[:, 1] ** 2 + 3 * X[:, 2] ** 4 + X[:, 3] + 4 * X[:, 4] ** 2 + 5 * X[:, 5]
        c4 = -282 + 7 * X[:, 1] + 3 * X[:, 2] + 10 * X[:, 3] ** 2 + X[:, 4] - X[:, 5]
        c5 = -196 + 23 * X[:, 1] + X[:, 2] ** 2 + 6 * X[:, 6] ** 2 - 8 * X[:, 7]
        c6 = 4 * X[:, 1] ** 2 + X[:, 2] ** 2 - 3 * X[:, 1] * X[:, 2] + 2 * X[:, 3] ** 2 + 5 * X[:, 6] - 11 * X[:, 7]

        return torch.column_stack([f1, f2, c1, c2, c3, c4, c5, c6])

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=21, device=self.device)
        pf = torch.column_stack([t, 1.0 - t])
        # return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.m}_D{self.d}.pf"))
        return pf


class DOC5(CMOP):
    def __init__(self, ref_num: int = 1000, **kwargs):
        self.m = 2
        self.d = 8
        self.lower = torch.tensor([0, 0, 0, 0, 100, 6.3, 5.9, 4.5])
        self.upper = torch.tensor([1, 1000, 40, 40, 300, 6.7, 6.4, 6.25])
        self.ref_num = ref_num

        kwargs.pop("m", None)
        kwargs.pop("d", None)
        super().__init__(d=self.d, m=self.m, n_iq=9, n_eq=0, lb=self.lower, ub=self.upper, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        X = clamp(X, self.lb, self.ub)
        g_temp = X[:, 1]
        g = g_temp - 193.724510070035 + 1

        f1 = X[:, 0]
        f2 = g * (1 - torch.sqrt(f1) / g)

        # Constraints in objective space
        c1 = torch.relu(-(f1 + f2 - 1))
        c2 = torch.relu(-(f1 + f2 - 1 - torch.abs(torch.sin(10 * torch.pi * (f1 - f2 + 1)))))

        c3 = torch.relu((f1 - 0.8) * (f2 - 0.6))

        # Constraints in decision space
        c4 = -X[:, 1] + 35 * X[:, 2] ** 0.6 + 35 * X[:, 3] ** 0.6
        c5 = (
            torch.abs(
                -300 * X[:, 3]
                + 7500 * X[:, 5]
                - 7500 * X[:, 6]
                - 25 * X[:, 4] * X[:, 5]
                + 25 * X[:, 4] * X[:, 6]
                + X[:, 3] * X[:, 4]
            )
            - 0.0001
        )
        c6 = (
            torch.abs(100 * X[:, 2] + 155.365 * X[:, 4] + 2500 * X[:, 7] - X[:, 2] * X[:, 4] - 25 * X[:, 4] * X[:, 7] - 15536.5)
            - 0.0001
        )
        c7 = torch.abs(-X[:, 5] + torch.log(-X[:, 4] + 900)) - 0.0001
        c8 = torch.abs(-X[:, 6] + torch.log(X[:, 4] + 300)) - 0.0001
        c9 = torch.abs(-X[:, 7] + torch.log(-2 * X[:, 4] + 700)) - 0.0001

        return torch.column_stack([f1, f2, c1, c2, c3, c4, c5, c6, c7, c8, c9])

    def pf(self):
        t = torch.cat([torch.arange(0, 9, device=self.device), torch.arange(16, 21, device=self.device)], dim=0) / 20
        # return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.m}_D{self.d}.pf"))
        return torch.column_stack([t, 1.0 - t])


class DOC6(CMOP):
    def __init__(self, ref_num: int = 1000, **kwargs):
        self.m = 2
        self.d = 11
        self.lower = torch.tensor([0, -10] + [-10] * 9)
        self.upper = torch.tensor([1, 10] + [10] * 9)
        self.ref_num = ref_num

        kwargs.pop("m", None)
        kwargs.pop("d", None)
        super().__init__(d=self.d, m=self.m, n_iq=10, n_eq=0, lb=self.lower, ub=self.upper, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        X = clamp(X, self.lb, self.ub)

        g_temp = (
            X[:, 1] ** 2
            + X[:, 2] ** 2
            + X[:, 1] * X[:, 2]
            - 14 * X[:, 1]
            - 16 * X[:, 2]
            + (X[:, 3] - 10) ** 2
            + 4 * (X[:, 4] - 5) ** 2
            + (X[:, 5] - 3) ** 2
            + 2 * (X[:, 6] - 1) ** 2
            + 5 * X[:, 7] ** 2
            + 7 * (X[:, 8] - 11) ** 2
            + 2 * (X[:, 9] - 10) ** 2
            + (X[:, 10] - 7) ** 2
            + 45
        )

        g = g_temp - 24.3062090681 + 1

        f1 = X[:, 0]
        f2 = g * (1 - torch.sqrt(f1) / g)

        # Constraints in objective space
        c1 = torch.relu(-(f1 + f2 - 1))
        c2 = torch.relu(-(f1 - 0.5) * (f1 + f2 - 1 - torch.abs(torch.sin(10 * torch.pi * (f1 - f2 + 1)))))

        # Constraints in decision space
        c3 = -105 + 4 * X[:, 1] + 5 * X[:, 2] - 3 * X[:, 7] + 9 * X[:, 8]
        c4 = 10 * X[:, 1] - 8 * X[:, 2] - 17 * X[:, 7] + 2 * X[:, 8]
        c5 = -8 * X[:, 1] + 2 * X[:, 2] + 5 * X[:, 9] - 2 * X[:, 10] - 12
        c6 = 3 * (X[:, 1] - 2) ** 2 + 4 * (X[:, 2] - 3) ** 2 + 2 * X[:, 3] ** 2 - 7 * X[:, 4] - 120
        c7 = 5 * X[:, 1] ** 2 + 8 * X[:, 2] + (X[:, 3] - 6) ** 2 - 2 * X[:, 4] - 40
        c8 = X[:, 1] ** 2 + 2 * (X[:, 2] - 2) ** 2 - 2 * X[:, 1] * X[:, 2] + 14 * X[:, 5] - 6 * X[:, 6]
        c9 = 0.5 * (X[:, 1] - 8) ** 2 + 2 * (X[:, 2] - 4) ** 2 + 3 * X[:, 5] ** 2 - X[:, 6] - 30
        c10 = -3 * X[:, 1] + 6 * X[:, 2] + 12 * (X[:, 9] - 8) ** 2 - 7 * X[:, 10]

        return torch.stack([f1, f2, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10], dim=1)

    def pf(self):
        t = torch.cat([torch.linspace(0, 0.5, self.ref_num, device=self.device), torch.arange(11, 21, device=self.device) / 20])
        # return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.m}_D{self.d}.pf"))
        return torch.column_stack([t, 1 - t])


class DOC7(CMOP):
    def __init__(self, ref_num: int = 1000, **kwargs):
        self.m = 2
        self.d = 11
        self.lower = torch.tensor([0] + [0] * 10)
        self.upper = torch.tensor([1] + [10] * 10)
        self.ref_num = ref_num

        kwargs.pop("m", None)
        kwargs.pop("d", None)
        super().__init__(d=self.d, m=self.m, n_iq=6, n_eq=0, lb=self.lower, ub=self.upper, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        X = clamp(X, self.lb, self.ub)
        c1 = torch.tensor(
            [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179],
            device=X.device,
            dtype=X.dtype,
        )
        X_temp = X[:, 1:11]
        sum_X = torch.sum(X_temp, dim=1, keepdims=True)
        g_temp = torch.sum(X_temp * (c1 + torch.log(1e-30 + X_temp / (1e-30 + sum_X))), dim=1)
        g = g_temp + 47.7648884595 + 1

        f1 = X[:, 0]
        f2 = g * (1 - torch.sqrt(f1) / g)

        # Constraints in objective space
        c1 = torch.relu(-(f1 + f2 - 1))
        c2 = torch.relu(-((f1 - 0.5) * (f1 + f2 - 1 - torch.abs(torch.sin(10 * torch.pi * (f1 - f2 + 1))))))
        c3 = torch.relu(-(torch.abs(-f1 + f2) / 2**0.5 - 0.1 / 2**0.5))

        # Constraints in decision space
        c4 = torch.abs(X[:, 2] + 2 * X[:, 3] + 2 * X[:, 4] + X[:, 7] + X[:, 10] - 2) - 0.0001
        c5 = torch.abs(X[:, 5] + 2 * X[:, 6] + X[:, 7] + X[:, 8] - 1) - 0.0001
        c6 = torch.abs(X[:, 4] + X[:, 8] + X[:, 9] + 2 * X[:, 10] + X[:, 10] - 1) - 0.0001

        return torch.column_stack([f1, f2, c1, c2, c3, c4, c5, c6])

    def pf(self):
        t = torch.cat(
            [torch.linspace(0, 0.45, self.ref_num, device=self.device), torch.arange(11, 21, device=self.device) / 20]
        )
        # return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.m}_D{self.d}.pf"))
        return torch.column_stack([t, 1 - t])


class DOC8(CMOP):
    def __init__(self, ref_num: int = 1000, **kwargs):
        self.m = 3
        self.d = 10
        self.lower = torch.tensor([0, 0, 500, 1000, 5000, 100, 100, 100, 100, 100])
        self.upper = torch.tensor([1, 1, 1000, 2000, 6000, 500, 500, 500, 500, 500])
        self.ref_num = ref_num

        kwargs.pop("m", None)
        kwargs.pop("d", None)
        super().__init__(d=self.d, m=self.m, n_iq=7, n_eq=0, lb=self.lower, ub=self.upper, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        X = clamp(X, self.lb, self.ub)
        g_temp = X[:, 2] + X[:, 3] + X[:, 4]
        g = g_temp - 7049.2480205286 + 1

        f1 = (X[:, 0] * X[:, 1]) * g
        f2 = (X[:, 0] * (1 - X[:, 1])) * g
        f3 = (1 - X[:, 0]) * g

        # Constraints in objective space
        c1 = torch.relu(-(f3 - 0.4) * (f3 - 0.6))

        # Constraints in decision space
        c2 = -1 + 0.0025 * (X[:, 5] + X[:, 7])
        c3 = -1 + 0.0025 * (X[:, 6] + X[:, 8] - X[:, 5])
        c4 = -1 + 0.01 * (X[:, 9] - X[:, 6])
        c5 = -X[:, 2] * X[:, 7] + 833.33252 * X[:, 5] + 100 * X[:, 2] - 83333.333
        c6 = -X[:, 3] * X[:, 8] + 1250 * X[:, 6] + X[:, 3] * X[:, 5] - 1250 * X[:, 5]
        c7 = -X[:, 4] * X[:, 9] + 1250000 + X[:, 4] * X[:, 6] - 2500 * X[:, 6]

        return torch.column_stack([f1, f2, f3, c1, c2, c3, c4, c5, c6, c7])

    def pf(self):
        pf, _ = uniform_sampling(self.ref_num * self.m, self.m)
        pf = pf.to(self.device)
        mask = (pf[:, 2] <= 0.4) | (pf[:, 2] >= 0.6)
        return pf[mask]


class DOC9(CMOP):
    def __init__(self, ref_num: int = 1000, **kwargs):
        self.m = 3
        self.d = 11
        self.lower = torch.tensor([0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        self.upper = torch.tensor([1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        self.ref_num = ref_num
        kwargs.pop("m", None)
        kwargs.pop("d", None)
        super().__init__(d=self.d, m=self.m, n_iq=14, n_eq=0, lb=self.lower, ub=self.upper, **kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        X = clamp(X, self.lb, self.ub)
        g_temp = -0.5 * (
            X[:, 2] * X[:, 5]
            - X[:, 3] * X[:, 4]
            + X[:, 4] * X[:, 10]
            - X[:, 6] * X[:, 10]
            + X[:, 6] * X[:, 9]
            - X[:, 7] * X[:, 8]
        )
        g = g_temp + 0.8660254038 + 1

        f1 = torch.cos(0.5 * torch.pi * X[:, 0]) * torch.cos(0.5 * torch.pi * X[:, 1]) * g
        f2 = torch.cos(0.5 * torch.pi * X[:, 0]) * torch.sin(0.5 * torch.pi * X[:, 1]) * g
        f3 = torch.sin(0.5 * torch.pi * X[:, 0]) * g

        # Constraints in objective space
        c1 = torch.relu(-(f1**2 + f2**2 - 1))

        # Constraints in decision space
        c2 = X[:, 4] ** 2 + X[:, 5] ** 2 - 1
        c3 = X[:, 10] ** 2 - 1
        c4 = X[:, 6] ** 2 + X[:, 7] ** 2 - 1
        c5 = X[:, 2] ** 2 + (X[:, 3] - X[:, 10]) ** 2 - 1
        c6 = (X[:, 2] - X[:, 6]) ** 2 + (X[:, 3] - X[:, 7]) ** 2 - 1
        c7 = (X[:, 2] - X[:, 8]) ** 2 + (X[:, 3] - X[:, 9]) ** 2 - 1
        c8 = (X[:, 4] - X[:, 6]) ** 2 + (X[:, 5] - X[:, 7]) ** 2 - 1
        c9 = (X[:, 4] - X[:, 8]) ** 2 + (X[:, 5] - X[:, 9]) ** 2 - 1
        c10 = X[:, 8] ** 2 + (X[:, 9] - X[:, 10]) ** 2 - 1
        c11 = X[:, 3] * X[:, 4] - X[:, 2] * X[:, 5]
        c12 = -X[:, 5] * X[:, 10]
        c13 = X[:, 7] * X[:, 10]
        c14 = X[:, 8] * X[:, 9] - X[:, 7] * X[:, 10]

        return torch.column_stack([f1, f2, f3, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14])

    def pf(self):
        pf, _ = uniform_sampling(self.ref_num * self.m, self.m - 1)
        pf = pf.to(self.device)
        pf = pf / (torch.sqrt(torch.sum(pf**2, dim=1, keepdim=True)) + 1e-12)
        zeros_col = torch.zeros(pf.size(0), 1, device=pf.device, dtype=pf.dtype)
        pf = torch.column_stack([pf, zeros_col])
        # return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.m}_D{self.d}.pf"))
        return pf
