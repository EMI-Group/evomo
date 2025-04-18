from functools import partial
from itertools import cycle

import torch
from evox.core import Problem
from evox.operators.sampling import uniform_sampling
from evox.problems.numerical import ackley_func, griewank_func, rastrigin_func, rosenbrock_func, sphere_func


def schwefel_func(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(x), dim=-1).values


class LSMOP(Problem):
    """
    The LSMOP test suite for large-scale multiobjective and many-objective optimization.

    :references:
        [1] R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, "Test Problems for Large-Scale
            Multiobjective and Many-Objective Optimization," IEEE Transactions on Cybernetics,
            vol. 47, no. 12, pp. 4108-4121, 2017. Available: https://ieeexplore.ieee.org/abstract/document/7553457
    """


    def __init__(self, d=None, m=None, ref_num=1000):
        """
        :param d: number of decision variables
        :param m: number of objectives
        :param ref_num: number of reference points, defaults to 1000
        """
        super().__init__()
        self.nk = 5
        self.m = 3 if m is None else m
        self.d = self.m * 100 if d is None else d
        self.ref_num = ref_num
        self._calc_pf()

        # Calculate the number of subgroup and their length
        c = [3.8 * 0.1 * (1 - 0.1)]
        for i in range(1, self.m):
            c.append(3.8 * c[-1] * (1 - c[-1]))
        c = torch.tensor(c, dtype=torch.float32)
        self.sublen = torch.floor(c / torch.sum(c) * (self.d - self.m + 1) / self.nk)
        self.len = torch.cat([torch.tensor([0]), torch.cumsum(self.sublen * self.nk, dim=0)], dim=0)
        self.sublen = tuple(map(int, self.sublen))
        self.len = tuple(map(int, self.len))

        self.sphere = sphere_func
        self.griewank = griewank_func
        self.rosenbrock = rosenbrock_func
        self.ackley = partial(ackley_func, a=20, b=0.2, c=2 * torch.pi)
        self.rastrigin = rastrigin_func
        self.schwefel = schwefel_func

    def evaluate(self, X: torch.Tensor):
        X_ = self._calc_X_(X)
        g = self._calc_g(X_)
        return self._calc_f(X_, g)

    def pf(self):
        return self._pf_value

    def _calc_pf(self):
        f = uniform_sampling(self.ref_num * self.m, self.m)[0] / 2
        self._pf_value = f

    def _calc_X_(self, X: torch.Tensor):
        m = self.m
        n, d = X.size()
        X_ = X.clone()
        X_[:, m - 1 :] = (1 + (torch.arange(m, d + 1, device=X_.device) / d).unsqueeze(0).repeat(n, 1)) * X_[:, m - 1 :] - X_[
            :, :1
        ] * 10
        return X_

    def _calc_g(self, X: torch.Tensor):
        return self._inner_calc_g([self.sphere], X)

    def _calc_f(self, X_: torch.Tensor, g):
        m = self.m
        n, d = X_.size()
        ones_col = torch.ones(n, 1, device=X_.device)
        cumprod_part = torch.cumprod(torch.cat([ones_col, X_[:, : m - 1]], dim=1), dim=1)
        f = (1 + g) * torch.flip(cumprod_part, [1]) * torch.cat([ones_col, 1 - torch.flip(X_[:, : m - 1], [1])], dim=1)
        return f

    def _inner_calc_g(self, inner_funcs, x):
        g_list = []

        for len_, sublen, func in zip(self.len, self.sublen, cycle(inner_funcs)):
            g_sum = []
            for j in range(self.nk):
                start = len_ + self.m - 1 + j * sublen
                end = start + sublen
                X_slice = x[:, start:end]
                g_sum.append(func(x=X_slice))
            g_list.append(torch.stack(g_sum, dim=1).sum(dim=1))

        g = torch.stack(g_list, dim=1)
        sublen_tensor = torch.tensor(self.sublen, device=x.device).float()
        g = g / sublen_tensor / self.nk
        return g


class LSMOP1(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)


class LSMOP2(LSMOP):
    def __init__(self, d=None, m=3, ref_num=1000):
        super().__init__(d, m, ref_num)

    def _calc_g(self, X: torch.Tensor):
        return self._inner_calc_g([self.griewank, self.schwefel], X)


class LSMOP3(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def _calc_g(self, X: torch.Tensor):
        return self._inner_calc_g([self.rastrigin, self.rosenbrock], X)


class LSMOP4(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def _calc_g(self, X: torch.Tensor):
        return self._inner_calc_g([self.ackley, self.griewank], X)


class LSMOP5(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def _calc_X_(self, X):
        m = self.m
        n, d = X.size()
        X_ = X.clone()
        X_[:, m - 1 :] = (
            1 + torch.cos(torch.arange(m, d + 1, device=X_.device) / d * torch.pi / 2).unsqueeze(0).repeat(n, 1)
        ) * X_[:, m - 1 :] - X_[:, :1] * 10
        return X_

    def _calc_f(self, X_: torch.Tensor, g):
        m = self.m
        n, d = X_.size()
        ones_col = torch.ones(n, 1, device=X_.device)
        cumprod_part = torch.cumprod(torch.cat([ones_col, torch.cos(X_[:, : m - 1] * torch.pi / 2)], dim=1), dim=1)
        last_part = torch.sin(torch.flip(X_[:, : m - 1], [1]) * torch.pi / 2)
        f = (
            (1 + g + torch.cat([g[:, 1:], torch.zeros(n, 1, device=X_.device)], dim=1))
            * torch.flip(cumprod_part, [1])
            * torch.cat([ones_col, last_part], dim=1)
        )
        return f

    def _calc_pf(self):
        f = uniform_sampling(self.ref_num * self.m, self.m)[0] / 2
        self._pf_value = f / torch.sqrt(torch.sum(f**2, dim=1, keepdims=True))


class LSMOP6(LSMOP5):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def _calc_g(self, X: torch.Tensor):
        return self._inner_calc_g([self.rosenbrock, self.schwefel], X)


class LSMOP7(LSMOP5):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def _calc_g(self, X: torch.Tensor):
        return self._inner_calc_g([self.ackley, self.rosenbrock], X)


class LSMOP8(LSMOP5):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def _calc_g(self, X: torch.Tensor):
        return self._inner_calc_g([self.griewank, self.sphere], X)


class LSMOP9(LSMOP):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def _calc_g(self, X: torch.Tensor):
        g = self._inner_calc_g([self.sphere, self.ackley], X)
        return 1 + torch.sum(g, dim=1, keepdims=True)

    def _calc_f(self, X_: torch.Tensor, g):
        m = self.m
        n, d = X_.size()
        f = torch.zeros(n, m, device=X_.device)
        f[:, : m - 1] = X_[:, : m - 1]
        f[:, m - 1 :] = (1 + g) * (
            m
            - torch.sum(
                f[:, : m - 1] / (1 + g) * (1 + torch.sin(3 * torch.pi * f[:, : m - 1])),
                dim=1,
                keepdims=True,
            )
        )
        return f

    def _calc_pf(self):
        interval = [0, 0.251412, 0.631627, 0.859401]
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        N = self.ref_num * self.m
        X = self._grid(N, self.m - 1)
        X = torch.where(
            X <= median,
            X * (interval[1] - interval[0]) / median + interval[0],
            (X - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
        )
        p = torch.cat(
            [
                X,
                2 * (self.m - torch.sum(X / 2 * (1 + torch.sin(3 * torch.pi * X)), dim=1, keepdims=True)),
            ],
            dim=1,
        )
        self._pf_value = p

    def _grid(self, N, M):
        gap = torch.linspace(0, 1, int(torch.ceil(torch.tensor(N ** (1 / M)))))
        c = torch.meshgrid(*([gap] * M), indexing="ij")
        w = torch.stack(c, dim=1).reshape(-1, M)
        return w
