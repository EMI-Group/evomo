from typing import Optional

import torch
from evox.core import Problem
from evox.operators.sampling import uniform_sampling
from evox.problems.numerical.basic import griewank_func, rastrigin_func, rosenbrock_func, sphere_func

from evomo.operators.selection import non_dominate_rank


class MAF(Problem):
    """
    The MAF benchmark test suite for evolutionary many-objective optimization.

    :references:
        [1] R. Cheng, M. Li, Y. Tian, X. Zhang, S. Yang, Y. Jin, and X. Yao, "A benchmark test suite
            for evolutionary many-objective optimization," Complex & Intelligent Systems, vol. 3,
            no. 1, pp. 67-81, 2017. Available: https://link.springer.com/article/10.1007/s40747-017-0039-7
    """

    def __init__(self, d: int = None, m: int = 3, ref_num: int = 1000, device: Optional[torch.device] = None):
        """
        Initialize the MAF problem instance.

        :param d: The dimensionality of the problem (number of decision variables). If None, it is set to m + 9.
        :param m: The number of objectives for the problem. Default is 3.
        :param ref_num: The reference number for evaluating the problem, default is 1000.
        :param device: The device to perform computations on (e.g., CPU or GPU). If None, the default device is used.
        """
        super().__init__()
        self.device = device or torch.get_default_device()
        self.m = m
        self.d = self.m + 9 if d is None else d
        self.ref_num = ref_num
        self._calc_pf()

    def _calc_pf(self):
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        self._pf_value = 1 - r

    def evaluate(self, X: torch.Tensor):
        pass

    def pf(self):
        return self._pf_value


class MAF1(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert d == m + 9, f"{self.__class__.__name__} is only defined for d = m + 9, got {d}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        g = torch.sum(torch.pow(X[:, m - 1 :] - 0.5, 2), dim=1)

        ones_col = torch.ones(n, 1, device=X.device)
        cumprod_term = torch.flip(torch.cumprod(torch.cat([ones_col, X[:, : m - 1]], dim=1), dim=1), [1])
        reversed_term = torch.cat([ones_col, 1 - torch.flip(X[:, : m - 1], [1])], dim=1)  # Reversed slice for last term
        repeat_g = (1 + g).unsqueeze(1)

        f = repeat_g - repeat_g * cumprod_term * reversed_term
        return f


class MAF2(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert d == m + 9, f"{self.__class__.__name__} is only defined for d = m + 9, got {d}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        g = torch.zeros(n, m, device=X.device)
        interval = int((d - m + 1) / m)
        g_cols = []
        for i in range(m):
            if i < m - 1:
                start = m + i * interval - 1
                seg = X[:, start : start + interval]
            else:
                start = m + (m - 1) * interval - 1
                seg = X[:, start:]

            seg = torch.where(seg == 0, 0.5, seg)
            g_cols.append(((seg - 0.5) ** 2).sum(dim=1, keepdim=True))
        g = torch.cat(g_cols, dim=1)

        f1 = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), torch.cos((X[:, : m - 1] / 2 + 0.25) * torch.pi / 2)], dim=1),
                dim=1,
            ),
            [1],
        )
        f2 = torch.cat(
            [torch.ones(n, 1, device=X.device), torch.sin((torch.flip(X[:, : m - 1], [1]) / 2 + 0.25) * torch.pi / 2)], dim=1
        )
        f = (1 + g) * f1 * f2
        return f

    def _calc_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        c = torch.zeros(n, m - 1, device=r.device)

        for i in range(n):
            for j in range(1, m):
                temp = r[i, j] / r[i, 0] * torch.prod(c[i, m - j : m - 1])
                c[i, m - j - 1] = torch.sqrt(1 / (1 + temp**2))

        if m > 5:
            c = c * (
                torch.cos(torch.tensor(torch.pi / 8, device=r.device))
                - torch.cos(torch.tensor(3 * torch.pi / 8, device=r.device))
                + torch.cos(torch.tensor(3 * torch.pi / 8, device=r.device))
            )
        else:
            c = c[
                torch.all(
                    (c >= torch.cos(torch.tensor(torch.pi * 3 / 8, device=r.device)))
                    & (c <= torch.cos(torch.tensor(torch.pi / 8, device=r.device))),
                    dim=1,
                ),
                :,
            ]
        f = torch.flip(
            torch.cumprod(torch.cat([torch.ones(c.size(0), 1, device=r.device), c[:, : m - 1]], dim=1), dim=1), [1]
        ) * torch.cat([torch.ones(c.size(0), 1, device=r.device), torch.sqrt(1 - torch.flip(c[:, : m - 1], [1]) ** 2)], dim=1)
        self._pf_value = f


class MAF3(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert d == m + 9, f"{self.__class__.__name__} is only defined for d = m + 9, got {d}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        g = 100 * (d - m + 1 + torch.sum((X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)), dim=1))
        f1 = (
            (1 + g).unsqueeze(1)
            * torch.flip(
                torch.cumprod(
                    torch.cat([torch.ones(n, 1, device=X.device), torch.cos(X[:, : m - 1] * torch.pi / 2)], dim=1),
                    dim=1,
                ),
                [1],
            )
            * torch.cat([torch.ones(n, 1, device=X.device), torch.sin(torch.flip(X[:, : m - 1], [1]) * torch.pi / 2)], dim=1)
        )
        f = torch.cat([f1[:, : m - 1] ** 4, (f1[:, m - 1] ** 2).view(-1, 1)], dim=1)
        return f

    def _calc_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)

        temp = (torch.sum(torch.sqrt(r[:, :-1]), dim=1) + r[:, -1]).view(-1, 1)
        f = r / torch.cat([(temp**2).repeat(1, m - 1), temp], dim=1)
        self._pf_value = f


class MAF4(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert d == m + 9, f"{self.__class__.__name__} is only defined for d = m + 9, got {d}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        g = 100 * (d - m + 1 + torch.sum((X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)), dim=1))
        f1 = (1 + g).unsqueeze(1) - (1 + g).unsqueeze(1) * torch.flip(
            torch.cumprod(
                torch.cat([torch.ones(n, 1, device=X.device), torch.cos(X[:, : m - 1] * torch.pi / 2)], dim=1), dim=1
            ),
            [1],
        ) * torch.cat([torch.ones(n, 1, device=X.device), torch.sin(torch.flip(X[:, : m - 1], [1]) * torch.pi / 2)], dim=1)
        f = f1 * torch.pow(2, torch.arange(1, m + 1, device=X.device))
        return f

    def _calc_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        r1 = r / torch.sqrt(torch.sum(r**2, dim=1, keepdim=True))
        f = (1 - r1) * torch.pow(2, torch.arange(1, m + 1, device=r.device))
        self._pf_value = f


class MAF5(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        alpha = 100
        Xfront = X[:, : m - 1] ** alpha
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1)
        f1 = (
            (1 + g).unsqueeze(1)
            * torch.flip(
                torch.cumprod(
                    torch.cat([torch.ones(n, 1, device=X.device), torch.cos(Xfront * torch.pi / 2)], dim=1),
                    dim=1,
                ),
                [1],
            )
            * torch.cat([torch.ones(n, 1, device=X.device), torch.sin(torch.flip(Xfront, [1]) * torch.pi / 2)], dim=1)
        )
        f = f1 * torch.pow(2, torch.arange(m, 0, -1, device=X.device))
        return f

    def _calc_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        r1 = r / torch.sqrt(torch.sum(r**2, dim=1, keepdim=True))
        f = r1 * torch.pow(2, torch.arange(m, 0, -1, device=r.device))
        self._pf_value = f


class MAF6(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert d == m + 9, f"{self.__class__.__name__} is only defined for d = m + 9, got {d}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1).unsqueeze(1)
        temp = g.repeat(1, m - 2)
        Xfront = X[:, : m - 1].clone()
        Xfront[:, 1:] = (1 + 2 * temp * X[:, 1 : m - 1]) / (2 + 2 * temp)
        f = (
            (1 + 100 * g)
            * torch.flip(
                torch.cumprod(
                    torch.cat([torch.ones(n, 1, device=X.device), torch.cos(Xfront * torch.pi / 2)], dim=1),
                    dim=1,
                ),
                [1],
            )
            * torch.cat([torch.ones(n, 1, device=X.device), torch.sin(torch.flip(Xfront, [1]) * torch.pi / 2)], dim=1)
        )
        return f

    def _calc_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, 2)
        r1 = r / torch.sqrt(torch.sum(r**2, dim=1, keepdim=True))

        if r1.size(1) < m:
            r1 = torch.cat((r1[:, torch.zeros(m - r1.size(1), device=r.device, dtype=torch.long)], r1), dim=1)
        f = r1 / torch.pow(torch.sqrt(torch.tensor(2.0)), torch.maximum(torch.tensor(self.m - 2), torch.tensor(0)).repeat(n, 1))

        self._pf_value = f


class MAF7(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert d == m + 19, f"{self.__class__.__name__} is only defined for d = m + 19, got {d}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        g = 1 + 9 * torch.mean(X[:, m - 1 :], dim=1)
        fFront = X[:, : m - 1]
        fRear = (1 + g) * (
            m
            - torch.sum(
                fFront / (1 + g.unsqueeze(1)) * (1 + torch.sin(3 * torch.pi * fFront)),
                dim=1,
            )
        )
        f = torch.cat([fFront, fRear.unsqueeze(1)], dim=1)
        return f

    @torch.jit.ignore
    def _calc_pf(self):
        m = self.m
        interval = torch.tensor([0, 0.251412, 0.631627, 0.859401], device=self.device)
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
        X = self._grid(self.ref_num * self.m, m - 1)
        X = torch.where(X <= median, X * (interval[1] - interval[0]) / median + interval[0], X)
        X = torch.where(
            X > median,
            (X - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
            X,
        )
        f = torch.cat(
            [
                X,
                2 * (self.m - torch.sum(X / 2 * (1 + torch.sin(3 * torch.pi * X)), dim=1)).view(-1, 1),
            ],
            dim=1,
        )
        self._pf_value = f

    def _grid(self, N: int, M: int):
        gap = torch.linspace(0, 1, steps=int(N ** (1 / M)), device=self.device)
        mesh = torch.meshgrid(*([gap] * M), indexing="ij")
        W = torch.cat([x.reshape(-1, 1) for x in mesh], dim=1)
        return W


class MAF8(MAF):
    def __init__(self, d=2, m=10, ref_num=1000, device: Optional[torch.device] = None):
        assert d == 2, f"{self.__class__.__name__} is only defined for d = 2, got {d}."
        assert m >= 3, f"{self.__class__.__name__} is only defined for m >= 3, got {m}."
        super().__init__(d, m, ref_num, device)
        self.points = self._getPoints()

    def evaluate(self, X: torch.Tensor):
        f = self._eucl_dis(X[:, :2], self.points)
        return f

    def _calc_pf(self):
        if not hasattr(self, "points"):
            self.points = self._getPoints()

        temp = torch.linspace(
            -1, 1, steps=int(torch.sqrt(torch.tensor(self.ref_num * self.m, device=self.device))), device=self.device
        )
        y, x = torch.meshgrid(temp, temp, indexing="ij")
        x = x.flatten()
        y = y.flatten()
        _points = torch.stack([x, y], dim=-1)
        ND = torch.stack([self._point_in_polygon(self.points, p) for p in _points])
        f = self._eucl_dis(torch.stack([x[ND], y[ND]], dim=-1), self.points)
        self._pf_value = f

    def _eucl_dis(self, X: torch.Tensor, Y: torch.Tensor):
        return torch.cdist(X, Y)

    def _getPoints(self):
        m = self.m
        theta, rho = self._cart2pol(torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))
        temp = torch.arange(1, m + 1, device=self.device).view(-1, 1)
        x, y = self._pol2cart(theta - temp * 2 * torch.pi / m, rho)
        return torch.cat([x, y], dim=1)

    def _cart2pol(self, x: torch.Tensor, y: torch.Tensor):
        rho = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        return theta, rho

    def _pol2cart(self, theta: torch.Tensor, rho: torch.Tensor):
        x = rho * torch.cos(theta)
        y = rho * torch.sin(theta)
        return x, y

    def _inside(self, x, a, b):
        return (torch.minimum(a, b) <= x) & (x < torch.maximum(a, b))

    def _ray_intersect_segment(self, point, seg_init, seg_term):
        y_dist = seg_term[1] - seg_init[1]
        # special case: y_dist == 0, check P_y == seg_init_y and P_x inside the segment
        judge_1 = (point[1] == seg_init[1]) & self._inside(point[0], seg_init[0], seg_term[0])
        # check intersection_x >= P_x.
        LHS = seg_init[0] * y_dist + (point[1] - seg_init[1]) * (seg_term[0] - seg_init[0])
        RHS = point[0] * y_dist
        # since it's an inequation, reverse the inequation if y_dist is negative.
        judge_2 = ((y_dist > 0) & (LHS >= RHS)) | ((y_dist < 0) & (LHS <= RHS))
        # check intersection_y, which is P_y is inside the segment
        judge_3 = self._inside(point[1], seg_init[1], seg_term[1])
        return ((y_dist == 0) & judge_1) | ((y_dist != 0) & judge_2 & judge_3)

    def _point_in_polygon(self, polygon, point):
        seg_term = torch.roll(polygon, 1, dims=0)
        is_intersect_list = []

        for i in range(polygon.size(0)):
            intersect = self._ray_intersect_segment(point, polygon[i], seg_term[i])
            is_intersect_list.append(intersect.unsqueeze(0))

        is_intersect = torch.cat(is_intersect_list, dim=0)
        is_vertex = torch.any(torch.all(polygon == point, dim=1))
        return (torch.sum(is_intersect) % 2 == 1) | is_vertex


class MAF9(MAF8):
    def __init__(self, d=2, m=10, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        f_cols = []
        for i in range(self.points.size(0) - 1):
            col_i = self._Point2Line(X, self.points[i : i + 2, :])
            f_cols.append(col_i.unsqueeze(1))
        last_col = self._Point2Line(X, torch.cat([self.points[-1, :].unsqueeze(0), self.points[0, :].unsqueeze(0)], dim=0))
        f_cols.append(last_col.unsqueeze(1))

        f = torch.cat(f_cols, dim=1)
        return f

    def _Point2Line(self, pop_dec: torch.Tensor, line: torch.Tensor):
        Distance = torch.abs(
            (line[0, 0] - pop_dec[:, 0]) * (line[1, 1] - pop_dec[:, 1])
            - (line[1, 0] - pop_dec[:, 0]) * (line[0, 1] - pop_dec[:, 1])
        ) / torch.sqrt((line[0, 0] - line[1, 0]) ** 2 + (line[0, 1] - line[1, 1]) ** 2)
        return Distance


class MAF10(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert d == m + 9, f"{self.__class__.__name__} is only defined for d = m + 9, got {d}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        d = self.d
        s = torch.arange(2, 2 * m + 1, step=2, device=X.device)
        z01 = X / torch.arange(2, d * 2 + 1, step=2, device=X.device)
        t0Front = z01[:, : m - 1]
        t0Rear = self._s_linear(z01[:, m - 1 :], 0.35)
        t0 = torch.cat([t0Front, t0Rear], dim=1)

        t = self._evaluate(t0, X)

        tRear = t[:, m - 1 : m]
        xFront = torch.maximum(tRear, torch.ones_like(tRear)) * (t[:, : m - 1] - 0.5) + 0.5
        x = torch.cat([xFront, tRear], dim=1)

        h_convex = self._convex(x)[:, : m - 1]
        h_mixed = self._mixed(x).unsqueeze(1)
        h = torch.cat([h_convex, h_mixed], dim=1)

        f = tRear + s * h
        return f

    def _calc_pf(self):
        m = self.m
        x, temp, a = self._pf_a()
        e = torch.abs(
            temp.unsqueeze(1) * (1 - torch.cos(torch.pi / 2 * a))
            - 1
            + (a + torch.cos(10 * torch.pi * a + torch.pi / 2) / 10 / torch.pi)
        )
        rank = torch.argsort(e, dim=1)

        x[:, 0] = a[0, torch.min(rank[:, :10], dim=1)[1]]
        f = self._convex(x)
        f[:, m - 1] = self._mixed(x)
        f = f * torch.arange(2, 2 * m + 1, 2, device=self.device)
        self._pf_value = f

    def _evaluate(self, t1: torch.Tensor, X: torch.Tensor):
        m = self.m
        d = self.d
        t2 = t1.clone()
        t2[:, m - 1 :] = self._b_flat(t1[:, m - 1 :], 0.8, 0.75, 0.85)

        t3 = t2**0.02

        outs = []
        for i in range(m - 1):
            y = t3[:, i : i + 2]
            w = torch.arange(2 * i, 2 * (i + 1) + 1, 2, device=X.device)
            outs.append(self._r_sum(y, w))

        y_last = t3[:, m - 1 : d]
        w_last = torch.arange(2 * m, 2 * d + 1, 2, device=X.device)
        outs.append(self._r_sum(y_last, w_last))

        t = torch.stack(outs, dim=1)

        return t

    def _s_linear(self, Y: torch.Tensor, a):
        return torch.abs(Y - a) / torch.abs(torch.floor(a - Y) + a)

    def _b_flat(self, Y: torch.Tensor, a, b, c):
        output = (
            a
            + torch.minimum(torch.zeros_like(Y - b, device=Y.device), torch.floor(Y - b)) * a * (b - Y) / b
            - torch.minimum(torch.zeros_like(c - Y, device=Y.device), torch.floor(c - Y)) * (1 - a) * (Y - c) / (1 - c)
        )
        return torch.round(output * 1e4) / 1e4

    def _r_sum(self, Y: torch.Tensor, W: torch.Tensor):
        return torch.sum(Y * W, dim=1) / torch.sum(W)

    def _convex(self, x: torch.Tensor):
        ones = torch.ones(x.size(0), 1, device=x.device)
        return torch.flip(
            torch.cumprod(torch.cat([ones, 1 - torch.cos(x[:, :-1] * torch.pi / 2)], dim=1), dim=1), [1]
        ) * torch.cat([ones, 1 - torch.sin(torch.flip(x[:, :-1], [1]) * torch.pi / 2)], dim=1)

    def _mixed(self, x: torch.Tensor):
        return 1 - x[:, 0] - torch.cos(10 * torch.pi * x[:, 0] + torch.pi / 2) / (10 * torch.pi)

    def _pf_a(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        c = torch.ones(n, m, device=r.device)

        for i in range(n):
            for j in range(1, m):
                temp = r[i, j] / r[i, 0] * torch.prod(1 - c[i, m - j : m - 1])
                c[i, m - j - 1] = (temp**2 - temp + torch.sqrt(2 * temp)) / (temp**2 + 1)

        x = torch.arccos(c) * 2 / torch.pi
        temp = (1 - torch.sin(torch.pi / 2 * x[:, 1])) * r[:, m - 1] / r[:, m - 2]
        a = torch.arange(0, 1.0001, 0.0001, device=r.device).unsqueeze(0)
        return x, temp, a


class MAF11(MAF10):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        d = m + 9 if d is None else d
        d = int((d - m + 1) / 2) * 2 + m - 1
        super().__init__(d, m, ref_num, device)

    def _calc_pf(self):
        m = self.m
        x, temp, a = self._pf_a()
        e = torch.abs(temp.unsqueeze(1) * (1 - torch.cos(torch.pi / 2 * a)) - 1 + a * torch.cos(5 * torch.pi * a) ** 2)
        rank = torch.argsort(e, dim=1)
        x[:, 0] = a[0, torch.min(rank[:, :10], dim=1)[1]]
        f = self._convex(x)
        f[:, m - 1] = self._mixed(x)
        non_dominated_rank = non_dominate_rank(f)
        f = f[non_dominated_rank == 0, :]
        f = f * torch.arange(2, 2 * m + 1, 2, device=x.device)
        self._pf_value = f

    def _evaluate(self, t1: torch.Tensor, X: torch.Tensor):
        m = self.m
        d = self.d
        L = d - (m - 1)

        t2Front = t1[:, : m - 1]
        t2Rear = (t1[:, m - 1 :: 2] + t1[:, m::2] + 2 * torch.abs(t1[:, m - 1 :: 2] - t1[:, m::2])) / 3
        t2 = torch.cat([t2Front, t2Rear], dim=1)

        outs = []
        w = torch.ones(1, device=X.device)
        for i in range(m - 1):
            y = t2[:, i : i + 1]
            outs.append(self._r_sum(y, w))

        y_last = t2[:, m - 1 : m - 1 + L // 2]
        w_last = torch.ones(L // 2, device=X.device)
        outs.append(self._r_sum(y_last, w_last))

        t = torch.stack(outs, dim=1)
        return t

    def _mixed(self, x: torch.Tensor):
        return self._disc(x)

    def _disc(self, x):
        return 1 - x[:, 0] * (torch.cos(5 * torch.pi * x[:, 0])) ** 2


class MAF12(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert d == m + 9, f"{self.__class__.__name__} is only defined for d = m + 9, got {d}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        d = self.d
        L = d - (m - 1)
        S = torch.arange(2, 2 * m + 1, step=2, device=X.device)

        z01 = X / torch.arange(2, d * 2 + 1, step=2, device=X.device)

        Y = (torch.flip(torch.cumsum(torch.flip(z01, [1]), dim=1), [1]) - z01) / torch.arange(d - 1, -1, -1, device=X.device)
        t1Front = z01[:, : d - 1] ** (
            0.02
            + (50 - 0.02)
            * (0.98 / 49.98 - (1 - 2 * Y[:, : d - 1]) * torch.abs(torch.floor(0.5 - Y[:, : d - 1]) + 0.98 / 49.98))
        )
        t1Rear = z01[:, d - 1 :]
        t1 = torch.cat([t1Front, t1Rear], dim=1)

        t2Front = self._s_decept(t1[:, : m - 1], 0.35, 0.001, 0.05)
        t2Rear = self._s_multi(t1[:, m - 1 :], 30, 95, 0.35)
        t2 = torch.cat([t2Front, t2Rear], dim=1)

        t3_cols = []
        for i in range(m - 1):
            temp = t2[:, i : i + 1]
            t3_cols.append(self._r_nonsep(temp, 1))

        diff = torch.abs(t2Rear.unsqueeze(2) - t2Rear.unsqueeze(1))
        SUM = diff.sum(dim=(1, 2)) * 0.5

        denom1 = torch.tensor(L / 2, device=X.device)
        denom2 = 1 + 2 * L - 2 * denom1
        last_col = (t2Rear.sum(dim=1) + 2 * SUM) / denom1 / denom2
        t3_cols.append(last_col)

        t3 = torch.stack(t3_cols, dim=1)

        x_cols = []
        max_factor = torch.maximum(last_col, torch.ones_like(last_col))
        for i in range(m - 1):
            x_cols.append(max_factor * (t3[:, i] - 0.5) + 0.5)
        x_cols.append(last_col)
        x = torch.stack(x_cols, dim=1)
        h = self._concave(x)
        f = x[:, m - 1].unsqueeze(1) + S * h
        return f

    def _calc_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        r = r / torch.sqrt(torch.sum(r**2, dim=1)).reshape(-1, 1)
        f = torch.arange(2, 2 * m + 1, 2) * r
        self._pf_value = f

    def _s_decept(self, Y: torch.Tensor, a, b, c):
        return 1 + (torch.abs(Y - a) - b) * (
            torch.floor(Y - a + b) * (1 - c + (a - b) / b) / (a - b)
            + torch.floor(a + b - Y) * (1 - c + (1 - a - b) / b) / (1 - a - b)
            + 1 / b
        )

    def _s_multi(self, Y: torch.Tensor, a, b, c):
        return (
            1
            + torch.cos((4 * a + 2) * torch.pi * (0.5 - torch.abs(Y - c) / 2 / (torch.floor(c - Y) + c)))
            + 4 * b * (torch.abs(Y - c) / 2 / (torch.floor(c - Y) + c)) ** 2
        ) / (b + 2)

    def _r_nonsep(self, Y: torch.Tensor, a):
        s1 = Y.sum(dim=1)
        if a > 1:
            s2 = sum((Y - Y.roll(-o, dims=1)).abs().sum(dim=1) for o in range(1, a))
        else:
            s2 = torch.zeros_like(s1)
        int_a2 = a // 2

        return (s1 + s2) / (Y.size(1) / a) * int_a2 * (1 + 2 * a - 2 * int_a2)

    def _concave(self, X: torch.Tensor):
        return torch.flip(
            torch.cumprod(
                torch.cat([torch.ones(X.shape[0], 1, device=X.device), torch.sin(X[:, :-1] * torch.pi / 2)], dim=1),
                dim=1,
            ),
            [1],
        ) * torch.cat(
            [
                torch.ones(X.size(0), 1, device=X.device),
                torch.cos(torch.flip(X[:, :-1], [1]) * torch.pi / 2),
            ],
            dim=1,
        )


class MAF13(MAF):
    def __init__(self, d=5, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert m >= 3, f"{self.__class__.__name__} is only defined for m >= 3, got {m}."
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        d = self.d
        Y = X - 2 * X[:, 1].view(-1, 1) * torch.sin(
            2 * torch.pi * X[:, 0].view(-1, 1) + torch.arange(1, d + 1, device=X.device) * torch.pi / d
        )
        f = torch.zeros(n, m, device=X.device)
        f0 = torch.sin(X[:, 0] * torch.pi / 2) + 2 * torch.mean(Y[:, 3:d:3] ** 2, dim=1)
        f1 = torch.cos(X[:, 0] * torch.pi / 2) * torch.sin(X[:, 1] * torch.pi / 2) + 2 * torch.mean(Y[:, 4:d:3] ** 2, dim=1)
        f2 = torch.cos(X[:, 0] * torch.pi / 2) * torch.cos(X[:, 1] * torch.pi / 2) + 2 * torch.mean(Y[:, 2:d:3] ** 2, dim=1)
        f3 = (
            (f[:, 0] ** 2 + f[:, 1] ** 10 + f[:, 2] ** 10 + 2 * torch.mean(Y[:, 3:d] ** 2, dim=1))
            .unsqueeze(1)
            .repeat(1, self.m - 3)
        )
        f = torch.cat([f0.unsqueeze(1), f1.unsqueeze(1), f2.unsqueeze(1), f3], dim=1)
        return f

    def _calc_pf(self):
        m = self.m
        r, n = uniform_sampling(self.ref_num * self.m, 3)
        r = r / torch.sqrt(torch.sum(r**2, dim=1, keepdim=True))
        f = torch.cat([r, (r[:, 0] ** 2 + r[:, 1] ** 10 + r[:, 2] ** 10).unsqueeze(1).repeat(1, m - 3)], dim=1)
        self._pf_value = f


class MAF14(MAF):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        assert d == 20 * m, f"{self.__class__.__name__} is only defined for d = 20 * m, got {d}."
        super().__init__(d, m, ref_num, device)
        nk = 2
        c = torch.zeros(self.m, device=device)
        c[0] = 3.8 * 0.1 * (1 - 0.1)
        for i in range(1, self.m):
            c[i] = 3.8 * c[i - 1] * (1 - c[i - 1])

        self.sublen = torch.floor(c / torch.sum(c) * (self.d - self.m + 1) / nk)
        self.len = torch.cat([torch.tensor([0]), torch.cumsum(self.sublen * nk, dim=0)], dim=0)
        self.sublen = tuple(map(int, self.sublen))
        self.len = tuple(map(int, self.len))

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        g = self._evaluate(X)
        f = (
            (1 + g)
            * torch.flip(torch.cumprod(torch.cat([torch.ones(n, 1, device=X.device), X[:, : m - 1]], dim=1), dim=1), [1])
            * torch.cat([torch.ones(n, 1, device=X.device), 1 - torch.flip(X[:, : m - 1], [1])], dim=1)
        )
        return f

    def _calc_pf(self):
        self._pf_value = uniform_sampling(self.ref_num * self.m, self.m)[0]

    def _evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        nk = 2
        new_X = self._modify_X(X)
        g = torch.zeros(n, m, device=X.device)

        for i in range(0, m, 2):
            g = self._inner_loop(i, self._func1, g, nk, new_X)

        for i in range(1, m, 2):
            g = self._inner_loop(i, self._func2, g, nk, new_X)
        g = g / torch.tensor(self.sublen, device=X.device).unsqueeze(0) * nk
        return g

    def _modify_X(self, X: torch.Tensor):
        new_X = X.clone()
        new_X[:, self.m - 1 :] = (1 + torch.arange(self.m, self.d + 1, device=X.device) / self.d).unsqueeze(0) * X[
            :, self.m - 1 :
        ] - (X[:, 0] * 10).unsqueeze(-1)
        return new_X

    def _inner_loop(self, i, inner_fun, g: torch.Tensor, nk, X: torch.Tensor):
        new_col = g[:, i].clone()
        for j in range(1, nk):
            start = self.len[i] + self.m - 1 + j * self.sublen[i]
            end = start + self.sublen[i]
            temp = X[:, start:end]
            new_col = new_col + inner_fun(temp)
        g_out = torch.cat(
            [g[:, :i], new_col.unsqueeze(1), g[:, i + 1 :]],
            dim=1,
        )
        return g_out

    def _func1(self, X):
        return rastrigin_func(X)

    def _func2(self, X):
        return rosenbrock_func(X)


class MAF15(MAF14):
    def __init__(self, d=None, m=3, ref_num=1000, device: Optional[torch.device] = None):
        super().__init__(d, m, ref_num, device)

    def evaluate(self, X: torch.Tensor):
        m = self.m
        n = X.size(0)
        g = self._evaluate(X)
        f = (1 + g + torch.cat([g[:, 1:], torch.zeros(n, 1, device=X.device)], dim=1)) * (
            1
            - torch.flip(
                torch.cumprod(
                    torch.cat([torch.ones(n, 1, device=X.device), torch.cos(X[:, : m - 1] * torch.pi / 2)], dim=1),
                    dim=1,
                ),
                [1],
            )
            * torch.cat([torch.ones(n, 1, device=X.device), torch.sin(torch.flip(X[:, : m - 1], [1]) * torch.pi / 2)], dim=1)
        )
        return f

    def _calc_pf(self):
        r, n = uniform_sampling(self.ref_num * self.m, self.m)
        r = 1 - r / torch.sqrt(torch.sum(r**2, axis=1)).reshape(-1, 1)
        self._pf_value = r

    def _modify_X(self, X: torch.Tensor):
        new_X = X.clone()
        new_X[:, self.m - 1 :] = (
            1 + torch.cos(torch.arange(self.m, self.d + 1, device=X.device).unsqueeze(0) / self.d * torch.pi / 2)
        ) * X[:, self.m - 1 :] - (X[:, 0] * 10).unsqueeze(-1)
        return new_X

    def _func1(self, X):
        return griewank_func(X)

    def _func2(self, X):
        return sphere_func(X)
