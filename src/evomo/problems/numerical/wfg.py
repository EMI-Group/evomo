import torch
from evox.core import Problem
from evox.operators.sampling import uniform_sampling


def _const(value: float, x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=x.dtype, device=x.device)


def _normalize(X: torch.Tensor) -> torch.Tensor:
    upper = 2 * torch.arange(1, X.size(1) + 1, dtype=X.dtype, device=X.device)
    return X / upper


def _s_linear(y: torch.Tensor, a: float) -> torch.Tensor:
    a_t = _const(a, y)
    return torch.abs(y - a_t) / torch.abs(torch.floor(a_t - y) + a_t)


def _s_decept(y: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    a_t = _const(a, y)
    b_t = _const(b, y)
    c_t = _const(c, y)
    return 1 + (torch.abs(y - a_t) - b_t) * (
        torch.floor(y - a_t + b_t) * (1 - c_t + (a_t - b_t) / b_t) / (a_t - b_t)
        + torch.floor(a_t + b_t - y) * (1 - c_t + (1 - a_t - b_t) / b_t) / (1 - a_t - b_t)
        + 1 / b_t
    )


def _s_multi(y: torch.Tensor, a: int, b: int, c: float) -> torch.Tensor:
    c_t = _const(c, y)
    tmp = torch.abs(y - c_t) / 2 / (torch.floor(c_t - y) + c_t)
    return (1 + torch.cos((4 * a + 2) * torch.pi * (0.5 - tmp)) + 4 * b * tmp**2) / (b + 2)


def _b_flat(y: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    a_t = _const(a, y)
    b_t = _const(b, y)
    c_t = _const(c, y)
    output = a_t + torch.minimum(_const(0.0, y), torch.floor(y - b_t)) * a_t * (b_t - y) / b_t
    output = output - torch.minimum(_const(0.0, y), torch.floor(c_t - y)) * (1 - a_t) * (y - c_t) / (1 - c_t)
    return torch.round(output * 1e4) / 1e4


def _b_poly(y: torch.Tensor, a: float) -> torch.Tensor:
    return y**a


def _b_param(y: torch.Tensor, Y: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    a_t = _const(a, y)
    b_t = _const(b, y)
    c_t = _const(c, y)
    return y ** (b_t + (c_t - b_t) * (a_t - (1 - 2 * Y) * torch.abs(torch.floor(0.5 - Y) + a_t)))


def _r_sum(y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.sum(y * w.unsqueeze(0), dim=1) / torch.sum(w)


def _r_nonsep(y: torch.Tensor, a: int) -> torch.Tensor:
    n = y.size(1)
    if a == 1:
        return y.squeeze(1)
    if a == 2:
        return (y[:, 0] + y[:, 1] + 2 * torch.abs(y[:, 0] - y[:, 1])) / 3
    if a == n:
        pairwise = torch.abs(y.unsqueeze(2) - y.unsqueeze(1))
        pair_sum = torch.triu(pairwise, diagonal=1).sum(dim=(1, 2))
        numerator = y.sum(dim=1) + 2 * pair_sum
    else:
        shifts = torch.stack([torch.roll(y, shifts=-k, dims=1) for k in range(a - 1)], dim=2)
        numerator = (y + torch.abs(y.unsqueeze(2) - shifts).sum(dim=2)).sum(dim=1)
    ceil_a = torch.ceil(torch.as_tensor(a / 2, dtype=y.dtype, device=y.device))
    return numerator / (n / a) / ceil_a / (1 + 2 * a - 2 * ceil_a)


def _convex(x: torch.Tensor) -> torch.Tensor:
    left = torch.flip(
        torch.cumprod(torch.cat([torch.ones((x.size(0), 1), dtype=x.dtype, device=x.device), 1 - torch.cos(x[:, :-1] * torch.pi / 2)], dim=1), dim=1),
        dims=[1],
    )
    right = torch.cat([torch.ones((x.size(0), 1), dtype=x.dtype, device=x.device), 1 - torch.sin(torch.flip(x[:, :-1], dims=[1]) * torch.pi / 2)], dim=1)
    return left * right


def _concave(x: torch.Tensor) -> torch.Tensor:
    left = torch.flip(
        torch.cumprod(torch.cat([torch.ones((x.size(0), 1), dtype=x.dtype, device=x.device), torch.sin(x[:, :-1] * torch.pi / 2)], dim=1), dim=1),
        dims=[1],
    )
    right = torch.cat([torch.ones((x.size(0), 1), dtype=x.dtype, device=x.device), torch.cos(torch.flip(x[:, :-1], dims=[1]) * torch.pi / 2)], dim=1)
    return left * right


def _linear_shape(x: torch.Tensor) -> torch.Tensor:
    left = torch.flip(
        torch.cumprod(torch.cat([torch.ones((x.size(0), 1), dtype=x.dtype, device=x.device), x[:, :-1]], dim=1), dim=1),
        dims=[1],
    )
    right = torch.cat([torch.ones((x.size(0), 1), dtype=x.dtype, device=x.device), 1 - torch.flip(x[:, :-1], dims=[1])], dim=1)
    return left * right


def _mixed(x: torch.Tensor) -> torch.Tensor:
    return 1 - x[:, 0] - torch.cos(10 * torch.pi * x[:, 0] + torch.pi / 2) / 10 / torch.pi


def _disc(x: torch.Tensor) -> torch.Tensor:
    return 1 - x[:, 0] * torch.cos(5 * torch.pi * x[:, 0]) ** 2


class WFG(Problem):
    """
    Base class for the Walking Fish Group benchmark problems.

    Decision variables follow the PlatEMO scale: ``lower = 0`` and
    ``upper = [2, 4, ..., 2*d]``. The position parameter ``k`` defaults to
    ``m - 1`` and should be a multiple of ``m - 1``.
    """

    def __init__(self, d: int | None = None, m: int = 3, ref_num: int = 1000, k: int | None = None):
        super().__init__()
        self.m = m
        self.k = m - 1 if k is None else k
        if self.k % (self.m - 1) != 0:
            raise ValueError("k should be a multiple of m - 1.")
        self.d = self._adjust_d(self.k + 10 if d is None else d)
        if self.d <= self.k:
            raise ValueError("d should be greater than k.")
        self.ref_num = ref_num
        self.lower = torch.zeros(self.d)
        self.upper = 2 * torch.arange(1, self.d + 1, dtype=torch.float32)

    def _adjust_d(self, d: int) -> int:
        return d

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def pf(self) -> torch.Tensor:
        f = uniform_sampling(self.ref_num * self.m, self.m)[0]
        f = f / torch.sqrt(torch.sum(f**2, dim=1, keepdim=True))
        return f * (2 * torch.arange(1, self.m + 1, dtype=f.dtype, device=f.device)).unsqueeze(0)

    def _calculate_x(self, t: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        if a is None:
            a = torch.ones(self.m - 1, dtype=t.dtype, device=t.device)
        x = torch.empty((t.size(0), self.m), dtype=t.dtype, device=t.device)
        x[:, :-1] = torch.maximum(t[:, -1:], a.unsqueeze(0)) * (t[:, :-1] - 0.5) + 0.5
        x[:, -1] = t[:, -1]
        return x

    def _objectives(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        scales = 2 * torch.arange(1, self.m + 1, dtype=x.dtype, device=x.device)
        return x[:, -1:] + scales.unsqueeze(0) * h

    def _reduce_by_sum(self, t: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        weights = torch.ones(t.size(1), dtype=t.dtype, device=t.device) if weights is None else weights
        out = torch.empty((t.size(0), self.m), dtype=t.dtype, device=t.device)
        step = self.k // (self.m - 1)
        for i in range(self.m - 1):
            start = i * step
            out[:, i] = _r_sum(t[:, start : start + step], weights[start : start + step])
        out[:, -1] = _r_sum(t[:, self.k :], weights[self.k :])
        return out


class WFG1(WFG):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        z01 = _normalize(X)
        weights = 2 * torch.arange(1, self.d + 1, dtype=X.dtype, device=X.device)
        t1 = z01.clone()
        t1[:, self.k :] = _s_linear(z01[:, self.k :], 0.35)
        t2 = t1.clone()
        t2[:, self.k :] = _b_flat(t1[:, self.k :], 0.8, 0.75, 0.85)
        t3 = _b_poly(t2, 0.02)
        t4 = self._reduce_by_sum(t3, weights)
        x = self._calculate_x(t4)
        h = _convex(x)
        h[:, -1] = _mixed(x)
        return self._objectives(x, h)


class WFG2(WFG):
    def _adjust_d(self, d: int) -> int:
        return int(torch.ceil(torch.as_tensor((d - self.k) / 2)).item()) * 2 + self.k

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        z01 = _normalize(X)
        distance = self.d - self.k
        t1 = z01.clone()
        t1[:, self.k :] = _s_linear(z01[:, self.k :], 0.35)
        t2 = torch.empty((X.size(0), self.k + distance // 2), dtype=X.dtype, device=X.device)
        t2[:, : self.k] = t1[:, : self.k]
        t2[:, self.k :] = _r_nonsep(t1[:, self.k :].reshape(X.size(0), distance // 2, 2).reshape(-1, 2), 2).reshape(X.size(0), distance // 2)
        t3 = self._reduce_by_sum(t2)
        x = self._calculate_x(t3)
        h = _convex(x)
        h[:, -1] = _disc(x)
        return self._objectives(x, h)


class WFG3(WFG2):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        z01 = _normalize(X)
        distance = self.d - self.k
        t1 = z01.clone()
        t1[:, self.k :] = _s_linear(z01[:, self.k :], 0.35)
        t2 = torch.empty((X.size(0), self.k + distance // 2), dtype=X.dtype, device=X.device)
        t2[:, : self.k] = t1[:, : self.k]
        t2[:, self.k :] = _r_nonsep(t1[:, self.k :].reshape(X.size(0), distance // 2, 2).reshape(-1, 2), 2).reshape(X.size(0), distance // 2)
        t3 = self._reduce_by_sum(t2)
        a = torch.cat([torch.ones(1, dtype=X.dtype, device=X.device), torch.zeros(self.m - 2, dtype=X.dtype, device=X.device)])
        x = self._calculate_x(t3, a)
        return self._objectives(x, _linear_shape(x))

    def pf(self) -> torch.Tensor:
        x0 = torch.linspace(0, 1, self.ref_num, dtype=torch.float32)
        x = torch.cat([x0[:, None], torch.full((self.ref_num, self.m - 2), 0.5), torch.zeros((self.ref_num, 1))], dim=1)
        scales = 2 * torch.arange(1, self.m + 1, dtype=x.dtype)
        return _linear_shape(x) * scales.unsqueeze(0)


class WFG4(WFG):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        z01 = _normalize(X)
        t1 = _s_multi(z01, 30, 10, 0.35)
        t2 = self._reduce_by_sum(t1)
        x = self._calculate_x(t2)
        return self._objectives(x, _concave(x))


class WFG5(WFG):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        z01 = _normalize(X)
        t1 = _s_decept(z01, 0.35, 0.001, 0.05)
        t2 = self._reduce_by_sum(t1)
        x = self._calculate_x(t2)
        return self._objectives(x, _concave(x))


class WFG6(WFG):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        z01 = _normalize(X)
        t1 = z01.clone()
        t1[:, self.k :] = _s_linear(z01[:, self.k :], 0.35)
        t2 = torch.empty((X.size(0), self.m), dtype=X.dtype, device=X.device)
        step = self.k // (self.m - 1)
        for i in range(self.m - 1):
            start = i * step
            t2[:, i] = _r_nonsep(t1[:, start : start + step], step)
        t2[:, -1] = _r_nonsep(t1[:, self.k :], self.d - self.k)
        x = self._calculate_x(t2)
        return self._objectives(x, _concave(x))


class WFG7(WFG):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        z01 = _normalize(X)
        counts = torch.arange(self.d - 1, -1, -1, dtype=X.dtype, device=X.device)
        Y = (torch.flip(torch.cumsum(torch.flip(z01, dims=[1]), dim=1), dims=[1]) - z01) / torch.clamp(counts, min=1).unsqueeze(0)
        t1 = z01.clone()
        t1[:, : self.k] = _b_param(z01[:, : self.k], Y[:, : self.k], 0.98 / 49.98, 0.02, 50)
        t2 = t1.clone()
        t2[:, self.k :] = _s_linear(t1[:, self.k :], 0.35)
        t3 = self._reduce_by_sum(t2)
        x = self._calculate_x(t3)
        return self._objectives(x, _concave(x))


class WFG8(WFG):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        z01 = _normalize(X)
        counts = torch.arange(self.d, dtype=X.dtype, device=X.device)
        Y = (torch.cumsum(z01, dim=1) - z01) / torch.clamp(counts, min=1).unsqueeze(0)
        t1 = z01.clone()
        t1[:, self.k :] = _b_param(z01[:, self.k :], Y[:, self.k :], 0.98 / 49.98, 0.02, 50)
        t2 = t1.clone()
        t2[:, self.k :] = _s_linear(t1[:, self.k :], 0.35)
        t3 = self._reduce_by_sum(t2)
        x = self._calculate_x(t3)
        return self._objectives(x, _concave(x))


class WFG9(WFG):
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        z01 = _normalize(X)
        counts = torch.arange(self.d - 1, -1, -1, dtype=X.dtype, device=X.device)
        Y = (torch.flip(torch.cumsum(torch.flip(z01, dims=[1]), dim=1), dims=[1]) - z01) / torch.clamp(counts, min=1).unsqueeze(0)
        t1 = z01.clone()
        t1[:, :-1] = _b_param(z01[:, :-1], Y[:, :-1], 0.98 / 49.98, 0.02, 50)
        t2 = torch.empty_like(t1)
        t2[:, : self.k] = _s_decept(t1[:, : self.k], 0.35, 0.001, 0.05)
        t2[:, self.k :] = _s_multi(t1[:, self.k :], 30, 95, 0.35)
        t3 = torch.empty((X.size(0), self.m), dtype=X.dtype, device=X.device)
        step = self.k // (self.m - 1)
        for i in range(self.m - 1):
            start = i * step
            t3[:, i] = _r_nonsep(t2[:, start : start + step], step)
        t3[:, -1] = _r_nonsep(t2[:, self.k :], self.d - self.k)
        x = self._calculate_x(t3)
        return self._objectives(x, _concave(x))
