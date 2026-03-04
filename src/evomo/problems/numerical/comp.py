import torch
from evox.core import Problem
from evox.operators.sampling import uniform_sampling


class SimpleCMOP(Problem):
    """
    Simple constrained multi-objective problem for testing.
    evaluate returns (f, cv).
    """

    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000):
        super().__init__()
        self.d = d
        self.m = m
        self.ref_num = ref_num
        self.sample, _ = uniform_sampling(self.ref_num * self.m, self.m)
        self.device = self.sample.device

    def evaluate(self, X: torch.Tensor):
        n, d = X.shape

        x0 = X[:, 0:1]
        x1 = X[:, 1:2]

        g = torch.sum((X[:, 1:] - 0.5) ** 2, dim=1, keepdim=True)

        f1 = x0
        f2 = 1.0 - torch.sqrt(torch.clamp(x0, min=0.0)) + g

        f = torch.cat([f1, f2], dim=1)

        c1 = x0 + x1 - 1.0
        c2 = 0.2 - x0 * x1

        cv = torch.clamp(c1, min=0.0) + torch.clamp(c2, min=0.0)

        return f, cv

    def pf(self):
        f = self.sample
        f1 = torch.clamp(f[:, 0:1], 0, 1)
        f2 = 1.0 - torch.sqrt(torch.clamp(f1, min=0.0))
        return torch.cat([f1, f2], dim=1)