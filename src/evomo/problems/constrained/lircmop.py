
__all__ = ['LIRCMOP1', 'LIRCMOP2', 'LIRCMOP3', 'LIRCMOP4', 'LIRCMOP5', 'LIRCMOP6', 'LIRCMOP7',
           'LIRCMOP8', 'LIRCMOP9', 'LIRCMOP10', 'LIRCMOP11', 'LIRCMOP12', 'LIRCMOP13', 'LIRCMOP14']

import torch
from evox.operators.sampling import uniform_sampling
from typing import Optional
from evomo.problems.constrained.base import CMOP

class LIRCMOP(CMOP):
    """
    Constrained benchmark MOP with large infeasible regions.

    Z. Fan, W. Li, X. Cai, H. Huang, Y. Fang, Y. You, J. Mo, C. Wei, and E. Goodman, 
    An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions, 
    Soft Computing, 2019, 23: 12491-12510.

    :param d: Number of decision variables.
    :param m: Number of objectives.
    :param n_iq: Number of inequality constraints.
    :param n_eq: Number of equality constraints.
    :param ref_num: Number of reference points used in the problem.
    """

    def __init__(self, 
                 d: int = None, 
                 m: int = None, 
                 n_iq: int=None, 
                 ref_num: int = 1000, 
                 **kwargs):

        self.device = kwargs.pop("device", torch.get_default_device())
        lb = torch.zeros(d, device=self.device)
        ub = torch.ones(d, device=self.device)
        
        super().__init__(d=d,m=m,n_iq=n_iq,n_eq=0,lb=lb,ub=ub,**kwargs)
        
        self.ref_num = ref_num

class LIRCMOP1(LIRCMOP):
    def __init__(self, 
                 d: int = 30, 
                 m: int = 2, 
                 n_iq: int = 2, 
                 ref_num: int = 1000, 
                 **kwargs):
        
        super().__init__(d=d, m=m, n_iq=n_iq,ref_num=ref_num, **kwargs)
        
    def g1(self, X: torch.Tensor) -> torch.Tensor:
        h = torch.sin(0.5 * torch.pi * X[:, 0:1])
        g = torch.sum((X[:, 2::2] - h) ** 2, dim=1, keepdim=True)
        return g

    def g2(self, X: torch.Tensor) -> torch.Tensor:
        h = torch.cos(0.5 * torch.pi * X[:, 0:1])
        g = torch.sum((X[:, 1::2] - h) ** 2, dim=1, keepdim=True)
        return g

    def f1(self, X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return X[:, 0:1] + g

    def f2(self, X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.0 - X[:, 0:1] ** 2 + g

    def c(self, g: torch.Tensor) -> torch.Tensor:
        return (0.5 - g) * (0.51 - g)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c12 = self.c(torch.cat([g1, g2], dim=1))
        return torch.cat([f1, f2, c12], dim=1)

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t**2], dim=1) + 0.5
        return pf
    
class LIRCMOP2(LIRCMOP1):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 2, ref_num: int = 1000, **kwargs):
        
        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num, **kwargs)

    def f2(self, X: torch.Tensor, g: torch.Tensor)-> torch.Tensor:
        return 1 - torch.sqrt(X[:, 0:1]) + g

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - torch.sqrt(t)], dim=1) + 0.5
        return pf
    
class LIRCMOP3(LIRCMOP1):
    def __init__(self, 
                 d: int = 30, 
                 m: int = 2, 
                 n_iq: int = 3, 
                 ref_num: int = 1000, 
                 **kwargs):
        
        super().__init__(d=d, m=m, n_iq=n_iq,ref_num=ref_num, **kwargs)

    def c3(self, X: torch.Tensor) -> torch.Tensor:
        return 0.5 - torch.sin(20 * torch.pi * X[:, 0:1])

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c12 = self.c(torch.cat([g1, g2], dim=1))
        c3 = self.c3(X)
        return torch.cat([f1, f2, c12, c3],dim=1)

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t**2], dim=1)
        mask = torch.sin(20.0 * torch.pi * t) >= 0.5
        pf = pf[mask.squeeze(1)] + 0.5
        return pf

class LIRCMOP4(LIRCMOP2):
    def __init__(self, 
                 d: int = 30, 
                 m: int = 2, 
                 n_iq: int = 3, 
                 ref_num: int = 1000, 
                 **kwargs):
        
        super().__init__(d=d, m=m, n_iq=n_iq,ref_num=ref_num, **kwargs)

    def c3(self, X: torch.Tensor) -> torch.Tensor:
        return 0.5 - torch.sin(20 * torch.pi * X[:, 0:1])

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c12 = self.c(torch.cat([g1, g2], dim=1))
        c3 = self.c3(X)
        return torch.cat([f1, f2, c12, c3],dim=1)

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - torch.sqrt(t)], dim=1)
        mask = torch.sin(20.0 * torch.pi * t).squeeze(1) >= 0.5
        pf = pf[mask] + 0.5
        return pf

class LIRCMOP5(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 2, ref_num: int = 1000, 
                 cons_params: Optional[torch.Tensor] = None,
                 **kwargs):
        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num, **kwargs)
        
        # Constraint parameters
        if cons_params is None:
            cons_params = torch.tensor([
                [1.6, 1.6, -0.25 * torch.pi, 2.0, 4.0, 0.1],
                [2.5, 2.5, -0.25 * torch.pi, 2.0, 8.0, 0.1],
            ])
        self.cons_params = cons_params

    def g1(self, X: torch.Tensor) -> torch.Tensor:
        # Odd indices
        i = torch.arange(2, self.d, 2, device=X.device, dtype=X.dtype)
        h = torch.sin(0.5 * torch.pi * X[:, 0:1] * i / self.d)
        g = torch.sum((X[:, 2::2] - h) ** 2, dim=1, keepdim=True)
        return g

    def g2(self, X: torch.Tensor) -> torch.Tensor:
        # Even indices
        i = torch.arange(1, self.d, 2, device=X.device, dtype=X.dtype)
        h = torch.cos(0.5 * torch.pi * X[:, 0:1] * i / self.d)
        g = torch.sum((X[:, 1::2] - h) ** 2, dim=1, keepdim=True)
        return g

    def f1(self, X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return X[:, 0:1] + 10.0 * g + 0.7057

    def f2(self, X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.sqrt(X[:, 0:1]) + 10.0 * g + 0.7057

    # Using broadcasting
    def c(self, f1: torch.Tensor, f2: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Rotation ellipse constraint
        params = params.to(device=f1.device, dtype=f1.dtype)
        p, q, theta, a, b, r = params.unbind(dim=1)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        inv_a2, inv_b2 = 1.0 / (a * a), 1.0 / (b * b)
        dx, dy = f1 - p.unsqueeze(0), f2 - q.unsqueeze(0)
        h1 = (dx * cos_t.unsqueeze(0) - dy * sin_t.unsqueeze(0)).square() * inv_a2.unsqueeze(0)
        h2 = (dx * sin_t.unsqueeze(0) + dy * cos_t.unsqueeze(0)).square() * inv_b2.unsqueeze(0)
        return r.unsqueeze(0) - h1 - h2 

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        G = self.c(f1, f2, self.cons_params)
        return torch.cat([f1, f2, G], dim=1)
    
    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - torch.sqrt(t)], dim=1) + 0.7057
        G = self.c(pf[:, [0]], pf[:, [1]],self.cons_params)
        mask = torch.all(G <= 0, dim=1)
        return pf[mask]
   

class LIRCMOP6(LIRCMOP5):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 2, ref_num: int = 1000, cons_params: Optional[torch.Tensor] = None,**kwargs):
        if cons_params is None:
            cons_params = torch.tensor([
                [1.8, 1.8, -0.25 * torch.pi, 2.0, 8.0, 0.1],
                [2.8, 2.8, -0.25 * torch.pi, 2.0, 8.0, 0.1],
            ])
        self.cons_params=cons_params

        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num, cons_params=cons_params,**kwargs)

    def f2(self, X, g):
        return 1 - X[:, 0:1] ** 2 + 10 * g + 0.7057
    
    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t**2], dim=1) + 0.7057
        G = self.c(pf[:, [0]], pf[:, [1]],self.cons_params)
        mask = torch.all(G <= 0, dim=1)
        return pf[mask]

    
class LIRCMOP7(LIRCMOP5):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 3, ref_num: int = 1000, **kwargs):
       
        cons_params = torch.tensor([
                [1.20, 1.20, -0.25 * torch.pi, 2.0,  6.0,  0.1],
                [2.25, 2.25, -0.25 * torch.pi, 2.5, 12.0, 0.1],
                [3.50, 3.50, -0.25 * torch.pi, 2.5, 10.0, 0.1],
            ])

        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num, cons_params = cons_params,**kwargs)

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - torch.sqrt(t)], dim=1) + 0.7057
        theta = torch.tensor(-0.25 * torch.pi, device=self.device)
        while True:
            c1 = (
                0.1
                - (((pf[:, 0] - 1.2) * torch.cos(theta) - (pf[:, 1] - 1.2) * torch.sin(theta)) ** 2) / 4
                - (((pf[:, 0] - 1.2) * torch.sin(theta) + (pf[:, 1] - 1.2) * torch.cos(theta)) ** 2) / 36
            )
            invalid = c1> 0
            if not invalid.any():
                break
            pf[invalid] = (pf[invalid] - 0.7057) * 1.001 + 0.7057
        return pf


class LIRCMOP8(LIRCMOP6):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 3, ref_num: int = 1000, **kwargs):

        cons_params = torch.tensor([
                [1.20, 1.20, -0.25 * torch.pi, 2.0,  6.0,  0.1],
                [2.25, 2.25, -0.25 * torch.pi, 2.5, 12.0, 0.1],
                [3.50, 3.50, -0.25 * torch.pi, 2.5, 10.0, 0.1],
            ])
   
        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num, cons_params=cons_params,**kwargs)
    
    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t**2], dim=1) + 0.7057
        theta = torch.tensor(-0.25 * torch.pi, device=self.device)
        while True:
            c1 = (
                0.1
                - (((pf[:, 0] - 1.2) * torch.cos(theta) - (pf[:, 1] - 1.2) * torch.sin(theta)) ** 2) / 4
                - (((pf[:, 0] - 1.2) * torch.sin(theta) + (pf[:, 1] - 1.2) * torch.cos(theta)) ** 2) / 36
            )
            invalid = c1> 0
            if not invalid.any():
                break
            pf[invalid] = (pf[invalid] - 0.7057) * 1.001 + 0.7057
        return pf

   

class LIRCMOP9(LIRCMOP5):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 2, ref_num: int = 1000, 
                 c1_params: Optional[torch.Tensor] = None,
                 c2_params: Optional[torch.Tensor] = None,
                 **kwargs):

        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num,**kwargs)

        if c1_params is None:
            c1_params = torch.tensor([1.4, 1.4, -0.25 * torch.pi, 1.5, 6.0, 0.1])  # p, q, theta, a, b, r
        if c2_params is None:
            c2_params = torch.tensor([0.25 * torch.pi, 2.0]) # alpha, c
        self.c1_params = c1_params
        self.c2_params = c2_params

    def f1(self, X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.7057 * X[:, 0:1] * (10 * g + 1)

    def f2(self,X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.7057 * (1 - X[:, 0:1] ** 2) * (10 * g + 1)

    # This rotated ellipse constraint is same as LIRCMOP5's c, but simplified to avoid repeated unsqueeze
    def c1(self, f1: torch.Tensor, f2: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        params = params.to(device=f1.device, dtype=f1.dtype)
        p, q, theta, a, b, r = params.unbind(dim=0)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        inv_a2, inv_b2 = 1.0 / (a * a), 1.0 / (b * b)
        dx = f1 - p
        dy = f2 - q
        h1 = (dx * cos_t - dy * sin_t) ** 2 * inv_a2
        h2 = (dx * sin_t + dy * cos_t) ** 2 * inv_b2
        return r - h1 - h2 #[N,1]

    # Sine constraint
    def c2(self, f1: torch.Tensor, f2: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        params = params.to(device=f1.device, dtype=f1.dtype)
        alpha, c = params.unbind(dim=0)
        s = torch.sin(alpha)
        co = torch.cos(alpha)
        h1 = f1 * s + f2 * co
        h2 = torch.sin(4.0 * torch.pi * (f1 * co - f2 * s))
        return c - h1 + h2

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c1(f1, f2, self.c1_params)
        c2 = self.c2(f1, f2, self.c2_params)
        return torch.cat([f1, f2, c1, c2], dim=1)
    
    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t.square()], dim=1) * 1.7057  # [M, 2]

        G1 = self.c1(pf[:, [0]], pf[:, [1]], self.c1_params)
        G2 = self.c2(pf[:, [0]], pf[:, [1]], self.c2_params)

        mask = torch.logical_and(G1 <= 0, G2 <= 0).squeeze(1)
        pf = pf[mask]

        endpoints = torch.tensor([[0.0, 2.182], [1.856, 0.0]],
                                dtype=pf.dtype, device=pf.device)
        pf = torch.cat([pf, endpoints], dim=0)
        return pf


class LIRCMOP10(LIRCMOP9):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 2, ref_num: int = 1000, 
                 c1_params: Optional[torch.Tensor] = None,
                 c2_params: Optional[torch.Tensor] = None,
                 **kwargs):

        if c1_params is None:
            c1_params = torch.tensor([1.1, 1.2, -0.25 * torch.pi, 2.0, 4.0, 0.1])  # p, q, theta, a, b, r
        if c2_params is None:
            c2_params = torch.tensor([0.25 * torch.pi, 1.0]) # alpha, c
        self.c1_params = c1_params
        self.c2_params = c2_params

        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num,c1_params=c1_params,c2_params=c2_params,**kwargs)

        
    def f2(self, X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.7057 * (1 - torch.sqrt(X[:, 0:1])) * (10 * g + 1)
    
    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - torch.sqrt(t)], dim=1) * 1.7057  # [M, 2]

        G1 = self.c1(pf[:, [0]], pf[:, [1]], self.c1_params)
        G2 = self.c2(pf[:, [0]], pf[:, [1]], self.c2_params)

        mask = torch.logical_and(G1 <= 0, G2 <= 0).squeeze(1)
        pf = pf[mask]

        endpoints = torch.tensor([[1.747,0]],dtype=pf.dtype, device=pf.device)
        pf = torch.cat([pf, endpoints], dim=0)
        return pf

class LIRCMOP11(LIRCMOP10):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 2, ref_num: int = 1000, 
                 c1_params: Optional[torch.Tensor] = None,
                 c2_params: Optional[torch.Tensor] = None,
                 **kwargs):
        if c1_params is None:
            c1_params = torch.tensor([1.2, 1.2, -0.25 * torch.pi, 1.5, 5.0, 0.1])  # p, q, theta, a, b, r
        if c2_params is None:
            c2_params = torch.tensor([0.25 * torch.pi,2.1]) # alpha, c

        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num,c1_params=c1_params,c2_params=c2_params,**kwargs)

    def pf(self):
        return torch.tensor([
                            [1.3965, 0.1591],
                            [1.0430, 0.5127],
                            [0.6894, 0.8662],
                            [0.3359, 1.2198],
                            [0.0106, 1.6016],
                            [0.0000, 2.1910],
                            [1.8730, 0.0000],
                        ], dtype=torch.float, device=self.device)
    
class LIRCMOP12(LIRCMOP9):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 2, ref_num: int = 1000, 
                 c1_params: Optional[torch.Tensor] = None,
                 c2_params: Optional[torch.Tensor] = None,
                 **kwargs):
        
        if c1_params is None:
            c1_params = torch.tensor([1.6, 1.6, -0.25 * torch.pi, 1.5, 6.0, 0.1])  # p, q, theta, a, b, r
        if c2_params is None:
            c2_params = torch.tensor([0.25 * torch.pi,2.5]) # alpha, c

        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num,c1_params=c1_params,c2_params=c2_params,**kwargs)

    def pf(self):
        return torch.tensor([
                            [1.6794, 0.4419],
                            [1.3258, 0.7955],
                            [0.9723, 1.1490],
                            [2.0320, 0.0990],
                            [0.6187, 1.5026],
                            [0.2652, 1.8562],
                            [0.0000, 2.2580],
                            [2.5690, 0.0000],
                        ], dtype=torch.float32, device=self.device)

class LIRCMOP13(LIRCMOP):
    def __init__(self, d: int = 30, m: int = 3, n_iq: int = 2, ref_num: int = 1000, 
                 cons_params: Optional[torch.Tensor] = None,
                 **kwargs):
        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num, **kwargs)
        
        if cons_params is None:
            cons_params = torch.tensor([[9.00, 4.00], [3.61, 3.24]])
        self.cons_params = cons_params

    def g(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sum(10.0 * (X[:, 2:] - 0.5).square(), dim=1, keepdim=True)
     
    def f123(self, X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        K  = 1.7057 + g                       
        a0 = 0.5 * torch.pi * X[:, 0:1]       
        a1 = 0.5 * torch.pi * X[:, 1:2]    
        c0, s0 = torch.cos(a0), torch.sin(a0)
        c1, s1 = torch.cos(a1), torch.sin(a1)
        f1 = K * c0 * c1
        f2 = K * c0 * s1
        f3 = K * s0
        return f1, f2, f3                           

    def c(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        params = params.to(device=f1.device, dtype=f1.dtype)
        a, b = params.unbind(dim=1) 
        h = f1 ** 2 + f2 ** 2 + f3 ** 2
        return -(h - a.unsqueeze(0)) * (h - b.unsqueeze(0))  # [N,K]
    
    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g(X)
        f1, f2, f3 = self.f123(X,g)
        G = self.c(f1, f2, f3,self.cons_params)
        return torch.cat([f1, f2, f3, G],dim=1)
    
    def pf(self):
        sample,_= uniform_sampling(self.ref_num * self.m, self.m) 
        sample = sample.to(self.device)
        pf = 1.7057 * sample / torch.sqrt((sample ** 2).sum(dim=1, keepdim=True))
        return pf

class LIRCMOP14(LIRCMOP13):
    def __init__(self, d: int = 30, m: int = 2, n_iq: int = 2, ref_num: int = 1000, **kwargs):

        cons_params = torch.tensor([[9.00,4.00],[3.61,3.24],[3.0625,2.56]])  

        super().__init__(d=d, m=m, n_iq=n_iq, ref_num=ref_num,cons_params=cons_params,**kwargs)
    
    def pf(self):
        sample,_= uniform_sampling(self.ref_num * self.m, self.m) 
        sample = sample.to(self.device)
        pf = torch.sqrt(torch.tensor(3.0625,device=self.device)) * sample / torch.sqrt((sample ** 2).sum(dim=1, keepdim=True))
        return pf