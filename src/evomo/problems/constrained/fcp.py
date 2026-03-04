
__all__ = ['FCP1', 'FCP2', 'FCP3', 'FCP4', 'FCP5']

import torch
from evomo.utils import get_pareto_front
from evox.operators.sampling import uniform_sampling
from evomo.problems.constrained.base import CMOP

class FCP(CMOP):
    """
    J. Yuan, H. Liu, Y. Ong, and Z. He, Indicator-based evolutionary
    algorithm for solving constrained multi-objective optimization problems,
    IEEE Transactions on Evolutionary Computation, 2022, 26(2): 379-391.
    """
    def __init__(self, d: int = None, ref_num: int = 1000,**kwargs):
        super().__init__(d=d,m=2,n_iq=1,n_eq=0,lb=torch.zeros(d),ub=torch.ones(d),**kwargs)
        
        self.ref_num = ref_num

    def g1(self, X: torch.Tensor) -> torch.Tensor:
        return 1.0 + 9.0 * torch.mean(X[:, 1:], dim=1, keepdim=True)
    
    def c1(self, g: torch.Tensor) -> torch.Tensor:
        Dis = torch.abs(9.0 - g)
        y1 = Dis ** 2 - 0.25
        y2 = (1.2 + torch.sin(Dis * torch.pi)) / (Dis + 1e-6)
        c = torch.min(torch.cat([y1, y2], dim=1), dim=1, keepdim=True).values
        return c
    
    def c2(self, x1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        c1 = torch.log(torch.sqrt((10.0 * x1 - 9.0) ** 2 + (g - 3.0) ** 2) + 0.5)
        c2 = torch.log(torch.sqrt((10.0 * x1 - 6.0) ** 2 + (g - 6.0) ** 2) + 0.05)
        c3 = (10.0 * x1 - 2**0.5) ** 2 + (g - 10.0) ** 2 - 2.0
        c4 = 1.2 + torch.sin(torch.pi * torch.sqrt(c3 + 2.0))
        c  = torch.min(torch.cat([c1, c2, c3, c4], dim=1), dim=1, keepdim=True).values
        return c
    
class FCP1(FCP):
    def __init__(self, d: int = 30, ref_num: int = 1000,**kwargs):
        super().__init__(d=d,ref_num=ref_num,**kwargs)
    
    def fn(self,X: torch.Tensor) -> torch.Tensor:
        g=self.g1(X)
        f1 = X[:, [0]] * g
        f2 = (1.0 - X[:, [0]]) * g
        c = self.c1(g)
        return torch.cat([f1, f2, c], dim=1)

    def pf(self):
        pf,_= uniform_sampling(self.ref_num * self.m, self.m) 
        pf=pf.to(self.device)
        return 8.5*pf
    
class FCP2(FCP):
    def __init__(self, d: int = 30, ref_num: int = 1000,**kwargs):
        super().__init__(d=d,ref_num=ref_num,**kwargs)

    def fn(self,X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        x1 = X[:, [0]]
        f1 = torch.cos(0.5 * torch.pi * x1) * g
        f2 = (torch.sin(0.5 * torch.pi * x1) + 0.2 * torch.sin(4.0 * torch.pi * x1)) * g
        c = self.c1(g)
        return torch.cat([f1, f2, c], dim=1)   # (N, 3)

    def pf(self) -> torch.Tensor:
        t  = torch.linspace(0.0, 1.0, steps=self.ref_num,device=self.device)
        f1 = torch.cos(0.5 * torch.pi * t)
        f2 = torch.sin(0.5 * torch.pi * t) + 0.2 * torch.sin(4.0 * torch.pi * t)
        pf = torch.column_stack([f1, f2])  # (K, 2)
        pf = get_pareto_front(pf)
        return 8.5 * pf


class FCP3(FCP):
    def __init__(self, d: int = 30, ref_num: int = 1000,**kwargs):
        super().__init__(d=d,ref_num=ref_num,**kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        x1 = X[:, [0]]
        t  = torch.remainder(torch.floor(100.0 * g), 2.0)
        g = g + t * (g - 9.0)**2
        f1 = torch.cos(0.5 * torch.pi * x1) * g
        f2 = torch.sin(0.5 * torch.pi * x1) * g
        c=self.c1(g)
        return torch.cat([f1, f2, c], dim=1)   # (N, 3)


    def pf(self):
        theta = 0.5 * torch.pi * torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device)
        pf = torch.column_stack([torch.cos(theta), torch.sin(theta)])  # (K,2)
        return 8.5 * pf


class FCP4(FCP):
    def __init__(self, d: int = 30, ref_num: int = 1000,**kwargs):
        super().__init__(d=d,ref_num=ref_num,**kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        x1 = X[:, [0]]
        t  = torch.remainder(torch.floor(100.0 * g), 2.0)
        g = g + t * (g - 9.0)**2
        f1 = (1-x1)*g
        f2 = (x1 + 0.2 * torch.sin(4.0 * torch.pi * x1)) * g
        c=self.c1(g)
        return torch.cat([f1, f2, c], dim=1)  


    def pf(self) -> torch.Tensor:
        t  = torch.linspace(0.0, 1.0, steps=self.ref_num,device=self.device)
        f1 = 1-t
        f2 = t + 0.2 * torch.sin(4.0 * torch.pi * t)
        pf = torch.column_stack([f1, f2])  # (K, 2)
        pf = get_pareto_front(pf)
        return 8.5 * pf

class FCP5(FCP):
    def __init__(self, d: int = 30, ref_num: int = 1000,**kwargs):
        super().__init__(d=d,ref_num=ref_num,**kwargs)

    def fn(self,X: torch.Tensor) -> torch.Tensor:
        g=self.g1(X)
        x1 = X[:, [0]]
        f1=x1*g
        f2=(1-x1)*g
        c=self.c2(x1,g)
        return torch.cat([f1, f2, c], dim=1)  

    def pf(self):
        t = 0.5 * torch.pi * torch.linspace( 0.0, 1.0, steps=self.ref_num, device=self.device)
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=self.device))

        c1x1 = 0.1 * torch.cat([9.0 + 0.5 * torch.cos(t), 9.0 - 0.5 * torch.cos(t)], dim=0)
        c2x1 = 0.1 * torch.cat([6.0 + 0.95 * torch.cos(t), 6.0 - 0.95 * torch.cos(t)], dim=0)
        c3x1 = 0.1 * torch.cat([sqrt2 + sqrt2 * torch.cos(t), sqrt2 - sqrt2 * torch.cos(t)], dim=0)

        c1g = torch.tile(3.0 - 0.5 * torch.sin(t), (2,))
        c2g = torch.tile(6.0 - 0.95 * torch.sin(t), (2,))
        c3g = torch.tile(10.0 - sqrt2 * torch.sin(t), (2,))

        x1 = torch.cat([c1x1, c2x1, c3x1], dim=0)  
        g  = torch.cat([c1g,  c2g,  c3g ], dim=0)  
        pf  = torch.stack([x1 * g, (1.0 - x1) * g], dim=1) 
        pf = get_pareto_front(pf)
        return pf
