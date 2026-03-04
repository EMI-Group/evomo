__all__ = ['DASCMOP1', 'DASCMOP2', 'DASCMOP3', 'DASCMOP4', 'DASCMOP5',
           'DASCMOP6', 'DASCMOP7', 'DASCMOP8', 'DASCMOP9']

import torch
from .utils import CMOP
from evox.operators.sampling import uniform_sampling
from typing import Optional


class DASCMOP(CMOP):
    """
    The DASCMOP test suite generator of constrained multibjecitve problems DASCMOP1-DASCMOP9.

    Z. Fan, W. Li, X. Cai, H. Li, C. Wei, Q. Zhang, K. Deb, and E. Goodman, 
    Difficulty adjustable and scalable constrained multi-objective test problem toolkit, 
    Evolutionary Computation, 2020, 28(3): 339-378.
  
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
                 difficulty: torch.Tensor=torch.tensor([0, 0.5, 0.5]), 
                 **kwargs):
        self.device = kwargs.pop("device", torch.get_default_device())
        lb = torch.zeros(d, device=self.device)
        ub = torch.ones(d, device=self.device)
        
        super().__init__(d=d,m=m,n_iq=n_iq,n_eq=0,lb=lb,ub=ub,**kwargs)
        self.device = kwargs.pop("device", torch.get_default_device())
        self.ref_num = ref_num
        self.m=m
       
        difficulty = difficulty.to(self.device)
        self.eta, self.zeta, self.gamma = difficulty

    # Both paper and PlatEMO use X instead of X[:, self.m - 1:]
    def g1(self, X: torch.Tensor) -> torch.Tensor:
        contrib = (X[:, self.m - 1:] - torch.sin(0.5 * torch.pi * X[:, 0:1])) ** 2
        return contrib.sum(dim=1, keepdim=True)

    def g2(self, X: torch.Tensor) -> torch.Tensor:
        z = X[:, self.m - 1:] - 0.5
        contrib = z ** 2 - torch.cos(20 * torch.pi * z)
        return (self.d - self.m + 1) + contrib.sum(dim=1, keepdim=True)

    def g3(self, X: torch.Tensor) -> torch.Tensor:
        j = torch.arange(self.m, self.d + 1, device=X.device, dtype=X.dtype).unsqueeze(0)
        contrib = (X[:, self.m - 1:] - torch.cos(0.25 * torch.pi * j / self.d * (X[:, 0:1] + X[:, 1:2])))**2
        return contrib.sum(dim=1, keepdim=True)


class DASCMOP1(DASCMOP):
    def __init__(self, d=30, m=2,n_iq=11,ref_num=1000,difficulty=torch.tensor([0, 0.5, 0.5]),**kwargs):
        super().__init__(d=d, m=m, n_iq=n_iq,ref_num=ref_num, difficulty=difficulty,**kwargs)

    def constraints(self, X: torch.Tensor, g: torch.Tensor, f0:Optional[torch.Tensor]=None, f1:Optional[torch.Tensor]=None):
        a = 20.0
        b = 2.0 * self.eta - 1.0
        d = torch.where(self.zeta == 0, torch.zeros_like(self.zeta), torch.full_like(self.zeta, 0.5))
        e = torch.where(self.zeta > 0, d - torch.log(self.zeta),     torch.full_like(self.zeta, 1e30))

        r = 0.5 * self.gamma
        p_k = torch.tensor([[0., 1.0, 0., 1.0, 2.0, 0., 1.0, 2.0, 3.0]], dtype=X.dtype, device=X.device)
        q_k = torch.tensor([[1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5]], dtype=X.dtype, device=X.device)
        a_k2 = 0.3
        b_k2 = 1.2
        theta_k = torch.tensor(-0.25 * torch.pi, dtype=X.dtype, device=X.device)

        c = torch.zeros((X.size(0), 2 + p_k.size(1)), dtype=X.dtype, device=X.device)

        c[:, 0:1] = torch.sin(a * torch.pi * X) - b
        c[:, 1:2] = torch.where(self.zeta == 1.0, 1e-4 - torch.abs(e - g),(e - g) * (g - d))

        if f0 is None:
            f0=X+g
        if f1 is None:
            f1=1-X**2+g
        c[:, 2:] = (((f0 - p_k) * torch.cos(theta_k) - (f1 - q_k) * torch.sin(theta_k)) ** 2 / a_k2
                    + ((f0 - p_k) * torch.sin(theta_k) + (f1 - q_k) * torch.cos(theta_k)) ** 2 / b_k2
                    - r)
        return -1 * c
    
    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - X[:, 0:1] ** 2 + g
        c = self.constraints(X[:, 0:1], g, f0, f1)
        return torch.cat([f0, f1, c],dim=1)
    
    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t**2], dim=1) + 0.5
        x1 = (torch.sqrt(1.0 - 4.0 * (-pf[:, 0:1] + pf[:, 1:2] - 1.0)) - 1.0) / 2.0
        sum1 = pf[:,0:1] - x1
        C = self.constraints(x1, sum1) 
        mask = torch.any(C > 0, dim=1)
        pf = pf[~mask]
        endpoint = torch.tensor([[1.5, 0.5]], dtype=pf.dtype, device=pf.device)
        pf = torch.cat([pf, endpoint], dim=0)
        return pf
    # def _calc_pareto_front(self, *args, **kwargs):
    #     fname = f"{self.name.lower()}_M{self.m}_D{self.d}.pf"
    #     return load_pareto_front_from_file(os.path.join("DASCMOP", fname))

class DASCMOP2(DASCMOP1):
    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - torch.sqrt(X[:, 0:1]) + g
        c = self.constraints(X[:, 0:1], g, f0, f1)
        return torch.cat([f0, f1, c],dim=1)
    
    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - torch.sqrt(t)], dim=1) + 0.5
        theta = torch.tensor(-0.25 * torch.pi, device=self.device)
        while True:
            C = (
                0.25
                - (((pf[:, 0] - 1.0) * torch.cos(theta) - (pf[:, 1] - 0.5) * torch.sin(theta)) ** 2) / 0.3
                - (((pf[:, 0] - 1.0) * torch.sin(theta) + (pf[:, 1] - 0.5) * torch.cos(theta)) ** 2) / 1.2
            )
            invalid = C > 0  
            if not invalid.any():
                break
            pf[invalid] = (pf[invalid] - 0.5) * 1.001 + 0.5
        return pf
    

class DASCMOP3(DASCMOP1):
    def __init__(self,difficulty=torch.tensor([0.5, 0.5, 0.5]),**kwargs):
        super().__init__(difficulty=difficulty,**kwargs)
    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g1(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - torch.sqrt(X[:, 0:1]) + 0.5 * torch.abs(torch.sin(5 * torch.pi * X[:, 0:1])) + g
        c = self.constraints(X[:, 0:1], g, f0, f1)
        return torch.cat([f0, f1, c],dim=1)

    def pf(self):
        return torch.tensor([[0.5000,1.5000],[0.5010,1.4762],[0.5020,1.4710],[0.5030,1.4688],[0.5040,1.4681],
                            [0.6502,1.4652],[0.7002,1.0541],[0.9044,0.8986],[1.1066,0.7729],[1.3008,0.6114],
                            [1.5000,0.5000],[0.9069,0.8951],[1.1126,0.7727],[0.9129,0.8950],[1.1151,0.7690],
                            [0.9153,0.8914],[1.1175,0.7653],[1.1200,0.7616],[0.9213,0.8913],[1.1260,0.7613],
                            [1.1285,0.7576]])

class DASCMOP4(DASCMOP1):
    def __init__(self,difficulty=torch.tensor([0.5, 0.5, 0.5]),**kwargs):
        super().__init__(difficulty=difficulty,**kwargs)
    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - X[:, 0:1] ** 2 + g
        c = self.constraints(X[:, 0:1], g, f0, f1)
        return torch.cat([f0, f1, c],dim=1)
    
    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)
        pf = torch.cat([t, 1.0 - t**2], dim=1) + 0.5
        X1 = (torch.sqrt(1 - 4 * (-pf[:, 0:1] + pf[:, 1:2] - 1)) - 1) / 2
        sum1 = pf[:, 0:1] - X1
        C = self.constraints(X1, sum1)  
        mask = torch.any(C > 0, dim=1)
        pf = pf[~mask]
        extra_point = torch.tensor([[1.5, 0.5]], device=self.device)
        pf = torch.cat([pf, extra_point], dim=0)
        return pf


class DASCMOP5(DASCMOP1):
    def __init__(self,difficulty=torch.tensor([0.5, 0.5, 0.5]),**kwargs):
        super().__init__(difficulty=difficulty,**kwargs)
    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - torch.sqrt(X[:, 0:1]) + g
        c = self.constraints(X[:, 0:1], g, f0, f1)
        return torch.cat([f0, f1, c],dim=1)
    
    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device).unsqueeze(1)  # (N,1)
        pf = torch.cat([t, 1.0 - torch.sqrt(t)], dim=1) + 0.5  # (N,2)
        sin_mask = torch.sin(20 * torch.pi * pf[:, 0]) >= -1e-10  
        pf = pf[sin_mask]
        theta = torch.tensor(-0.25 * torch.pi, device=self.device)
        while True:
            C = (
                0.25
                - (((pf[:, 0] - 1.0) * torch.cos(theta) - (pf[:, 1] - 0.5) * torch.sin(theta)) ** 2) / 0.3
                - (((pf[:, 0] - 1.0) * torch.sin(theta) + (pf[:, 1] - 0.5) * torch.cos(theta)) ** 2) / 1.2
            )
            invalid = C > 0  
            if not invalid.any():
                break
            pf[invalid] = (pf[invalid] - 0.5) * 1.001 + 0.5
        return pf


class DASCMOP6(DASCMOP1):
    def __init__(self,difficulty=torch.tensor([0.5, 0.5, 0.5]),**kwargs):
        super().__init__(difficulty=difficulty,**kwargs)
    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)
        f0 = X[:, 0:1] + g
        f1 = 1.0 - torch.sqrt(X[:, 0:1]) + 0.5 * torch.abs(torch.sin(5 * torch.pi * X[:, 0:1])) + g
        c = self.constraints(X[:, 0:1], g, f0, f1)
        return torch.cat([f0, f1, c],dim=1)
    
    def pf(self):
        return torch.tensor(
            [[0.5000,1.5000],[0.5010,1.4762],[0.5020,1.4710],[0.5030,1.4688],[0.5040,1.4681],
            [0.6502,1.4652],[0.7002,1.0541],[0.9044,0.8986],[1.1066,0.7729],[1.3008,0.6114],
            [1.5000,0.5000],[0.9069,0.8951],[1.1126,0.7727],[0.9129,0.8950],[1.1151,0.7690],
            [0.9153,0.8914],[1.1175,0.7653],[1.1200,0.7616],[0.9213,0.8913],[1.1260,0.7613],
            [1.1285,0.7576]])

class DASCMOP7(DASCMOP):
    def __init__(self,d=30, m=3,n_iq=7, ref_num=1000, difficulty=torch.tensor([0.5, 0.5, 0.5]), **kwargs):
        super().__init__(d=d, m=m, n_iq=n_iq,ref_num=ref_num, difficulty=difficulty,**kwargs)

    def constraints(self, X: torch.Tensor, g: torch.Tensor,
                f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        a = 20.0
        b = 2.0 * self.eta - 1.0
        d = 0.5 if self.zeta != 0 else 0.0
        if self.zeta > 0:
            e = d - torch.log(self.zeta)
        else:
            e = 1e30
        r = 0.5 * self.gamma
  
        inv_sqrt3 = 1.0 / (3.0 ** 0.5)
        x_k = torch.tensor([[1.0, 0.0, 0.0, inv_sqrt3]], dtype=X.dtype, device=X.device)
        y_k = torch.tensor([[0.0, 1.0, 0.0, inv_sqrt3]], dtype=X.dtype, device=X.device)
        z_k = torch.tensor([[0.0, 0.0, 1.0, inv_sqrt3]], dtype=X.dtype, device=X.device)

        c = torch.zeros((X.size(0), 3 + x_k.size(1)), dtype=X.dtype, device=X.device)

        c[:, 0:1] = torch.sin(a * torch.pi * X[:, 0:1]) - b
        c[:, 1:2] = torch.cos(a * torch.pi * X[:, 1:2]) - b
   
        if self.zeta == 1.0:
            c[:, 2:3] = 1e-4 - torch.abs(e - g)
        else:
            c[:, 2:3] = (e - g) * (g - d)

        c[:, 3:] = (f0 - x_k) ** 2 + (f1 - y_k) ** 2 + (f2 - z_k) ** 2 - r ** 2
        return -1 * c

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)
        f0 = X[:, 0:1] * X[:, 1:2] + g
        f1 = X[:, 1:2] * (1.0 - X[:, 0:1]) + g
        f2 = 1 - X[:, 1:2] + g
        c = self.constraints(X, g,f0, f1, f2)
        return torch.cat([f0, f1, f2, c],dim=1)
    
    def pf(self):
        pf,_= uniform_sampling(self.ref_num * self.m, self.m) 
        pf = pf.to(self.device)
        eps = 1e-12
        X1 = 1.0 / (1.0 + pf[:, 1] / (pf[:, 0] + eps))
        X2 = pf[:, 0] / (X1 + eps)
        C1 = -torch.sin(20.0 * torch.pi * X1)
        C2 = -torch.cos(20.0 * torch.pi * X2)
        mask = torch.logical_or(C1 > 1e-2, C2 > 1e-2)
        pf = pf[~mask]
        return pf+0.5

class DASCMOP8(DASCMOP7):
    def objectives(self, X: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f0 = torch.cos(0.5 * torch.pi * X[:, 0:1]) * torch.cos(0.5 * torch.pi * X[:, 1:2]) + g
        f1 = torch.cos(0.5 * torch.pi * X[:, 0:1]) * torch.sin(0.5 * torch.pi * X[:, 1:2]) + g
        f2 = torch.sin(0.5 * torch.pi * X[:, 0:1]) + g
        return torch.cat([f0, f1, f2], dim=1)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g2(X)
        F = self.objectives(X, g)
        c = self.constraints(X, g, F[:, 0:1], F[:, 1:2], F[:, 2:3])
        return torch.cat([F, c],dim=1)
    
    def pf(self):
        pf, _ = uniform_sampling(self.ref_num, 3)
        pf = pf.to(self.device)
        pf = pf / (torch.sqrt(torch.sum(pf**2, dim=1, keepdim=True)) + 1e-12)
        eps = 1e-12
        X2 = torch.atan(pf[:, 1] / (pf[:, 0] + eps)) / (0.5 * torch.pi)
        X1 = torch.acos(pf[:, 0] / (torch.cos(0.5 * torch.pi * X2) + eps)) / (0.5 * torch.pi)
        C1 = -torch.sin(20.0 * torch.pi * X1)
        C2 = -torch.cos(20.0 * torch.pi * X2)
        mask = torch.logical_or(C1 > 1e-2, C2 > 1e-2)
        pf = pf[~mask]

        return pf+0.5

class DASCMOP9(DASCMOP8):
    def fn(self, X: torch.Tensor) -> torch.Tensor:
        g = self.g3(X)
        F = self.objectives(X, g)
        c = self.constraints(X, g, F[:, 0:1], F[:, 1:2], F[:, 2:3])
        return torch.cat([F, c],dim=1)


