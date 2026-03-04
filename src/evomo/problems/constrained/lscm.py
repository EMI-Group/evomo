
__all__ = ['LSCM1', 'LSCM2', 'LSCM3', 'LSCM4', 'LSCM5', 'LSCM6', 'LSCM7', 'LSCM8', 'LSCM9', 'LSCM10', 'LSCM11', 'LSCM12']

from typing import List, Tuple, Dict, Optional
import torch
from evomo.utils import get_pareto_front
from evox.operators.sampling import uniform_sampling, grid_sampling
from evox.utils import clamp
from .sdc import CEC_2006_information, CEC_2006_fitness, Distance_function
from evomo.problems.constrained.base import CMOP


def duan_yueshu(index: int) -> Tuple[list, list, int, int, list, int, int]:
    """
    Returns:
      nk:           list, number of variables in each segment
      c_index:      list, indexes of CEC_problem used for each segment (mapped)
      CCT:          int, 1=linear, 2=nonlinear (cos)
      DCT:          int, 1=linear, 2=nonlinear (cos)
      Dis_function: list, two distance function IDs [a, b] (used alternately per segment)
      PF_form:      int, 1=linear, 2=convex, 3=disconnected
      Q_form:       int, 1=single, 2=multiple, 3=full
    """
    CEC_problem = [1, 2, 3, 6, 9, 10, 11, 12, 14, 18, 24]

    match index:
        case 1:
            nk = [5,4,6,6,3,5,5,5,5,2]
            c_index = [2]*10
            CCT, DCT = 1, 2
            Dis_function = [1, 2]
            PF_form, Q_form = 1, 1
        case 2:
            nk = [5,4,6,6,3,5,5,5,5,2]
            c_index = [4]*10
            CCT, DCT = 1, 1
            Dis_function = [1, 1]
            PF_form, Q_form = 3, 1
        case 3:
            nk = [5,4,6,6,3,5,5,5,5,2]
            c_index = [10]*10
            CCT, DCT = 1, 1
            Dis_function = [1, 3]
            PF_form, Q_form = 2, 2
        case 4:
            nk = [5,4,3,5,4,3,5,4,3,5]
            c_index = [4,11,4,11,4,11,4,11,4,11]
            CCT, DCT = 2, 2
            Dis_function = [4, 3]
            PF_form, Q_form = 3, 1
        case 5:
            nk = [5,4,3,5,4,3,5,4,3,5]
            c_index = [2,1,2,1,2,1,2,1,2,1]
            CCT, DCT = 1, 1
            Dis_function = [2, 3]
            PF_form, Q_form = 1, 3
        case 6:
            nk = [5,3,5,4,3,5,4,3,5,2]
            c_index = [7,8,7,8,7,8,7,8,7,8]
            CCT, DCT = 2, 1
            Dis_function = [1, 5]
            PF_form, Q_form = 1, 1
        case 7:
            nk = [3,6,5,6,3,6,5,6,3,6]
            c_index = [3,11,4,3,11,4,3,11,4,3]
            CCT, DCT = 1, 1
            Dis_function = [4, 5]
            PF_form, Q_form = 1, 2
        case 8:
            nk = [4,4,5,6,3,5,5,5,5,2]
            c_index = [10,11,9,10,11,9,10,11,9,10]
            CCT, DCT = 2, 2
            Dis_function = [1, 1]
            PF_form, Q_form = 2, 1
        case 9:
            nk = [4,5,4,5,4,5,4,5,4,5]
            c_index = [4,7,11,4,7,11,4,7,11,4]
            CCT, DCT = 1, 1
            Dis_function = [1, 4]
            PF_form, Q_form = 2, 2
        case 10:
            nk = [4,5,4,5,4,5,4,5,4,5]
            c_index = [3]*10
            CCT, DCT = 2, 2
            Dis_function = [1, 2]
            PF_form, Q_form = 1, 1
        case 11:
            nk = [2,3,2,3,2,3,2,3,2,3]
            c_index = [4,6,4,6,4,6,4,6,4,6]
            CCT, DCT = 1, 2
            Dis_function = [2, 3]
            PF_form, Q_form = 2, 2
        case 12:
            nk = [11,4,3,11,4,3,11,4,3,11]
            c_index = [4,5,6,4,4,5,6,4,4,5]
            CCT, DCT = 1, 1
            Dis_function = [1, 1]
            PF_form, Q_form = 3, 1
        case _:
            raise ValueError(f"Unknown LSCM index: {index}")

    c_index = [CEC_problem[i - 1] for i in c_index]
    return nk, c_index, CCT, DCT, Dis_function, PF_form, Q_form


import torch

import torch

def PF_function(PopDec: torch.Tensor, G: torch.Tensor, PF_form: int, Q_form: int) -> torch.Tensor:
    N, M = PopDec.shape
    one  = PopDec.new_ones((N, 1))
    zero = PopDec.new_zeros((N, 1))
    X    = PopDec[:, :M-1]
    Xr   = torch.flip(X, [1])  # Reversed decision variables, corresponding to MATLAB's PopDec(:,M-1:-1:1)

    if PF_form == 1:
        C = torch.flip(torch.cumprod(torch.cat([one, X], 1), 1), [1])
        B = torch.cat([one, 1.0 - Xr], 1)
        if   Q_form == 1: coeff = 1.0 + G
        elif Q_form == 2: coeff = 1.0 + G + torch.cat([G[:, 1:], zero], 1)
        elif Q_form == 3: coeff = 1.0 + G.sum(1, keepdim=True)
        return coeff * C * B

    if PF_form == 2:
        C = torch.flip(torch.cumprod(torch.cat([one, torch.cos(X * (0.5 * torch.pi))], 1), 1), [1])
        B = torch.cat([one, torch.sin(Xr * (0.5 * torch.pi))], 1)
        if   Q_form == 1: coeff = 1.0 + G
        elif Q_form == 2: coeff = 1.0 + G + torch.cat([G[:, 1:], zero], 1)
        elif Q_form == 3: coeff = 1.0 + G.sum(1, keepdim=True)
        return coeff * C * B

    if PF_form == 3 and Q_form == 1:
        Obj = PopDec.new_empty((N, M))
        Obj[:, :M-1] = X
        g   = 2.0 + G[:, -1:].contiguous()        
        Obj[:, -1:] = g * (
            M - ((X /g) * (1.0 + torch.sin(3.0 * torch.pi * X))).sum(1, keepdim=True)
        )
        return Obj


class LSCM(CMOP):
    """
    K. Qiao, J. Liang, K. Yu, W. Guo, C. Yue, B. Qu, and P. N. Suganthan.
    Benchmark problems for large-scale constrained multi-objective
    optimization with baseline results. Swarm and Evolutionary Computation,
    2024, 86: 101504.

    ku_index = 1: index of LSCM function
    base: the dimension of each segment
    duan: the number of segments
    DCT: the form of unconstraint variable linkage
    CCT: the form of constraint variable linkage
    Dis_function: index of distance functions
    Q_form: index of Q functions
    PF_form: index of PF functions
    CEC_Problem: index of constraint functions
    lu: upper and lower bounds of constraint functions
    high_D_C: dimensions of constraint functions
    aaa: used for constraint functions
    optimal_f: optimal objective values of constraint functions
    
    nk: Number of subcomponents in each subset
    sublen:	Number of variables in each subset
    len: Cumulative sum of lengths of variable groups
    
    CV: store the index of constaint variables
    Dis: store the index of distance function variables
    
    h:  objective function value of constraint function
    HC: constraint function value of constraint function
    """

    def __init__(self, ku_index: int, d: int = 100, m: int = 2, ref_num: int = 1000,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs):
        if device is None:
            self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if dtype is None:
            self.dtype = torch.get_default_dtype()
        else:
            self.dtype = dtype

        self.m = m
        self.ref_num = ref_num

        self.base = 100 if d == 100 else ((d + 999) // 1000) * 100
        self.d    = ((d + self.base // 2) // self.base) * self.base  
        self.duan = self.d // self.base

        lb = torch.zeros(self.d, device=self.device, dtype=self.dtype)
        ub = torch.ones (self.d, device=self.device, dtype=self.dtype)
        super().__init__(d=self.d, m=m, n_iq=1, n_eq=0,
                        lb=lb, ub=ub, device=self.device, dtype=self.dtype, **kwargs)

        # Read segment configuration
        a, b, self.CCT, self.DCT, self.Dis_function, self.PF_form, self.Q_form = duan_yueshu(ku_index)
        self.nk = a[:self.duan]
        self.CEC_Problem = b[:self.duan]

        # CEC_2006 Information (per segment)
        self.lu: List[torch.Tensor] = [None]*self.duan
        self.high_D_C: List[int]    = [0]*self.duan
        self.aaa: List[Optional[torch.Tensor]] = [None]*self.duan
        self.optimal_f: List[torch.Tensor] = [None]*self.duan
        for j in range(self.duan):
            self.lu[j], self.high_D_C[j], self.aaa[j], self.optimal_f[j] = CEC_2006_information(self.CEC_Problem[j], device=self.device, dtype=self.dtype)

        c = torch.empty((self.m,),device=self.device, dtype=self.dtype)
        c[0] = 3.8*0.1*(1.0-0.1)
        for i in range(1, self.m):
            c[i] = 3.8 * c[i-1] * (1.0 - c[i-1])

        # Variable index organization
        self.sublen: List[torch.Tensor] = [None]*self.duan
        self.len:    List[torch.Tensor] = [None]*self.duan
        self.CV:     List[torch.Tensor] = [None]*self.duan
        self.Dis:    List[Dict[str, torch.Tensor]] = [None]*self.duan
        self.variable_information(c)

    def variable_information(self, c: torch.Tensor):
        """
        Dis{i,j}{h},i indicates the index of obj.duan,
        j indicates the index of objectives,h indicates the index of nk
        """
        frac = c / c.sum()
        self.sublen, self.len, self.CV, self.Dis = [None]*self.duan, [None]*self.duan, [None]*self.duan, [None]*self.duan

        for j in range(self.duan):
            # sublen{j}: length is self.m
            budget_j = (self.base - (self.m if j == 0 else 0) - self.high_D_C[j])
            self.sublen[j] = torch.floor(frac * (budget_j / self.nk[j])).to(torch.long)

            # Start points accumulation
            self.len[j] = torch.cat([torch.zeros(1, device=c.device, dtype=torch.long),
                                    torch.cumsum(self.sublen[j] * self.nk[j], dim=0)], dim=0)

            # CV Index
            cv_start = (self.m if j == 0 else j * self.base)
            self.CV[j] = torch.arange(cv_start, cv_start + self.high_D_C[j], device=c.device, dtype=torch.long)
            cv_last = self.CV[j][-1]

            # Start and end of each sub-group for each objective
            temp = self.len[j][:-1] + cv_last + 1                             # (m,)
            nk = self.nk[j]
            i_idx = torch.arange(nk, device=c.device, dtype=torch.long).unsqueeze(0)   # (1, nk)
            start = temp.unsqueeze(1) + self.sublen[j].unsqueeze(1) * i_idx            # (m, nk)
            end   = start + self.sublen[j].unsqueeze(1)                                 # (m, nk)

            # Flatten indices (one-time tensorization)
            # Lmax = self.base
            # grid = torch.arange(Lmax, device=c.device, dtype=torch.long)               # (Lmax,)
            # lengths_flat = self.sublen[j].unsqueeze(1).expand(self.m, nk).reshape(-1)  # (m*nk,)
            # mask = grid.unsqueeze(0) < lengths_flat.unsqueeze(1)                       # (m*nk, Lmax)
            # start_flat = start.reshape(-1, 1)                                          # (m*nk, 1)
            # idx_block  = start_flat + grid.unsqueeze(0)                                # (m*nk, Lmax)
            # idx = idx_block[mask]                                                      # (sum(lengths),)

            self.Dis[j] = {"start": start, "end": end}

    def variable_linkage(self, X: torch.Tensor) -> torch.Tensor:

        device, dtype = X.device, X.dtype
        PPP = X.clone()

        for j in range(self.duan):
            # ---- 1) Current segment constraint variable indices & values ----
            CV_idx = self.CV[j]                         # (H_j,)
            P = X.index_select(1, CV_idx)              # (N, H_j)

            # ---- 2) Get last_P (last distance variable column of previous segment; first segment takes column 0) ----
            if j == 0:
                last_cols = torch.tensor([0], device=device, dtype=torch.long)  # (1,)
            else:
                prev_end_excl = self.Dis[j - 1]["end"][self.m - 1, self.nk[j - 1] - 1] # End position index of all subgroups in previous segment
                last_cols = prev_end_excl.add(-1).reshape(1)   
            last_P = PPP.index_select(1, last_cols)                             # (N,1)

            # ---- 3) CCT: In-segment CV [-1, 1] mapping ----
            PP = last_P + P
            a2 = torch.remainder(PP, 0.5)
            if self.CCT == 1:
                mapped = -2.0 * a2 + 1.0
            else:  # self.CCT == 2
                mapped = torch.cos(a2 * torch.pi)
            X[:, CV_idx] = mapped

            # ---- 4) control_D index range ----
            start0   = self.Dis[j]["start"][0, 0]
            end_excl = self.Dis[j]["end"][self.m - 1, self.nk[j] - 1]
            control_idx = torch.arange(start0, end_excl, device=device, dtype=torch.long)

            # ---- 5) DCT: linkage on control_D ----
            L = control_idx.shape[0]
            t = (torch.arange(1, L + 1, device=device, dtype=dtype) / L).unsqueeze(0)

            if self.DCT == 1:
                mult = 1.0 + t
            else:  # self.DCT == 2
                mult = 1.0 + torch.cos(0.5 * torch.pi * t)

            block = X.index_select(1, control_idx)
            X[:, control_idx] = mult * block - last_P

        return X


    def fn(self, X: torch.Tensor) -> torch.Tensor:
        # 1) Bound clamping + Variable linkage
        X = clamp(X, self.lb, self.ub)
        N = X.shape[0]
        PopDec = self.variable_linkage(X)

        # 2) Accumulators
        G = torch.zeros((N, self.m), device=X.device, dtype=X.dtype)
        Con = torch.zeros((N,), device=X.device, dtype=X.dtype)

        # 3) Segment accumulation (strictly consistent with MATLAB Evaluation)
        for mm in range(self.duan):
            # ---- CEC high-dimensional variable mapping ----
            P = PopDec.index_select(1, self.CV[mm])                                  # (N, H_mm)
            lower, upper = self.lu[mm][0], self.lu[mm][1]
            new_P = lower.unsqueeze(0) + (upper - lower).unsqueeze(0) * P

            # ---- CEC fitness and constraints ----
            h, hc = CEC_2006_fitness(new_P, self.CEC_Problem[mm], self.aaa[mm], self.optimal_f[mm])
            G = G + h.unsqueeze(1) / self.m
            Con = Con + hc

            # ---- Distance term: Odd/Even objectives use Dis_function[0]/[1] respectively ----
            nk_mm = self.nk[mm]
            start = self.Dis[mm]["start"]   # (m, nk_mm)
            end   = self.Dis[mm]["end"]     # (m, nk_mm)
 
            # Per goal (m is small), odd/even goals use Dis_function[0]/[1] respectively
            for i in range(self.m):
                dis_id = self.Dis_function[0] if (i % 2 == 0) else self.Dis_function[1]
                acc = torch.zeros((N,), device=X.device, dtype=X.dtype)
                for j in range(nk_mm):
                    idx  = torch.arange(start[i, j], end[i, j], device=X.device)      # (L_ij,)
                    block = PopDec.index_select(1, idx)                    # (N, L_ij)
                    acc = acc + Distance_function(block, dis_id).squeeze()           # (N,)
                G[:, i] = G[:, i] + acc / self.sublen[mm][i] /nk_mm

        # 4) Objectives
        F = PF_function(PopDec[:, :self.m], G, self.PF_form, self.Q_form)

        # 5) Return [F, Con]
        return torch.cat([F, Con.unsqueeze(1)], dim=1)

    def pf(self):
        if self.PF_form in (1, 2):
            pf, _ = uniform_sampling(self.ref_num * self.m, self.m)
            pf = pf.to(self.device, self.dtype)
            if self.PF_form == 2:
                pf = pf / torch.sqrt((pf**2).sum(dim=1, keepdim=True))
        else:
            interval = torch.tensor([0.0, 0.251412, 0.631627, 0.859401], device=self.device, dtype=self.dtype)
            a, b, c, d = interval  # [0, 0.251412, 0.631627, 0.859401]
            median = (b - a) / ((d - c) + (b - a))

            X, _ = grid_sampling(self.ref_num * self.m,self.m - 1)
            X = X.to(device=self.device)

            # Piecewise linear refraction to [a, b] U [c, d]
            mask = (X <= median)
            X1 = torch.empty_like(X)
            # Left segment: map to [a, b]
            X1[mask] = X[mask] * (b - a) / median + a
            # Right segment: map to [c, d]
            X1[~mask] = (X[~mask] - median) * (d - c) / (1.0 - median) + c

            # Last dimension: 2*(M - sum( X/2 .* (1+sin(3*pi*X)) , 2 ))
            last = 2.0 * (self.m - torch.sum(X1 * 0.5 * (1.0 + torch.sin(3.0 * torch.pi * X1)), dim=1, keepdim=True))
            pf = torch.cat([X1, last], dim=1)
        return pf


class LSCM1(LSCM):
    def __init__(self, d: int = 100, m: int = 2, ref_num:int=1000, **kwargs):
        super().__init__(ku_index=1, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM2(LSCM):
    def __init__(self, d: int = 100, m: int = 2, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=2, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM3(LSCM):
    def __init__(self, d: int = 100, m: int = 2, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=3, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM4(LSCM):
    def __init__(self, d: int = 100, m: int = 2, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=4, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM5(LSCM):
    def __init__(self, d: int = 100, m: int = 2, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=5, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM6(LSCM):
    def __init__(self, d: int = 100, m: int = 2, ref_num:int=1000, **kwargs):
        super().__init__(ku_index=6, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM7(LSCM):
    def __init__(self, d: int = 100, m: int = 2, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=7, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM8(LSCM):
    def __init__(self, d: int = 100, m: int = 2, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=8, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM9(LSCM):
    def __init__(self, d: int = 100, m: int = 2, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=9, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM10(LSCM):
    def __init__(self, d: int = 100, m: int = 3, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=10, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM11(LSCM):
    def __init__(self, d: int = 100, m: int = 3, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=11, d=d, m=m, ref_num=ref_num,**kwargs)

class LSCM12(LSCM):
    def __init__(self, d: int = 100, m: int = 3, ref_num:int=1000,**kwargs):
        super().__init__(ku_index=12, d=d, m=m, ref_num=ref_num,**kwargs)