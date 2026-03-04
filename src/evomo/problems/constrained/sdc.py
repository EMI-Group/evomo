
__all__ = ['SDC1', 'SDC2', 'SDC3', 'SDC4', 'SDC5', 'SDC6', 'SDC7', 
           'SDC8', 'SDC9', 'SDC10', 'SDC11', 'SDC12', 'SDC13', 'SDC14', 'SDC15']

from typing import Tuple, Optional
import torch
from evomo.utils import get_pareto_front
from evox.operators.sampling import uniform_sampling, grid_sampling
from evox.utils import clamp
from evomo.problems.constrained.base import CMOP


def information(index: int) -> list:
    CEC_problem     = [1, 2, 3, 6, 9,10,11,12,14,18,19,24,15, 5, 1]
    shape_problem   = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1]
    b          = [10,100,15,115,19,125,10,100,15,115,19,115,125,15,10]
    Distance_prob   = [2, 1, 4, 4, 3, 5, 5, 3, 3, 2, 1, 3, 5, 1, 2]

    HCT, DCT        = 0.5, 0.5
    high_type       = [1,2,1,1,2,2,2,2,1,1,1,2,2,2,1]
    dis_type        = [2,1,1,2,2,2,1,1,2,2,1,1,1,1,2]

    i = index - 1  
    a = [CEC_problem[i], Distance_prob[i],
         HCT,           high_type[i],
         DCT,           dis_type[i],
         shape_problem[i], b[i]]
    return a


def CEC_2006_information(problem: int, *, device: torch.device | None = None,
        dtype: torch.dtype | None = None):
    p = problem
    lu = n = aaa = optimal = None

    match p:
        case 1:
            lu = torch.tensor([[0]*13,
                               [1,1,1,1,1,1,1,1,1,100,100,100,1]], dtype=dtype, device=device)
            n, optimal = 13, -15.0

        case 2:
            lu = torch.stack([torch.zeros(20, dtype=dtype, device=device),
                              10*torch.ones(20, dtype=dtype, device=device)])
            n, optimal = 20, -0.803619

        case 3:
            lu = torch.stack([torch.zeros(10, dtype=dtype, device=device),
                              torch.ones(10, dtype=dtype, device=device)])
            n, optimal = 10, -1.0

        case 4:
            lu = torch.tensor([[78,33,27,27,27],
                               [102,45,45,45,45]], dtype=dtype, device=device)
            n, optimal = 5, -30665.539

        case 5:
            lu = torch.tensor([[0,0,-0.55,-0.55],
                               [1200,1200,0.55,0.55]], dtype=dtype, device=device)
            n, optimal = 4, 5126.4981

        case 6:
            lu = torch.tensor([[13,0],[100,100]], dtype=dtype, device=device)
            n, optimal = 2, -6961.81388

        case 7:
            lu = torch.stack([-10*torch.ones(10, dtype=dtype, device=device),
                               10*torch.ones(10, dtype=dtype, device=device)])
            n, optimal = 10, 24.306

        case 8:
            lu = torch.tensor([[0,0],[10,10]], dtype=dtype, device=device)
            n, optimal = 2, 0.095825

        case 9:
            lu = torch.stack([-10*torch.ones(7, dtype=dtype, device=device),
                               10*torch.ones(7, dtype=dtype, device=device)])
            n, optimal = 7, 680.6300573

        case 10:
            lu = torch.tensor([[100,1000,1000,10,10,10,10,10],
                               [10000,10000,10000,1000,1000,1000,1000,1000]], dtype=dtype, device=device)
            n, optimal = 8, 7049.2480

        case 11:
            lu = torch.tensor([[-1,-1],[1,1]], dtype=dtype, device=device)
            n, optimal = 2, 0.75

        case 12:
            lu = torch.tensor([[0,0,0],[10,10,10]], dtype=dtype, device=device)
            n, optimal = 3, -1.0
            rng = torch.arange(1, 10, device=device)
            I, J, K = torch.meshgrid(rng, rng, rng, indexing='ij')
            aaa = torch.stack([I.reshape(-1), J.reshape(-1), K.reshape(-1)], dim=1).to(dtype)

        case 13:
            lu = torch.tensor([[-2.3,-2.3,-3.2,-3.2,-3.2],
                               [ 2.3, 2.3, 3.2, 3.2, 3.2]], dtype=dtype, device=device)
            n, optimal = 5, 0.0539498

        case 14:
            lu = torch.stack([torch.zeros(10, dtype=dtype, device=device),
                              10*torch.ones(10, dtype=dtype, device=device)])
            n, optimal = 10, -47.7648884595

        case 15:
            lu = torch.stack([torch.zeros(3, dtype=dtype, device=device),
                              10*torch.ones(3, dtype=dtype, device=device)])
            n, optimal = 3, 961.7150222899

        case 16:
            lu = torch.tensor([[704.4148,68.6,0,193,25],
                               [906.3855,288.88,134.75,287.0966,84.1988]], dtype=dtype, device=device)
            n, optimal = 5, -1.9051552586

        case 17:
            lu = torch.tensor([[0,0,340,340,-1000,0],
                               [400,1000,420,420,1000,0.5236]], dtype=dtype, device=device)
            n, optimal = 6, 8853.5396748064

        case 18:
            lu = torch.tensor([[-10,-10,-10,-10,-10,-10,-10,-10,0],
                               [ 10, 10, 10, 10, 10, 10, 10, 10,20]], dtype=dtype, device=device)
            n, optimal = 9, -0.8660254038

        case 19:
            lu = torch.stack([torch.zeros(15, dtype=dtype, device=device),
                              10*torch.ones(15, dtype=dtype, device=device)])
            n, optimal = 15, 32.6555929502

        case 20:
            lu = torch.stack([torch.zeros(24, dtype=dtype, device=device),
                              10*torch.ones(24, dtype=dtype, device=device)])
            n, optimal = 24, 0.2049794002

        case 21:
            lu = torch.tensor([[0,0,0,100,6.3,5.9,4.5],
                               [1000,40,40,300,6.7,6.4,6.25]], dtype=dtype, device=device)
            n, optimal = 7, 193.72451007

        case 22:
            lu = torch.tensor([[0,0,0,0,0,0,0,100,100,100.01,100,100,0,0,0,0.01,0.01,-4.7,-4.7,-4.7,-4.7,-4.7],
                               [20000,1e6,1e6,1e6,4e7,4e7,4e7,299.99,399.99,300,400,600,500,500,500,300,400,6.25,6.25,6.25,6.25,6.25]],
                              dtype=dtype, device=device)
            n, optimal = 22, 236.4309755040

        case 23:
            lu = torch.tensor([[0,0,0,0,0,0,0,0,0.01],
                               [300,300,100,200,100,300,100,200,0.03]], dtype=dtype, device=device)
            n, optimal = 9, -400.0551

        case 24:
            lu = torch.tensor([[0,0],[3,4]], dtype=dtype, device=device)
            n, optimal = 2, -5.5080132716

        case _:
            raise ValueError(f"Unknown CEC 2006 problem: {p}")

    return lu, n, aaa, torch.tensor(optimal,dtype=dtype, device=device)

def CEC_2006_fitness(P: torch.Tensor,problem: int,aaa: torch.Tensor,optimal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    N, D = P.shape
    device, dtype = P.device, P.dtype
    ncon_map = {
        1:9, 2:2, 3:1, 4:6, 5:5, 6:2, 7:8, 8:2, 9:4, 10:6,
        11:1, 12:1, 13:3, 14:3, 15:2, 16:38, 17:4, 18:13,
        19:5, 20:20, 21:6, 22:20, 23:6, 24:2, 25:1
    }
    ncon = ncon_map[problem]
    g = torch.zeros((N, ncon), device=device, dtype=dtype)
   
    match problem:

        case 1:
            g[:,0] = 2*P[:,0] + 2*P[:,1] + P[:,9]  + P[:,10] - 10
            g[:,1] = 2*P[:,0] + 2*P[:,2] + P[:,9]  + P[:,11] - 10
            g[:,2] = 2*P[:,1] + 2*P[:,2] + P[:,10] + P[:,11] - 10
            g[:,3] = -8*P[:,0] + P[:,9]
            g[:,4] = -8*P[:,1] + P[:,10]
            g[:,5] = -8*P[:,2] + P[:,11]
            g[:,6] = -2*P[:,3] - P[:,4] + P[:,9]
            g[:,7] = -2*P[:,5] - P[:,6] + P[:,10]
            g[:,8] = -2*P[:,7] - P[:,8] + P[:,11]
            f = 5*P[:,0:4].sum(1) - 5*(P[:,0:4]**2).sum(1) - P[:,4:13].sum(1)

        case 2:
            g[:,0] = 0.75 - P.prod(1)
            g[:,1] = P.sum(1) - 7.5*D
            num = (torch.cos(P)**4).sum(1).abs() - 2*(torch.cos(P)**2).prod(1)
            den = torch.sqrt(1e-30 + (torch.arange(1, D+1, device=device, dtype=dtype) * (P**2)).sum(1))
            f = - num.abs() / den

        case 3:
            g[:,0] = (P.pow(2).sum(1) - 1.0).abs() - 1e-4
            f = - (torch.sqrt(torch.tensor(10.0, device=device, dtype=dtype)))**10 * P.prod(1)

        case 4:
            g[:,0] =  85.334407 + 0.0056858*P[:,1]*P[:,4] + 0.0006262*P[:,0]*P[:,3] - 0.0022053*P[:,2]*P[:,4] - 92
            g[:,1] = -85.334407 - 0.0056858*P[:,1]*P[:,4] - 0.0006262*P[:,0]*P[:,3] + 0.0022053*P[:,2]*P[:,4]
            g[:,2] =  80.51249  + 0.0071317*P[:,1]*P[:,4] + 0.0029955*P[:,0]*P[:,1] + 0.0021813*P[:,2]**2 - 110
            g[:,3] = -80.51249  - 0.0071317*P[:,1]*P[:,4] - 0.0029955*P[:,0]*P[:,1] - 0.0021813*P[:,2]**2 + 90
            g[:,4] =  9.300961  + 0.0047026*P[:,2]*P[:,4] + 0.0012547*P[:,0]*P[:,2] + 0.0019085*P[:,2]*P[:,3] - 25
            g[:,5] = -9.300961  - 0.0047026*P[:,2]*P[:,4] - 0.0012547*P[:,0]*P[:,2] - 0.0019085*P[:,2]*P[:,3] + 20
            f = 5.3578547*P[:,2]**2 + 0.8356891*P[:,0]*P[:,4] + 37.293239*P[:,0] - 40792.141

        case 5:
            g[:,0] = -P[:,3] + P[:,2] - 0.55
            g[:,1] = -P[:,2] + P[:,3] - 0.55
            g[:,2] = (1000*torch.sin(-P[:,2]-0.25) + 1000*torch.sin(-P[:,3]-0.25) + 894.8 - P[:,0]).abs() - 1e-4
            g[:,3] = (1000*torch.sin(P[:,2]-0.25) + 1000*torch.sin(P[:,2]-P[:,3]-0.25) + 894.8 - P[:,1]).abs() - 1e-4
            g[:,4] = (1000*torch.sin(P[:,3]-0.25) + 1000*torch.sin(P[:,3]-P[:,2]-0.25) + 1294.8).abs() - 1e-4
            f = 3*P[:,0] + 1e-6*P[:,0]**3 + 2*P[:,1] + (2e-6/3)*P[:,1]**3

        case 6:
            g[:,0] = -(P[:,0]-5)**2 - (P[:,1]-5)**2 + 100
            g[:,1] =  (P[:,0]-6)**2 + (P[:,1]-5)**2 - 82.81
            f = (P[:,0]-10)**3 + (P[:,1]-20)**3

        case 7:
            g[:,0] = -105 + 4*P[:,0] + 5*P[:,1] - 3*P[:,6] + 9*P[:,7]
            g[:,1] = 10*P[:,0] - 8*P[:,1] - 17*P[:,6] + 2*P[:,7]
            g[:,2] = -8*P[:,0] + 2*P[:,1] + 5*P[:,8] - 2*P[:,9] - 12
            g[:,3] = 3*(P[:,0]-2)**2 + 4*(P[:,1]-3)**2 + 2*P[:,2]**2 - 7*P[:,3] - 120
            g[:,4] = 5*P[:,0]**2 + 8*P[:,1] + (P[:,2]-6)**2 - 2*P[:,3] - 40
            g[:,5] = P[:,0]**2 + 2*(P[:,1]-2)**2 - 2*P[:,0]*P[:,1] + 14*P[:,4] - 6*P[:,5]
            g[:,6] = 0.5*(P[:,0]-8)**2 + 2*(P[:,1]-4)**2 + 3*P[:,4]**2 - P[:,5] - 30
            g[:,7] = -3*P[:,0] + 6*P[:,1] + 12*(P[:,8]-8)**2 - 7*P[:,9]
            f = P[:,0]**2 + P[:,1]**2 + P[:,0]*P[:,1] - 14*P[:,0] - 16*P[:,1] + (P[:,2]-10)**2 + 4*(P[:,3]-5)**2 + \
                (P[:,4]-3)**2 + 2*(P[:,5]-1)**2 + 5*P[:,6]**2 + 7*(P[:,7]-11)**2 + 2*(P[:,8]-10)**2 + (P[:,9]-7)**2 + 45

        case 8:
            g[:,0] = P[:,0]**2 - P[:,1] + 1
            g[:,1] = 1 - P[:,0] + (P[:,1]-4)**2
            f = -(torch.sin(2*torch.pi*P[:,0])**3) * torch.sin(2*torch.pi*P[:,1]) / (P[:,0]**3 * (P[:,0] + P[:,1]) + 1e-30)

        case 9:
            g[:,0] = -127 + 2*P[:,0]**2 + 3*P[:,1]**4 + P[:,2] + 4*P[:,3]**2 + 5*P[:,4]
            g[:,1] = -282 + 7*P[:,0] + 3*P[:,1] + 10*P[:,2]**2 + P[:,3] - P[:,4]
            g[:,2] = -196 + 23*P[:,0] + P[:,1]**2 + 6*P[:,5]**2 - 8*P[:,6]
            g[:,3] = 4*P[:,0]**2 + P[:,1]**2 - 3*P[:,0]*P[:,1] + 2*P[:,2]**2 + 5*P[:,5] - 11*P[:,6]
            f = (P[:,0]-10)**2 + 5*(P[:,1]-12)**2 + P[:,2]**4 + 3*(P[:,3]-11)**2 + 10*P[:,4]**6 + \
                7*P[:,5]**2 + P[:,6]**4 - 4*P[:,5]*P[:,6] - 10*P[:,5] - 8*P[:,6]

        case 10:
            g[:,0] = -1 + 0.0025*(P[:,3]+P[:,5])
            g[:,1] = -1 + 0.0025*(P[:,4]+P[:,6]-P[:,3])
            g[:,2] = -1 + 0.01*(P[:,7]-P[:,4])
            g[:,3] = -P[:,0]*P[:,5] + 833.33252*P[:,3] + 100*P[:,0] - 83333.333
            g[:,4] = -P[:,1]*P[:,6] + 1250*P[:,4] + P[:,1]*P[:,3] - 1250*P[:,3]
            g[:,5] = -P[:,2]*P[:,7] + 1250000 + P[:,2]*P[:,4] - 2500*P[:,4]
            f = P[:,0] + P[:,1] + P[:,2]

        case 11:
            g[:,0] = (P[:,1] - P[:,0]**2).abs() - 1e-4
            f = P[:,0]**2 + (P[:,1]-1)**2

        case 12:
            f = -(100 - (P[:,0]-5)**2 - (P[:,1]-5)**2 - (P[:,2]-5)**2) / 100
            if aaa is None:
                raise ValueError("problem 12 requires `aaa` with shape (729, 3).")
            X3 = P[:, :3].unsqueeze(1)                                 # (N,1,3)
            A3 = aaa.to(device=device, dtype=dtype).unsqueeze(0)       # (1,M,3)
            D2 = ((X3 - A3) ** 2).sum(dim=2)                           # (N,M)
            g[:,0] = D2.min(dim=1).values - 0.0625

        case 13:
            g[:,0] = (P[:,0:5].pow(2).sum(1) - 10).abs() - 1e-4
            g[:,1] = (P[:,1]*P[:,2] - 5*P[:,3]*P[:,4]).abs() - 1e-4
            g[:,2] = (P[:,0]**3 + P[:,1]**3 + 1).abs() - 1e-4
            f = torch.exp(P[:,0]*P[:,1]*P[:,2]*P[:,3]*P[:,4])

        case 14:
            c = torch.tensor([-6.089, -17.164, -34.054, -5.914, -24.721,
                              -14.986, -24.1, -10.708, -26.662, -22.179],
                             device=device, dtype=dtype)
            sP = P.sum(1, keepdim=True)
            numer = P * (c + torch.log(1e-30 + P / (1e-30 + sP)))
            g[:,0] = (P[:,0] + 2*P[:,1] + 2*P[:,2] + P[:,5] + P[:,9]  - 2).abs() - 1e-4
            g[:,1] = (P[:,3] + 2*P[:,4] + P[:,5] + P[:,6] - 1).abs()  - 1e-4
            g[:,2] = (P[:,2] + P[:,6] + P[:,7] + 2*P[:,8] + P[:,9] - 1).abs() - 1e-4
            f = numer.sum(1)

        case 15:
            g[:,0] = (P[:,0:3].pow(2).sum(1) - 25).abs() - 1e-4
            g[:,1] = (8*P[:,0] + 14*P[:,1] + 7*P[:,2] - 56).abs() - 1e-4
            f = 1000 - P[:,0]**2 - 2*P[:,1]**2 - P[:,2]**2 - P[:,0]*P[:,1] - P[:,0]*P[:,2]

        case 16:
            y1 = P[:,1] + P[:,2] + 41.6
            c1 = 0.024*P[:,3] - 4.62
            y2 = 12.5/c1 + 12
            c2 = 0.0003535*P[:,0]**2 + 0.5311*P[:,0] + 0.08705*y2*P[:,0]
            c3 = 0.052*P[:,0] + 78 + 0.002377*y2*P[:,0]
            y3 = c2 / c3
            y4 = 19*y3
            c4 = 0.04782*(P[:,0] - y3) + 0.1956*(P[:,0] - y3)**2 / P[:,1] + 0.6376*y4 + 1.594*y3
            c5 = 100*P[:,1]
            c6 = P[:,0] - y3 - y4
            c7 = 0.950 - c4/c5
            y5 = c6 * c7
            y6 = P[:,0] - y5 - y4 - y3
            c8 = (y5 + y4) * 0.995
            y7 = c8 / y1
            y8 = c8 / 3798
            c9 = y7 - 0.0663*y7/y8 - 0.3153
            y9  = 96.82/c9 + 0.321*y1
            y10 = 1.29*y5 + 1.258*y4 + 2.29*y3 + 1.71*y6
            y11 = 1.71*P[:,0] - 0.452*y4 + 0.580*y3
            c10 = 12.3/752.3
            c11 = 1.75*y2*0.995*P[:,0]
            c12 = 0.995*y10 + 1998.0
            y12 = c10*P[:,0] + (c11/c12)
            y13 = c12 - 1.75*y2
            y14 = 3623.0 + 64.4*P[:,1] + 58.4*P[:,2] + (146312.0/(y9 + P[:,4]))
            c13 = 0.995*y10 + 60.8*P[:,1] + 48*P[:,3] - 0.1121*y14 - 5095.0
            y15 = y13 / c13
            y16 = 148000.0 - 331000.0*y15 + 40.0*y13 - 61.0*y15*y13
            c14 = 2324*y10 - 28740000*y2
            y17 = 14130000 - 1328.0*y10 - 531.0*y11 + (c14/c12)
            c15 = (y13/y15) - (y13/0.52)
            c16 = 1.104 - 0.72*y15
            c17 = y9 + P[:,4]

            g[:,0]  = 0.28/0.72 * y5 - y4
            g[:,1]  = P[:,2] - 1.5*P[:,1]
            g[:,2]  = 3496*y2/c12 - 21
            g[:,3]  = 110.6 + y1 - 62212/c17
            g[:,4]  = 213.1 - y1
            g[:,5]  = y1 - 405.23
            g[:,6]  = 17.505 - y2
            g[:,7]  = y2 - 1053.6667
            g[:,8]  = 11.275 - y3
            g[:,9]  = y3 - 35.03
            g[:,10] = 214.228 - y4
            g[:,11] = y4 - 665.585
            g[:,12] = 7.458 - y5
            g[:,13] = y5 - 584.463
            g[:,14] = 0.961 - y6
            g[:,15] = y6 - 265.916
            g[:,16] = 1.612 - y7
            g[:,17] = y7 - 7.046
            g[:,18] = 0.146 - y8
            g[:,19] = y8 - 0.222
            g[:,20] = 107.99 - y9
            g[:,21] = y9 - 273.366
            g[:,22] = 922.693 - y10
            g[:,23] = y10 - 1286.105
            g[:,24] = 926.832 - y11
            g[:,25] = y11 - 1444.046
            g[:,26] = 18.766 - y12
            g[:,27] = y12 - 537.141
            g[:,28] = 1072.163 - y13
            g[:,29] = y13 - 3247.039
            g[:,30] = 8961.448 - y14
            g[:,31] = y14 - 26844.086
            g[:,32] = 0.063 - y15
            g[:,33] = y15 - 0.386
            g[:,34] = 71084.33 - y16
            g[:,35] = -140000 + y16
            g[:,36] = 2802713 - y17
            g[:,37] = y17 - 12146108

            f = 0.000117*y14 + 0.1365 + 0.00002358*y13 + 0.000001502*y16 + 0.0321*y12 \
                + 0.004324*y5 + 0.0001*(c15/c16) + 37.48*(y2/c12) - 0.0000005843*y17

        case 17:
            g[:,0] = (-P[:,0] + 300 - P[:,2]*P[:,3]/131.078*torch.cos(1.48477 - P[:,5]) + 0.90798*P[:,2]**2/131.078*torch.cos(1.47588)).abs() - 1e-4
            g[:,1] = (-P[:,1] - P[:,2]*P[:,3]/131.078*torch.cos(1.48477 + P[:,5]) + 0.90798*P[:,3]**2/131.078*torch.cos(1.47588)).abs() - 1e-4
            g[:,2] = (-P[:,4] - P[:,2]*P[:,3]/131.078*torch.sin(1.48477 + P[:,5]) + 0.90798*P[:,3]**2/131.078*torch.sin(1.47588)).abs() - 1e-4
            g[:,3] = (200 - P[:,2]*P[:,3]/131.078*torch.sin(1.48477 - P[:,5]) + 0.90798*P[:,2]**2/131.078*torch.sin(1.47588)).abs() - 1e-4
            f = torch.where(P[:,0] < 300, 30*P[:,0], 31*P[:,0])
            seg2 = torch.where(P[:,1] < 100, 28*P[:,1],
                   torch.where(P[:,1] < 200, 29*P[:,1],
                   torch.where(P[:,1] < 1000, 30*P[:,1], 30*P[:,1])))
            f = f + seg2

        case 18:
            g[:,0]  = P[:,2]**2 + P[:,3]**2 - 1
            g[:,1]  = P[:,8]**2 - 1
            g[:,2]  = P[:,4]**2 + P[:,5]**2 - 1
            g[:,3]  = P[:,0]**2 + (P[:,1]-P[:,8])**2 - 1
            g[:,4]  = (P[:,0]-P[:,4])**2 + (P[:,1]-P[:,5])**2 - 1
            g[:,5]  = (P[:,0]-P[:,6])**2 + (P[:,1]-P[:,7])**2 - 1
            g[:,6]  = (P[:,2]-P[:,4])**2 + (P[:,3]-P[:,5])**2 - 1
            g[:,7]  = (P[:,2]-P[:,6])**2 + (P[:,3]-P[:,7])**2 - 1
            g[:,8]  = P[:,6]**2 + (P[:,7]-P[:,8])**2 - 1
            g[:,9]  = P[:,1]*P[:,2] - P[:,0]*P[:,3]
            g[:,10] = -P[:,2]*P[:,8]
            g[:,11] =  P[:,4]*P[:,8]
            g[:,12] =  P[:,5]*P[:,6] - P[:,4]*P[:,7]
            f = -0.5*(P[:,0]*P[:,3] - P[:,1]*P[:,2] + P[:,2]*P[:,8] - P[:,4]*P[:,8] + P[:,4]*P[:,7] - P[:,5]*P[:,6])

        case 19:
            a = torch.tensor([[-16, 2, 0, 1, 0],
                              [  0,-2, 0, 0.4, 2],
                              [ -3.5,0, 2, 0, 0],
                              [  0,-2, 0,-4, -1],
                              [  0,-9,-2, 1,-2.8],
                              [  2, 0,-4, 0, 0],
                              [ -1,-1,-1,-1,-1],
                              [ -1,-2,-3,-2,-1],
                              [  1, 2, 3, 4, 5],
                              [  1, 1, 1, 1, 1]], device=device, dtype=dtype)
            b = torch.tensor([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1], device=device, dtype=dtype)
            c = torch.tensor([[30,-20,-10, 32,-10],
                              [-20,39, -6,-31, 32],
                              [-10,-6, 10, -6,-10],
                              [32,-31,-6, 39,-20],
                              [-10,32,-10,-20, 30]], device=device, dtype=dtype)
            dvec = torch.tensor([4,8,10,6,2], device=device, dtype=dtype)
            evec = torch.tensor([-15,-27,-36,-18,-12], device=device, dtype=dtype)

            X11_15 = P[:,10:15]    # cols 11..15 (0-indexed)
            X1_10  = P[:,0:10]

            Cx = X11_15 @ c[:,0]; Cy = X11_15 @ c[:,1]; Cz = X11_15 @ c[:,2]
            Cu = X11_15 @ c[:,3]; Cv = X11_15 @ c[:,4]
            Ax1 = X1_10 @ a[:,0]; Ax2 = X1_10 @ a[:,1]; Ax3 = X1_10 @ a[:,2]
            Ax4 = X1_10 @ a[:,3]; Ax5 = X1_10 @ a[:,4]

            g[:,0] = -2*Cx - 3*dvec[0]*P[:,10]**2 - evec[0] + Ax1
            g[:,1] = -2*Cy - 3*dvec[1]*P[:,11]**2 - evec[1] + Ax2
            g[:,2] = -2*Cz - 3*dvec[2]*P[:,12]**2 - evec[2] + Ax3
            g[:,3] = -2*Cu - 3*dvec[3]*P[:,13]**2 - evec[3] + Ax4
            g[:,4] = -2*Cv - 3*dvec[4]*P[:,14]**2 - evec[4] + Ax5

            f = (X11_15 @ c[:,0]) * P[:,10] \
              + (X11_15 @ c[:,1]) * P[:,11] \
              + (X11_15 @ c[:,2]) * P[:,12] \
              + (X11_15 @ c[:,3]) * P[:,13] \
              + (X11_15 @ c[:,4]) * P[:,14] \
              + 2 * (X11_15**3 @ dvec) \
              - (X1_10 @ b)

        case 20:
            a = torch.tensor([0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09,
                              0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09],
                             device=device, dtype=dtype)
            b = torch.tensor([44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097,
                              44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.079],
                             device=device, dtype=dtype)
            c = torch.tensor([123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64],
                             device=device, dtype=dtype)
            dvec = torch.tensor([31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1],
                                device=device, dtype=dtype)
            e = torch.tensor([0.1, 0.3, 0.4, 0.3, 0.6, 0.3], device=device, dtype=dtype)

            S = P.sum(1)
            g[:,0] = (P[:,0]  + P[:,12]) / (S + e[0])
            g[:,1] = (P[:,1]  + P[:,13]) / (S + e[1])
            g[:,2] = (P[:,2]  + P[:,14]) / (S + e[2])
            g[:,3] = (P[:,6]  + P[:,18]) / (S + e[3])
            g[:,4] = (P[:,7]  + P[:,19]) / (S + e[4])
            g[:,5] = (P[:,8]  + P[:,20]) / (S + e[5])

            S13 = (P[:,12:24] / b[12:24]).sum(1)
            S12 = (P[:,0:12]  / b[0:12]).sum(1)
            g[:,6]  = (P[:,12]/(b[12]*S13) - c[0]*P[:,0]/(40*b[0]*S12)).abs() - 1e-4
            g[:,7]  = (P[:,13]/(b[13]*S13) - c[1]*P[:,1]/(40*b[1]*S12)).abs() - 1e-4
            g[:,8]  = (P[:,14]/(b[14]*S13) - c[2]*P[:,2]/(40*b[2]*S12)).abs() - 1e-4
            g[:,9]  = (P[:,15]/(b[15]*S13) - c[3]*P[:,3]/(40*b[3]*S12)).abs() - 1e-4
            g[:,10] = (P[:,16]/(b[16]*S13) - c[4]*P[:,4]/(40*b[4]*S12)).abs() - 1e-4
            g[:,11] = (P[:,17]/(b[17]*S13) - c[5]*P[:,5]/(40*b[5]*S12)).abs() - 1e-4
            g[:,12] = (P[:,18]/(b[18]*S13) - c[6]*P[:,6]/(40*b[6]*S12)).abs() - 1e-4
            g[:,13] = (P[:,19]/(b[19]*S13) - c[7]*P[:,7]/(40*b[7]*S12)).abs() - 1e-4
            g[:,14] = (P[:,20]/(b[20]*S13) - c[8]*P[:,8]/(40*b[8]*S12)).abs() - 1e-4
            g[:,15] = (P[:,21]/(b[21]*S13) - c[9]*P[:,9]/(40*b[9]*S12)).abs() - 1e-4
            g[:,16] = (P[:,22]/(b[22]*S13) - c[10]*P[:,10]/(40*b[10]*S12)).abs() - 1e-4
            g[:,17] = (P[:,23]/(b[23]*S13) - c[11]*P[:,11]/(40*b[11]*S12)).abs() - 1e-4

            g[:,18] = (P.sum(1) - 1).abs() - 1e-4
            coeff = 0.7302 * 530 * 14.7 / 40
            g[:,19] = ((P[:,0:12] / dvec).sum(1) + coeff * (P[:,12:24] / b[12:24]).sum(1) - 1.671).abs() - 1e-4

            f = P @ a

        case 21:
            g[:,0] = -P[:,0] + 35*P[:,1]**0.6 + 35*P[:,2]**0.6
            g[:,1] = (-300*P[:,2] + 7500*P[:,4] - 7500*P[:,5] - 25*P[:,3]*P[:,4] + 25*P[:,3]*P[:,5] + P[:,2]*P[:,3]).abs() - 1e-4
            g[:,2] = (100*P[:,1] + 155.365*P[:,3] + 2500*P[:,6] - P[:,1]*P[:,3] - 25*P[:,3]*P[:,6] - 15536.5).abs() - 1e-4
            g[:,3] = (-P[:,4] + torch.log(-P[:,3] + 900)).abs() - 1e-4
            g[:,4] = (-P[:,5] + torch.log(P[:,3] + 300)).abs() - 1e-4
            g[:,5] = (-P[:,6] + torch.log(-2*P[:,3] + 700)).abs() - 1e-4
            f = P[:,0]

        case 22:
            g[:,0]  = -P[:,0] + P[:,1]**0.6 + P[:,2]**0.6 + P[:,3]**0.6
            g[:,1]  = (P[:,4] - 100000*P[:,7] + 1e7).abs() - 1e-4
            g[:,2]  = (P[:,5] + 100000*P[:,7] - 100000*P[:,8]).abs() - 1e-4
            g[:,3]  = (P[:,6] + 100000*P[:,8] - 5e7).abs() - 1e-4
            g[:,4]  = (P[:,4] + 100000*P[:,9]  - 3.3e7).abs() - 1e-4
            g[:,5]  = (P[:,5] + 100000*P[:,10] - 4.4e7).abs() - 1e-4
            g[:,6]  = (P[:,6] + 100000*P[:,11] - 6.6e7).abs() - 1e-4
            g[:,7]  = (P[:,4] - 120*P[:,1]*P[:,12]).abs() - 1e-4
            g[:,8]  = (P[:,5] -  80*P[:,2]*P[:,13]).abs() - 1e-4
            g[:,9]  = (P[:,6] -  40*P[:,3]*P[:,14]).abs() - 1e-4
            g[:,10] = (P[:,7] - P[:,10] + P[:,15]).abs() - 1e-4
            g[:,11] = (P[:,8] - P[:,11] + P[:,16]).abs() - 1e-4
            g[:,12] = (-P[:,17] + torch.log(P[:,9]  - 100)).abs() - 1e-4
            g[:,13] = (-P[:,18] + torch.log(-P[:,7] + 300)).abs() - 1e-4
            g[:,14] = (-P[:,19] + torch.log(P[:,15])).abs()       - 1e-4
            g[:,15] = (-P[:,20] + torch.log(-P[:,8] + 400)).abs() - 1e-4
            g[:,16] = (-P[:,21] + torch.log(P[:,16])).abs()       - 1e-4
            g[:,17] = (-P[:,7] - P[:,9] + P[:,12]*P[:,17] - P[:,12]*P[:,18] + 400).abs() - 1e-4
            g[:,18] = (P[:,7] - P[:,8] - P[:,10] + P[:,13]*P[:,19] - P[:,13]*P[:,20] + 400).abs() - 1e-4
            g[:,19] = (P[:,8] - P[:,11] - 4.60517*P[:,14] + P[:,14]*P[:,21] + 100).abs() - 1e-4
            f = P[:,0]

        case 23:
            g[:,0] = P[:,8]*P[:,2] + 0.02*P[:,5] - 0.025*P[:,4]
            g[:,1] = P[:,8]*P[:,3] + 0.02*P[:,6] - 0.015*P[:,7]
            g[:,2] = (P[:,0] + P[:,1] - P[:,2] - P[:,3]).abs() - 1e-4
            g[:,3] = (0.03*P[:,0] + 0.01*P[:,1] - P[:,8]*(P[:,2] + P[:,3])).abs() - 1e-4
            g[:,4] = (P[:,2] + P[:,5] - P[:,4]).abs() - 1e-4
            g[:,5] = (P[:,3] + P[:,6] - P[:,7]).abs() - 1e-4
            f = -9*P[:,4] - 15*P[:,7] + 6*P[:,0] + 16*P[:,1] + 10*(P[:,5] + P[:,6])

        case 24:
            g[:,0] = -2*P[:,0]**4 + 8*P[:,0]**3 - 8*P[:,0]**2 + P[:,1] - 2
            g[:,1] = -4*P[:,0]**4 + 32*P[:,0]**3 - 88*P[:,0]**2 + 96*P[:,0] + P[:,1] - 36
            f = -P[:,0] - P[:,1]

        case 25:
            f = -(P[:,0] + P[:,1] - (P[:,0]**2 + 2*P[:,2]**2 + P[:,1]**2 + 2*P[:,0]*P[:,2] + 2*P[:,1]*P[:,2]))
            g[:,0] = (P[:,0] + P[:,1] + P[:,2] - 1).abs()

    term = torch.clamp(g, min=0)
    conV = term.sum(1)
    conV[conV < 1e-6] = 0

    objF = f - optimal
    objF = torch.where(objF.abs() <= 1e-3, torch.zeros_like(objF), objF)
    return objF, conV
        


def Distance_function(x: torch.Tensor, problem: int) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    upper = torch.full_like(x, 10.0)
    lower = torch.zeros_like(x)
    x = (upper - lower) * x + lower

    if problem == 1:
        # Sphere
        f = (x**2).sum(dim=1, keepdim=True)
    elif problem == 2:
        # Schwefel 
        f = x.abs().amax(dim=1, keepdim=True)
    elif problem == 3:
        # Rastrigin
        f = (x**2 - 10.0*torch.cos(2.0*torch.pi*x) + 10.0).sum(dim=1, keepdim=True)
    elif problem == 4:
        # Griewank
        term1 = (x**2).sum(dim=1, keepdim=True) / 4000.0
        idx = torch.arange(1, x.size(1)+1, device=device, dtype=dtype)
        term2 = torch.cos(x / torch.sqrt(idx)).prod(dim=1, keepdim=True)
        f = term1 - term2 + 1.0
    elif problem == 5:
        # Ackley
        f = (
            20
            - 20 * torch.exp(-0.2 * torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)))
            - torch.exp(torch.mean(torch.cos(2.0 * torch.pi * x), dim=1, keepdim=True))
            + torch.e
        )
    else:
        raise ValueError(f"Unknown distance problem: {problem}")

    f = torch.where(f < 1e-8, torch.zeros(1, device=device, dtype=dtype), f)
    return f

class SDC(CMOP):
    """
    Scalable high-dimensional decicsion constraint benchamrk

    K. Qiao, J. Liang, K. Yu, C. Yue, H. Lin, D. Zhang, and B. Qu. 
    Evolutionary constrained multiobjective optimization: scalable
    high-dimensional constraint benchmarks and algorithm. 
    IEEE Transactions on Evolutionary Computation, 2024, 28(4): 965-979.
    
    param:
    THETA_
    a
    CEC_Problem: index of high-dimension constraint function
    Distance_problem: index of distance function
    HCT: rate of unconstrained distance function variables that are used for high-dimensional constraint function
    HCT_type: type of transformation function
    DCT: rate of unconstrained distance function variables that are used for variable linkage
    DCT_type: type of variable linkage function
    Shape_problem: tyep of shape function
    b: a parameter of shape function that is used to control the overlap degree between CPF and UPF
    lu: upper and lower bounds of high-dimensional constraint function
    high_D_C: dimensions of high-dimensional constraint function
    aaa: used for high-dimensional constraint function
    optial_f: optimal objective value of high-dimensional constraints
    q1_upper: upper bound of shape function
    q1_lower:lower bound of shape function
    h: objective function value of high-dimensional constraint function
    HC: constraint function value of high-dimensional constraint function
    max_D_con: maximal number of variables that are used for transformation operator   
    """

    def __init__(self, info_idx: int, d: int = 30, m: int = 2, ref_num: int=1000,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,**kwargs):
        if device is None:
            self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if dtype is None:
            self.dtype = torch.get_default_dtype()
        else:
            self.dtype = dtype
        self.m=m
        self.ref_num=ref_num

        self.info = information(info_idx)
        (self.CEC_Problem,
         self.Distance_problem,
         self.HCT,
         self.HCT_type,
         self.DCT,
         self.DCT_type,
         self.Shape_problem,
         self.b) = self.info

        lu, high_D_C, aaa, opt_f = CEC_2006_information(self.CEC_Problem,dtype=self.dtype,device=self.device)
        self.lu = lu              # (2, high_D_C)
        self.high_D_C = high_D_C
        self.aaa = aaa
        self.optial_f = opt_f

        # Maximum dimension used for transformation operator
        self.max_D_con = (self.high_D_C + torch.ceil(torch.tensor((d - m - self.high_D_C) * self.HCT,device=self.device))).to(torch.long)

        # Shape parameters
        if self.Shape_problem == 1:
            self.q1_upper = torch.full((self.m,), 4.0, device=self.device, dtype=self.dtype)
            self.q1_lower = torch.zeros((self.m,), device=self.device, dtype=self.dtype)
        else:
            self.q1_upper = torch.full((self.m,), 2.0, device=self.device, dtype=self.dtype)
            self.q1_lower = torch.zeros((self.m,), device=self.device, dtype=self.dtype)


        lb = torch.zeros(d, device=self.device, dtype=self.dtype)
        ub = torch.ones(d,  device=self.device, dtype=self.dtype)

        super().__init__(d=d, m=m, n_iq=3, n_eq=0, lb=lb, ub=ub, device=self.device, dtype=self.dtype,**kwargs)

    def _transformation_operator(self, P: torch.Tensor) -> torch.Tensor:
        device, dtype = P.device, P.dtype
        N,D = P.shape
        lower = self.lu[0]   
        upper = self.lu[1]  
        scale = upper - lower 
        new_P = torch.empty((N, self.high_D_C), device=P.device, dtype=P.dtype)
        if D > self.high_D_C:
            q = D // self.high_D_C
            r = D % self.high_D_C
            counts = torch.full((self.high_D_C,), q, device=device, dtype=torch.long)
            if r > 0:
                counts[:r] += 1

            # Construct a column-to-group one-hot matrix A: shape (D, H), A[i, i%H] = 1
            col_idx = torch.arange(D, device=device)
            grp_idx = torch.remainder(col_idx, self.high_D_C)                  # (D,)
            A = torch.zeros((D, self.high_D_C), device=device, dtype=dtype)    # Use dtype for matmul
            A[col_idx, grp_idx] = 1.0

            a1 = P @ A
            a2 = torch.remainder(a1, 0.5)
            if self.HCT_type == 2:
                tempa = torch.cos(a2 * torch.pi)                   # (N, H)
            else:
                tempa = -2.0 * a2 + 1.0

            base = P[:, :self.high_D_C]                                        # (N, H)

            # Group selection mask: True -> more than 1 column in group, use HCT; False -> only 1 column, use base
            use_hct = (counts > 1).to(dtype=dtype)                 # (H,), 0/1
            # Broadcast to (N, H) by columns
            use_hct = use_hct.unsqueeze(0)                         # (1, H)
            mapped = lower + scale * tempa                     # (N, H)
            scaled_base = lower + scale * base                 # (N, H)
            new_P = use_hct * mapped + (1 - use_hct) * scaled_base
        else:
            new_P = lower[:D] + P * scale[:D]
        return new_P

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        N, D = X.shape
        device, dtype = X.device, X.dtype

        new_P = self._transformation_operator(X[:, self.m : self.m + self.max_D_con])

        # 计算高维约束的目标项与约束
        h, HC = CEC_2006_fitness(new_P, self.CEC_Problem, self.aaa, self.optial_f)   # (N,1), (N,1) or (N,K)

        # Variable linkage (coupling) DCT
        P = X.clone()
        control_D = torch.ceil(torch.tensor((self.d - self.high_D_C - self.m) * self.DCT,device=device)).to(torch.long)
        if self.DCT_type == 1:
            mult = 1.0 + torch.arange(1, control_D + 1, device=device, dtype=dtype) / control_D
        else:
            mult = 1.0 + torch.cos(0.5 * torch.pi * (torch.arange(1, control_D + 1, device=device, dtype=dtype) / control_D))
        P[:, D - control_D : D] =  P[:, D - control_D : D] * mult  - P[:, [0]]

        # Distance term
        dis_P = Distance_function(P[:, self.m + self.high_D_C :], self.Distance_problem)  # (N,1)

        # Angle THETA
        THETA = torch.atan2(torch.abs(X[:, 1]), X[:, 0]).unsqueeze(1)  # shape: (N,1)
      
        # 目标
        Pop = (self.q1_upper - self.q1_lower) * X[:, :self.m] + self.q1_lower
        T_ = (1.0 - torch.sum(Pop[:, :self.m] ** 2, dim=1, keepdim=True))**2 + h.unsqueeze(1) + dis_P  # (N,1)
        G_left  = torch.cat([torch.ones((N,1), device=device, dtype=dtype),
                             torch.cumprod(torch.sin(THETA), dim=1)], dim=1)  # (N,2)
        G_right = torch.cat([torch.cos(THETA),torch.ones((N,1), device=device, dtype=dtype)], dim=1)  # (N,2)
        G = G_left * G_right
        F = G * (1+T_) # (N,2)

        # Constraints (two shapes)
        if self.Shape_problem == 1:
            cc = self.b / 10.0
            c1 = cc**2 * Pop[:, [0]]**2 + Pop[:, [1]]**2 - cc**2
            c2 = -cc * (Pop[:, [0]] - 1.0) - Pop[:, [1]]
        else:
            l = torch.atan(Pop[:, [1]] / Pop[:, [0]])
            l = torch.where(torch.isnan(l), torch.ones_like(l), l)
            cc = self.b / 100.0
            c1 = Pop[:, [0]]**2 + Pop[:, [1]]**2 - (cc + 0.05 + 0.4 * torch.sin(4.0 * l)**16)**2
            c2 = (cc - 0.2 * torch.sin(4.0 * l)**8)**2 - Pop[:, [0]]**2 - Pop[:, [1]]**2

        return torch.cat([F, c1, c2, HC.unsqueeze(1)], dim=1)   

    def pf(self):
        pf, _ = uniform_sampling(self.ref_num * self.m, self.m)
        pf = pf.to(self.device, self.dtype)
        pf = pf / torch.sqrt((pf**2).sum(dim=1, keepdim=True))
        return pf


class SDC1(SDC):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        super().__init__(info_idx=1, d=d, m=m, ref_num = ref_num,**kwargs)


class SDC2(SDC):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        super().__init__(info_idx=2, d=d, m=m, ref_num = ref_num,**kwargs)


class SDC3(SDC):
    def __init__(self, info_idx=3, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=info_idx, d=d, m=m, ref_num = ref_num,**kwargs)

    def pf(self):
        # 1) Take a line segment in the first quadrant and L2-normalize to unit arc
        t = torch.linspace(0.0, 1.0, steps=self.ref_num,
                           device=self.device, dtype=self.dtype).unsqueeze(1)
        R = torch.cat([t, 1.0 - t], dim=1)
        R = R / torch.sqrt((R**2).sum(dim=1, keepdim=True))

        # 2) Linear constraint: cc*x + y >= cc  (cc = b/10)
        cc = self.b / 10.0
        # while True: 
        #     c = cc * R[:, [0]] + R[:, [1]] - cc 
        #     invalid = c < 0 
        #     if not invalid.any(): 
        #         break 
        #     R[invalid.squeeze(1)] *= 1.001
        denom = cc * R[:, [0]] + R[:, [1]]                                   # (N,1)
        s = torch.clamp(cc / denom, min=1.0)                                         # Need to scale to exactly satisfy the constraint
        R = R * s

        # 3) Polar coordinate remap to the front
        theta = torch.atan2(R[:, [1]], R[:, [0]])
        hx = (1.0 - (R**2).sum(dim=1, keepdim=True))**2
        R = torch.cat([torch.cos(theta) * (1.0 + hx),
                       torch.sin(theta) * (1.0 + hx)], dim=1)
        return R

class SDC4(SDC):
    def __init__(self,info_idx=4, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=info_idx, d=d, m=m, ref_num = ref_num,**kwargs)

    def pf(self):
        t = torch.linspace(0.0, 1.0, steps=self.ref_num, device=self.device, dtype=self.dtype).unsqueeze(1)
        R = torch.cat([t, 1.0 - t], dim=1)
        R = R / torch.sqrt((R**2).sum(dim=1, keepdim=True))

        cc = self.b / 100.0
        # while True:
        #     ang = torch.atan(R[:, [1]] / R[:, [0]])
        #     l = torch.sin(4.0 * ang)**8
        #     c = (cc - 0.2 * l)**2 - (R[:, [0]]**2 + R[:, [1]]**2)
        #     invalid = c > 0
        #     if not invalid.any():
        #         break
        #     R[invalid.squeeze(1)] *= 1.001
     
        theta_ang = torch.atan2(R[:, [1]], R[:, [0]])       # (N,1)
        s4 = torch.sin(4.0 * theta_ang)
        r_in_sq = (cc - 0.2 * (s4**8))**2                                # (N,1)
        r_sq = (R**2).sum(dim=1, keepdim=True)                           # approx 1
        # Need to scale radius up to >= r_in: scale factor s = sqrt(r_in_sq / r_sq), with minimum 1.0
        s = torch.sqrt(r_in_sq / r_sq)
        s = torch.clamp(s, min=1.0)
        R = R * s

        theta = torch.atan2(R[:, [1]], R[:, [0]])
        hx = (1.0 - (R**2).sum(dim=1, keepdim=True))**2
        R = torch.cat([torch.cos(theta) * (1.0 + hx), torch.sin(theta) * (1.0 + hx)], dim=1)
        return get_pareto_front(R)

class SDC5(SDC3):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=5, d=d, m=m, ref_num = ref_num,**kwargs)

class SDC6(SDC4):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=6, d=d, m=m, ref_num = ref_num,**kwargs)

class SDC7(SDC):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=7, d=d, m=m, ref_num = ref_num,**kwargs)

class SDC8(SDC):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=8, d=d, m=m, ref_num = ref_num,**kwargs)

class SDC9(SDC3):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=9, d=d, m=m, ref_num = ref_num,**kwargs)



class SDC10(SDC4):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=10, d=d, m=m, ref_num = ref_num,**kwargs)
    

class SDC11(SDC3):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=11, d=d, m=m, ref_num = ref_num,**kwargs)
    

class SDC12(SDC4):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=12, d=d, m=m, ref_num = ref_num,**kwargs)


class SDC13(SDC4):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=13, d=d, m=m, ref_num = ref_num,**kwargs)

    
class SDC14(SDC3):
    def __init__(self, d: int = 30, m: int = 2, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=14, d=d, m=m, ref_num = ref_num,**kwargs)


class SDC15(SDC):
    def __init__(self, d: int = 30, m: int = 3, ref_num: int = 1000, **kwargs):
        self.ref_num = ref_num
        super().__init__(info_idx=15, d=d, m=m, ref_num = ref_num,**kwargs)

    def fn(self, X: torch.Tensor) -> torch.Tensor:
        
        """
        Return: [F1, F2, F3, c1, c2, HC]  with shape (N, 6)
        - F*: 3 objectives (M=3)
        - c1,c2: shape constraints from SDC15
        - HC: aggregated high-dimensional constraint violation from CEC_2006
        """
        N, D = X.shape
        device, dtype = X.device, X.dtype

        new_P = self._transformation_operator(X[:, self.m : self.m + self.max_D_con])
        h, HC = CEC_2006_fitness(new_P, self.CEC_Problem, self.aaa, self.optial_f)

        P = X.clone()
        control_D = torch.ceil(
            torch.tensor((self.d - self.high_D_C - self.m) * self.DCT, device=device)
        ).to(torch.long)

        # Variable linkage
        if control_D > 0:
            # Last control_D variables
            idx_start = D - control_D
            idx_end = D
            t = torch.arange(1, control_D + 1, device=device, dtype=dtype) / control_D  # (control_D,)
            if self.DCT_type == 1:
                mult = 1.0 + t 
            else:  
                mult = 1.0 + torch.cos(0.5 * torch.pi * t)
            P[:, idx_start:idx_end] = P[:, idx_start:idx_end] * mult.unsqueeze(0) - P[:, [0]]

        dis_P = Distance_function(P[:, self.m + self.high_D_C :], self.Distance_problem)
 
        # Angle
        PopDec_head = X[:, : self.m]                                      # (N,3)
        Sx = torch.cumsum(PopDec_head.pow(2), dim=1)                      # (N,3), 累加（正向）
        Sx = torch.flip(torch.flip(Sx, dims=[1]).cumsum(dim=1), dims=[1]) # (N,3), Reverse accumulation: cumsum(...,'reverse')
        # angle(:,k) = atan( sqrt(Sx(:,k+1)) ./ PopDec(:,k) ),  k=1..M-1
        numerator = torch.sqrt(Sx[:, 1:])                                 # (N,2)
        denominator = PopDec_head[:, : self.m - 1]                        # (N,2)
        angle = torch.atan(numerator / denominator)                       # (N,2)
        angle = torch.where(torch.isnan(angle), torch.ones_like(angle), angle)  # NaN -> 1
        THETA = 2.0 / torch.pi * angle                                    # (N,2)

        # ===== 目标函数 =====
        # Shape scaling Pop (first 3 dimensions)
        Pop = (self.q1_upper - self.q1_lower) * X[:, :self.m] + self.q1_lower
        sum_pop2 = (Pop**2).sum(dim=1, keepdim=True)                  # (N,1)
        if h.ndim == 1:
            h = h.unsqueeze(1)                                            # (N,1)
        T_ = (1.0 - sum_pop2)**2 + h + dis_P                          # (N,1)

        # G_ = [1, cumprod(sin(pi/2*THETA),2)] .* [cos(pi/2*THETA), 1]
        s = torch.sin(0.5 * torch.pi * THETA)                           
        left = torch.cat([torch.ones((N, 1), device=device, dtype=dtype),
                        torch.cumprod(s, dim=1)], dim=1)                # (N,3)
        right = torch.cat([torch.cos(0.5 * torch.pi * THETA),
                        torch.ones((N, 1), device=device, dtype=dtype)], dim=1)  # (N,3)
        G = left * right                                                  # (N,3)
        F = G * (1.0 + T_)                                                # (N,3) -> F1,F2,F3

        # ===== 6) Constraints (Two shape constraints + high-dimensional constraint) =====
        c1 = 0.5 - sum_pop2                                               # (N,1)
        c2 = -2.0 - sum_pop2                                              # (N,1)
        if HC.ndim == 1:
            HC = HC.unsqueeze(1)                                          # (N,1)

        # ===== 7) Concatenate output =====
        return torch.cat([F, c1, c2, HC], dim=1)                          # (N, 6)