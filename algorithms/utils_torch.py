import torch

def NDSort(PopObj, nSort):
    [N, M] = PopObj.size()
    no = 0
    orders = [nSort] * N
    dom_nums = [0] * N
    dom_sets = [set()] * N
    for i in range(N):
        indi1 = PopObj[i,:]
        for j in range(i+1, N):
            indi2 = PopObj[j,:]
            if compare(indi1, indi2, M):
                dom_nums[j] += 1
                dom_sets[i].add(j)
            elif compare(indi2, indi1, M):
                dom_nums[i] += 1
                dom_sets[j].add(i)
    end = True
    while nSort > 0 and end:
        no += 1
        nSort -= 1
        end = False
        for i in range(len(dom_nums)):
            if dom_nums[i] == 0:
                orders[i] = no
                dom_nums[i] = -1
                end = True
        for i in range(len(dom_nums)):
            if dom_nums[i] == -1:
                for j in dom_sets[i]:
                    dom_nums[j] -= 1
                dom_nums[i] = -2
    orders = torch.tensor(orders)
    orders = torch.where(orders < no, orders, no)
    return orders, no

def compare(com1, com2, M):
    res = True
    for i in range(M):
        if com1[i] > com2[i]:
          res = False
    return res

def TournamentSelection(K,N, FrontNo, CrowdDis):
    parents = torch.randint(high=N, size=(K,N))
    parents = torch.where(FrontNo[parents[0,:]] < FrontNo[parents[1,:]] or
                     (FrontNo[parents[0,:]] - FrontNo[parents[1,:]] == 0 and
                      CrowdDis[parents[0,:]] > CrowdDis[parents[1,:]]),
                     parents[0,:], parents[1,:])
    # return the index of population
    return parents

# 生产均匀向量
def UniformPoint(N,M):
    H = 1
    while factorial(H+M)/(factorial(H+1)*factorial(M-1)) <= N:
        H += 1
    w1 = Permutation(list(range(H+1)),M-1)
    w2 = torch.tensor(w1)
    w3 = w2.sum(1)
    redundancy = []
    for i in range(w3.size()[0]):
        if w3[i] <= H:
            w1[i].append(H-w3[i].item())
        else:
            redundancy.append(w1[i])
    for i in redundancy:
        w1.remove(i)
    W = torch.tensor(w1, dtype=torch.float64) / H
    W = torch.where(W == 0, 1e-6, W)
    return W, W.size()[0]



def factorial(N):
    x = N
    while N != 1:
        N = N -1
        x *= N
    return x

def ArraySum(array):
    res = 0
    for i in array:
        res += i
    return res

def PermutationNumber(N, M):
    return int(factorial(N) / (factorial(N-M) * factorial(M)))


def Permutation(sequence, length):
    res = []
    if len(sequence) < length:
        pass
    elif length == 1:
        for i in sequence:
            res.append([i])
    else:
        for i in range(len(sequence) - 1):
            next_res = Permutation(sequence[i+1:], length-1)
            for j in next_res:
                j.append(sequence[i])
            res += next_res
    return res

# a = UniformPoint(100,3)
# print(a)

def CrowdingDistance(PopObj, FrontNo):
    [N, M] = PopObj.size()
    cd = [0.0] * N
    fronts = torch.unique(FrontNo)
    for i in range(fronts.size()[0]):
        front = torch.nonzero(FrontNo == fronts[i]).squeeze_()
        Fmax, x = torch.max(PopObj[front,:],dim=0)
        Fmin, x = torch.min(PopObj[front,:],dim=0)
        for k in range (M):
            x, rank = torch.sort(PopObj[front,k])
            cd[front[rank[0]]] = float('inf')
            cd[front[rank[-1]]] = float('inf')
            for j in range(1,len(front)-1):
                cd[front[rank[j]]] += (PopObj[front[rank[j+1]],k] + PopObj[front[rank[j-1]],k]) / (Fmax[k]- Fmin[k])

    return torch.tensor(cd)
