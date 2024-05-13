import jax
import jax.numpy as jnp


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
    # orders = jnp.array(orders)
    orders = jnp.where(orders < no, orders, no)
    return orders, no

def compare(com1, com2, M):
    res = True
    for i in range(M):
        if com1[i] > com2[i]:
          res = False
    return res

def CrowdingDistance(PopObj, FrontNo):
    [N, M] = PopObj.size()
    cd = [0.0] * N
    fronts = jnp.unique(FrontNo)
    for i in range(fronts.size()[0]):
        front = jnp.nonzero(FrontNo == fronts[i]).squeeze_()
        Fmax = jnp.max(PopObj[front,:],axis=0)
        Fmin = jnp.min(PopObj[front,:],axis=0)
        for k in range (M):
            rank = jnp.sort(PopObj[front,k])
            cd[front[rank[0]]] = float('inf')
            cd[front[rank[-1]]] = float('inf')
            for j in range(1,len(front)-1):
                cd[front[rank[j]]] += (PopObj[front[rank[j+1]],k] + PopObj[front[rank[j-1]],k]) / (Fmax[k]- Fmin[k])
                
    return cd

def TournamentSelection(K,N, FrontNo, CrowdDis):
    parents = jnp.randint(high=N, size=(K,N))
    parents = jnp.where(FrontNo[parents[0,:]] < FrontNo[parents[1,:]] or
                     (FrontNo[parents[0,:]] - FrontNo[parents[1,:]] == 0 and
                      CrowdDis[parents[0,:]] > CrowdDis[parents[1,:]]),
                     parents[0,:], parents[1,:])
    # return the index of population
    return parents