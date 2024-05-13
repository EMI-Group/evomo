import jax
import jax.numpy as jnp

def NDSort(PopObj, nSort):
    N, M = PopObj.shape
    no = 0
    orders = jnp.full(N, nSort)
    dom_nums = jnp.zeros(N, dtype=int)
    dom_sets = [set() for _ in range(N)]
    
    def compare(indi1, indi2):
        return jnp.all(indi1 <= indi2) and jnp.any(indi1 < indi2)
    
    for i in range(N):
        for j in range(i + 1, N):
            if compare(PopObj[i, :], PopObj[j, :]):
                dom_nums = dom_nums.at[j].add(1)
                dom_sets[i].add(j)
            elif compare(PopObj[j, :], PopObj[i, :]):
                dom_nums = dom_nums.at[i].add(1)
                dom_sets[j].add(i)
    
    while nSort > 0:
        no += 1
        nSort -= 1
        next_dom_nums = dom_nums
        for i in range(N):
            if dom_nums[i] == 0:
                orders = orders.at[i].set(no)
                next_dom_nums = next_dom_nums.at[i].set(-1)
                for j in dom_sets[i]:
                    next_dom_nums = next_dom_nums.at[j].add(-1)
        if jnp.all(next_dom_nums == dom_nums):
            break
        dom_nums = next_dom_nums

    orders = jnp.where(orders <= no, orders, no)
    return orders, no

def CrowdingDistance(PopObj, FrontNo):
    N, M = PopObj.shape
    cd = jnp.zeros(N)
    fronts = jnp.unique(FrontNo)
    
    def calculate_front(front):
        Fmax = jnp.max(PopObj[front, :], axis=0)
        Fmin = jnp.min(PopObj[front, :], axis=0)
        front_cd = jnp.zeros(N)
        
        for k in range(M):
            sorted_indices = jnp.argsort(PopObj[front, k])
            sorted_front = front[sorted_indices]
            # 为了匹配长度，在diff之前和之后添加最大值和最小值
            values = PopObj[sorted_front, k]
            extended_values = jnp.concatenate((jnp.array([Fmin[k]]), values, jnp.array([Fmax[k]])))
            distances = jnp.diff(extended_values)  # 应该有len(front) + 1 个值
            distances = distances[1:-1]  # 去掉头尾，恢复到正确的长度
            if jnp.any(Fmax[k] - Fmin[k] > 0):  # 防止除以0
                scaled_distances = distances / (Fmax[k] - Fmin[k])
                # 确保距离更新在正确的范围内
                front_cd = front_cd.at[sorted_front[1:-1]].add(scaled_distances)  # 更新距离
            
        front_cd = front_cd.at[sorted_front[[0, -1]]].set(jnp.inf)
        return front_cd
    
    for f in fronts:
        front_indices = jnp.where(FrontNo == f)[0]
        cd = cd.at[front_indices].set(calculate_front(front_indices))
    
    return cd

def TournamentSelection(K, N, FrontNo, CrowdDis):
    idx = jnp.arange(N)
    parents = jax.random.randint(jax.random.PRNGKey(0), shape=(K, 2), minval=0, maxval=N)
    
    def select(p):
        cond = (FrontNo[p[0]] < FrontNo[p[1]]) | ((FrontNo[p[0]] == FrontNo[p[1]]) & (CrowdDis[p[0]] > CrowdDis[p[1]]))
        return jnp.where(cond, p[0], p[1])
    
    return jnp.apply_along_axis(select, 1, parents)

# Usage example
PopObj = jax.random.uniform(jax.random.PRNGKey(0), (100, 3))
nSort = 10
orders, no = NDSort(PopObj, nSort)
