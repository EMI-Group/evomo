import jax
import jax.numpy as jnp
# from utils_torch import NDSort as NDSort_torch, CrowdingDistance as CrowdingDistance_torch, TournamentSelection as TournamentSelection_torch 
def NDSort_old(PopObj, nSort):
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


def NDSort(PopObj, nSort=None):
    N, M = PopObj.shape
    no = -1
    orders = jnp.full(N, nSort)
    dom_nums = jnp.zeros(N, dtype=int)

    # 广播比较所有对
    def compare_all(PopObj):
        return (jnp.all(PopObj[:, None, :] <= PopObj[None, :, :], axis=-1) & 
                jnp.any(PopObj[:, None, :] < PopObj[None, :, :], axis=-1))
    
    dominated = compare_all(PopObj)
    
    # 更新主导数字
    dom_nums = jnp.sum(dominated, axis=0)
    if nSort is None:
        nSort = PopObj.shape[0]
    while nSort > 0:
        no += 1
        next_dom_nums = dom_nums
        for i in range(N):
            if dom_nums[i] == 0:
                nSort -= 1
                orders = orders.at[i].set(no)
                next_dom_nums = next_dom_nums.at[i].set(-1)
                for j in range(N):
                    if dominated[i, j]:
                        next_dom_nums = next_dom_nums.at[j].add(-1)
        # if jnp.all(next_dom_nums == dom_nums):
        #     break
        dom_nums = next_dom_nums
        

    orders = jnp.where(orders <= no, orders, no+1)
    return orders, no


def CrowdingDistance(PopObj, FrontNo, maxF=None):
    N, M = PopObj.shape
    cd = jnp.zeros(N)
    
    def calculate_front(front):
        if front.shape[0] <= 2:
            return jnp.array([jnp.inf] * front.shape[0])
        
        front_cd = jnp.zeros(front.shape[0])
        
        def calculate_distance(k):
            current_popobj = PopObj[front, k]
            sorted_indices = jnp.argsort(current_popobj)
            values = current_popobj[sorted_indices]
            distances = values[2:] - values[:-2] # 应该有len(front) - 2 个值
            scaled_distances = distances / (values[-1] - values[0])
            scaled_distances = jnp.concatenate((jnp.array([jnp.inf]), scaled_distances, jnp.array([jnp.inf])))
            real_distances = scaled_distances.at[sorted_indices].set(scaled_distances)
            return real_distances
        
        front_cd = jax.vmap(calculate_distance)(jnp.arange(M))
        front_cd = front_cd.sum(axis=0)
        return front_cd
    
    if maxF is None:
        fronts = jnp.unique(FrontNo)
        for f in fronts:
            front_indices = jnp.where(FrontNo == f)[0]
            cd = cd.at[front_indices].set(calculate_front(front_indices))
    else:
        front_indices = jnp.where(FrontNo == maxF)[0]
        cd = cd.at[front_indices].set(calculate_front(front_indices))
        front_indices = jnp.where(FrontNo != maxF)[0]
        cd = cd.at[front_indices].set(-jnp.inf)
        
        
    
    return cd


def TournamentSelection(K, N, FrontNo, CrowdDis):
    idx = jnp.arange(N)
    parents = jax.random.randint(jax.random.PRNGKey(0), shape=(K, 2), minval=0, maxval=N)
    
    def select(p):
        cond = (FrontNo[p[0]] < FrontNo[p[1]]) | ((FrontNo[p[0]] == FrontNo[p[1]]) & (CrowdDis[p[0]] > CrowdDis[p[1]]))
        return jnp.where(cond, p[0], p[1])
    
    return jnp.apply_along_axis(select, 1, parents)


# 测试函数
def test_NDSort():
    import numpy as np
    import torch
    from evox.operators import non_dominated_sort
    np.random.seed(1)
    PopObj = np.random.rand(10, 2)
    
    # 转换为JAX数组
    PopObj_jax = jnp.array(PopObj)
    
    print("start NDSort compare")

    # 获取两个版本的输出
    # orders_old, no_old = NDSort_old(PopObj_jax, 5)
    orders_new, no_new = NDSort(PopObj_jax, 5)
    oder_evox = non_dominated_sort(PopObj_jax)
    # PopObj_torch = torch.tensor(PopObj)
    # orders_old, no_old = NDSort_torch(PopObj_torch, 5)
    # orders_old, no_old = jnp.array(orders_old), jnp.array(no_old)
    # 比较输出
    print(oder_evox-orders_new)
    # assert jnp.all(orders_old == orders_new), "Order arrays do not match"
    # assert no_old == no_new, "No values do not match"

    print("All tests passed!")


def test_CrowdingDistance():
    from evox.operators import crowding_distance
    import numpy as np
    import torch
    np.random.seed(2)
    PopObj = np.random.rand(100, 3)
    FrontNo = np.random.randint(0, 3, 100)
    maxF = 2
    
    # 转换为JAX数组
    PopObj_jax = jnp.array(PopObj)
    FrontNo_jax = jnp.array(FrontNo)
    
    print("start CrowdingDistance compare")

    # 获取两个版本的输出
    cd = CrowdingDistance(PopObj_jax, FrontNo_jax, maxF)
    cd_evox = crowding_distance(PopObj_jax, FrontNo==maxF)
    # cd_new = CrowdingDistance(PopObj_jax, FrontNo_jax, maxF)
    # PopObj_torch = torch.tensor(PopObj)
    # FrontNo_torch = torch.tensor(FrontNo)

    # 获取两个版本的输出
    # cd_torch = jnp.array(CrowdingDistance_torch(PopObj_torch, FrontNo_torch))
    
    # 比较输出
    # print(cd_evox-cd)
    assert jnp.all(cd == cd_evox), "Crowding distance arrays do not match"

    print("All tests passed!")

# 调用测试函数
# test_NDSort()
# test_CrowdingDistance()

# a = jnp.arange(10)-5
# print(jnp.where(a>0))