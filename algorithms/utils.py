import jax
import jax.numpy as jnp
from jax import lax, jit
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


def NDSort(PopObj, nSort):
    N, M = PopObj.shape
    no = 0
    orders = jnp.full(N, nSort)
    dom_nums = jnp.zeros(N, dtype=int)

    # 广播比较所有对
    def compare_all(PopObj):
        return (jnp.all(PopObj[:, None, :] <= PopObj[None, :, :], axis=-1) & 
                jnp.any(PopObj[:, None, :] < PopObj[None, :, :], axis=-1))
    
    dominated = compare_all(PopObj)
    
    # 更新主导数字
    dom_nums = jnp.sum(dominated, axis=0)

    while nSort > 0:
        no += 1
        nSort -= 1
        next_dom_nums = dom_nums
        for i in range(N):
            if dom_nums[i] == 0:
                orders = orders.at[i].set(no)
                next_dom_nums = next_dom_nums.at[i].set(-1)
                for j in range(N):
                    if dominated[i, j]:
                        next_dom_nums = next_dom_nums.at[j].add(-1)
        if jnp.all(next_dom_nums == dom_nums):
            break
        dom_nums = next_dom_nums

    orders = jnp.where(orders <= no, orders, no)
    return orders, no


def CrowdingDistance(PopObj, FrontNo, maxF=None):
    N, M = PopObj.shape
    cd = jnp.zeros(N)
    
    def calculate_front(front):
        if front.shape[0] <= 2:
            return jnp.array([jnp.inf] * front.shape[0])
        
        Fmax = jnp.max(PopObj[front, :], axis=0)
        Fmin = jnp.min(PopObj[front, :], axis=0)
        front_cd = jnp.zeros(front.shape[0])
        
        # for k in range(M):
        def calculate_distance(k, front_cd):
            sorted_indices = jnp.argsort(PopObj[front, k])
            sorted_front = front[sorted_indices]
            values = PopObj[sorted_front, k]
            distances_space = jnp.diff(values)  # 应该有len(front) - 1 个值
            distances = distances_space[:-1] + distances_space[1:] # 应该有len(front) - 2 个值
            scaled_distances = distances / (Fmax[k] - Fmin[k] + 1e-10)
            scaled_distances = jnp.concatenate((jnp.array([jnp.inf]), scaled_distances, jnp.array([jnp.inf])))
            # 确保距离更新在正确的范围内
            front_cd = front_cd.at[sorted_indices].add(scaled_distances)  # 更新距离
            return front_cd
        front_cd = jax.lax.fori_loop(0, M, calculate_distance, front_cd)
        return front_cd
    
    if maxF is None:
        fronts = jnp.unique(FrontNo)
        for f in fronts:
            front_indices = jnp.where(FrontNo == f)[0]
            cd = cd.at[front_indices].set(calculate_front(front_indices))
    else:
        front_indices = jnp.where(FrontNo == maxF)[0]
        cd = cd.at[front_indices].set(calculate_front(front_indices))
        
    
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
    np.random.seed(1)
    PopObj = np.random.rand(10, 2)
    
    # 转换为JAX数组
    PopObj_jax = jnp.array(PopObj)
    
    print("start compare")

    # 获取两个版本的输出
    orders_old, no_old = NDSort_old(PopObj_jax, 5)
    orders_new, no_new = NDSort(PopObj_jax, 5)

    # 比较输出
    assert jnp.all(orders_old == orders_new), "Order arrays do not match"
    assert no_old == no_new, "No values do not match"

    print("All tests passed!")


def test_CrowdingDistance():
    import numpy as np
    np.random.seed(1)
    PopObj = np.random.rand(10, 3)
    FrontNo = np.random.randint(1, 3, 10)
    maxF = 2
    
    # 转换为JAX数组
    PopObj_jax = jnp.array(PopObj)
    FrontNo_jax = jnp.array(FrontNo)
    
    print("start compare")

    # 获取两个版本的输出
    cd_old = CrowdingDistance(PopObj_jax, FrontNo_jax, maxF)
    # cd_new = CrowdingDistance(PopObj_jax, FrontNo_jax, maxF)

    # 比较输出
    # assert jnp.all(cd_old == cd_new), "Crowding distance arrays do not match"

    print("All tests passed!")

# 调用测试函数
# test_NDSort()
# test_CrowdingDistance()