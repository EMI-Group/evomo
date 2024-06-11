import jax
import jax.numpy as jnp

from algorithms.utils import NDSort, CrowdingDistance, TournamentSelection
from evox.operators import mutation, crossover, non_dominated_sort, selection, sampling
from evox.utils import cos_dist
from evox import Algorithm, State, jit_class
from jax import jit
import time

class NSGA3Origin2(Algorithm):

    def __init__(
        self,
        lb,
        ub,
        pop_size,
        n_objs,
        uniform_init=True,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.n_neighbor = jnp.ceil(self.pop_size / 10).astype(int)
        self.uniform_init = uniform_init

        self.mutation = mutation_op if mutation_op else mutation.Polynomial((lb, ub))
        self.crossover = crossover_op if crossover_op else crossover.SimulatedBinary(type=2)
        self.selection = selection_op if selection_op else selection.UniformRand(1)
        self.sampling = sampling.UniformSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey = jax.random.split(key)
        initializer = jax.nn.initializers.glorot_normal()
        if self.uniform_init:
            population = (
                jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
                * (self.ub - self.lb)
                + self.lb
            )
        else:
            population = initializer(subkey, shape=(self.pop_size, self.dim))
        ref = self.sampling(subkey)[0]

        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            ref=ref,
            key=key,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state
    
    def ask(self, state):
        key, mut_key, x_key = jax.random.split(state.key, 3)
        crossovered = self.crossover(x_key, state.population)
        next_generation = self.mutation(mut_key, crossovered)
        next_generation = jnp.clip(next_generation, self.lb, self.ub)

        return next_generation, state.update(next_generation=next_generation, key=key)
        
    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

       
        FrontNo = non_dominated_sort(merged_pop)
        order = jnp.argsort(FrontNo)
        MaxNo = FrontNo[order[self.pop_size]]
        Next = FrontNo < MaxNo
        Last = jnp.where(FrontNo == MaxNo)[0]
        
        # 进行最后一轮选择
        select_key, key = jax.random.split(state.key)
        Choose = LastSelection(merged_fitness[Next], merged_fitness[Last], self.pop_size - jnp.sum(Next), state.ref, select_key)
        Next = Next.at[Last[Choose]].set(True)
        
        state = state.update(
            population=merged_pop[Next],
            fitness=merged_fitness[Next],
            key=key
        )
        return state
        

def LastSelection(PopObj1, PopObj2, K, Z, key):
    PopObj = jnp.vstack([PopObj1, PopObj2])
    Zmin = jnp.min(PopObj, axis=0)
    PopObj = PopObj - Zmin
    N, M = PopObj.shape
    N1 = PopObj1.shape[0]
    N2 = PopObj2.shape[0]
    NZ = Z.shape[0]

    # 原版 归一化
    Extreme = jnp.zeros(M, dtype=int)
    w = jnp.eye(M) + 1e-6
    # for i in range(M):
    #     Extreme = Extreme.at[i].set(jnp.argmin(jnp.max(PopObj / w[i], axis=1), axis=0))

    def get_streme(i, Extreme):
        Extreme = Extreme.at[i].set(jnp.argmin(jnp.max(PopObj / w[i], axis=1), axis=0))
        return Extreme

    Extreme = jax.lax.fori_loop(0, M, get_streme, Extreme)

    ''' 
    #改版，更快一点点
    #(100, 500) objedct: 3 iteration: 100 igd: 19.244148
    #原版 nsga3:
    #time: 863.1117389202118
    #改版 nsga3:
    #time: 782.9815998077393
    
    
    max_indices = jnp.argmax(PopObj, axis=0)
    unique_indices, counts = jnp.unique(max_indices, return_counts=True)

    # 如果有重复的最大值索引，为重复索引的列找次大值
    while jnp.any(counts > 1):
        for index in unique_indices[counts > 1]:
            # 找出具有相同最大行索引的所有列
            columns = jnp.where(max_indices == index)[0]
            # 对每一列处理，除了第一列
            for col in columns[1:]:
                # 设置当前最大值所在行的值为负无穷大，寻找次大值
                temp_matrix = PopObj.at[index, col].set(-jnp.inf)
                # 计算次大值的索引
                new_index = jnp.argmax(temp_matrix[:, col], axis=0)
                # 更新索引数组
                max_indices = max_indices.at[col].set(new_index)

        # 重新检查是否还有重复
        unique_indices, counts = jnp.unique(max_indices, return_counts=True)
    Extreme = max_indices
    '''

    Hyperplane = jnp.linalg.solve(PopObj[Extreme, :], jnp.ones(M))
    a = jnp.max(PopObj, axis=0) if jnp.any(jnp.isnan(Hyperplane)) else 1.0 / Hyperplane
    PopObj = PopObj / a

    # 关联每个解到参考点
    cos_distance = cos_dist(PopObj, Z)

    dist = jnp.linalg.norm(PopObj, axis=-1, keepdims=True) * jnp.sqrt(
        1 - cos_distance**2
    )
    pi = jnp.argmin(dist, axis=1)
    dist_point = dist[jnp.arange(N), pi]
    rho = jnp.bincount(pi[:N1], length=NZ)

    # 环境选择
    Choose = jnp.zeros(N2, dtype=bool)
    Zchoose = jnp.ones(NZ, dtype=bool)
    while jnp.sum(Choose) < K:
        Temp = jnp.where(Zchoose)[0]
        Jmin = jnp.argmin(rho[Temp])
        j = Temp[Jmin]
        I = jnp.where((Choose == 0) & (pi[N1:] == j))[0]
        if I.shape[0] > 0:
            if rho[j] == 0:
                s = jnp.argmin(dist_point[N1 + I])
            else:
                key, choice_key = jax.random.split(key)
                s = jax.random.choice(choice_key, I.shape[0])
            Choose = Choose.at[I[s]].set(True)
            rho = rho.at[j].add(1)
        else:
            Zchoose = Zchoose.at[j].set(False)

    return Choose
