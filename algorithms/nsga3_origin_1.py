import jax
import jax.numpy as jnp

from utils import NDSort, CrowdingDistance, TournamentSelection
from evox.operators import mutation, crossover
from evox.operators.sampling import UniformSampling
from evox.utils import cos_dist
from evox import Algorithm, State, jit_class
from jax import jit

class NSGA3Origin:

    def __init__(
        self,
        lb,
        ub,
        pop_size,
        n_objs,
        num_generation=1000,
        problem=None,
        key=None,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.n_neighbor = jnp.ceil(self.pop_size / 10).astype(int)

        self.problem = problem
        self.mutation = mutation_op if mutation_op else mutation.Polynomial((lb, ub))
        self.crossover = crossover_op if crossover_op else crossover.SimulatedBinary(type=2)
        self.key = key if key is not None else jax.random.PRNGKey(0)
        self.sample = jit(UniformSampling(self.pop_size, self.n_objs))
        self.loop_num = num_generation
        self.ref = self.sample(self.key)[0]
        # dfault: 10000

    def run(self):
        key, init_key, loop_key = jax.random.split(self.key, 3)
        self.key = key
        population = (
            jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        for i in range(self.loop_num):
            env_key, x_key, mut_key, loop_key = jax.random.split(loop_key, 4)
            # MatingPool = TournamentSelection(2, self.pop_size, FrontNo, CrowdDis)
            # mating_pop = population[MatingPool]
            mating_pop = population
            crossovered = self.crossover(x_key, mating_pop)
            offspring = self.mutation(mut_key, crossovered)
            population = self.envSelect(jnp.vstack((population, offspring)), env_key)
        return population

    def envSelect(self, population, key):
        PopObj, _ = self.problem.evaluate(State(), population)
        FrontNo, MaxNo = NDSort(PopObj, self.pop_size)
        Next = FrontNo < MaxNo 
        Last = jnp.where(FrontNo == MaxNo)[0]
        # 进行最后一轮选择
        Choose = LastSelection(PopObj[Next], PopObj[Last], self.pop_size - jnp.sum(Next), self.ref, key)
        Next = Next.at[Last[Choose]].set(True)
        return population[Next]


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

    def get_streme(i, Extreme):
        Extreme = Extreme.at[i].set(jnp.argmin(jnp.max(PopObj / w[i], axis=1), axis=0))
        return Extreme
    
    Extreme = jax.lax.fori_loop(0, M, get_streme, Extreme)
   
    Hyperplane = jnp.linalg.solve(PopObj[Extreme, :], jnp.ones(M))
    a = 1.0 / Hyperplane
    a = jnp.where(jnp.isnan(a), jnp.max(PopObj, axis=0), a)
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
