import jax
import jax.numpy as jnp

from utils import NDSort, CrowdingDistance, TournamentSelection
from evox.operators import mutation, crossover
from evox.operators.sampling import UniformSampling
from evox.utils import pairwise_euclidean_dist
from evox import Algorithm, State, jit_class, problems
from jax import jit
import math

class NSGA2Origin:
    def __init__(self, lb, ub, pop_size, n_objs, num_generation=1, problem=None, key=None, mutation_op=None, crossover_op=None):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.n_neighbor = jnp.ceil(self.pop_size / 10).astype(int)

        self.problem = problem
        self.mutation = jit(mutation_op) if mutation_op else jit(mutation.Polynomial((lb, ub)))
        self.crossover = jit(crossover_op) if crossover_op else jit(crossover.SimulatedBinary(type=2))
        self.key = key if key is not None else jax.random.PRNGKey(0)
        self.sample = UniformSampling(self.pop_size, self.n_objs)
        self.loop_num = num_generation # dfault: 10000

    def run(self):
        init_key, loop_key = jax.random.split(self.key)
        population = (
            jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        for i in range(self.loop_num):
            x_key, mut_key, loop_key = jax.random.split(loop_key, 3)
            # MatingPool = TournamentSelection(2, self.pop_size, FrontNo, CrowdDis)
            # mating_pop = population[MatingPool]
            mating_pop = population
            crossovered = self.crossover(x_key, mating_pop)
            offspring = self.mutation(mut_key, crossovered)
            next_generation = jnp.clip(offspring, self.lb, self.ub)
            population, FrontNo, CrowdDis = self.envSelect(jnp.vstack((population, next_generation)))

        return population

    def envSelect(self, population):
        PopObj, _ = self.problem.evaluate(State(), population)
        FrontNo, MaxNo = NDSort(PopObj, self.pop_size)
        Next = FrontNo < MaxNo
        CrowDis = CrowdingDistance(PopObj, FrontNo, MaxNo)
        Last = jnp.where(FrontNo == MaxNo)[0]
        rank = jnp.argsort(CrowDis[Last], descending=True)
        Next = Next.at[Last[rank[:self.pop_size-Next.sum()]]].set(True)
        selected = Next
        population = population[selected]
        FrontNo = FrontNo[selected]
        CrowdDis = CrowDis[selected]
        return population, FrontNo, CrowdDis

if __name__ == "__main__":
    pop_size = 100
    num_generations = 100
    key = jax.random.PRNGKey(42)
    n_objs = 3
    problem = problems.numerical.DTLZ2(m=3)
    lb = jnp.full(shape=(12,), fill_value=0)
    ub = jnp.full(shape=(12,), fill_value=1)

    moead = NSGA2Origin(lb, ub, pop_size, n_objs, num_generations, problem, key)
    moead.run()
