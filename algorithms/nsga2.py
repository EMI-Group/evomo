import jax
import jax.numpy as jnp

from utils import NDSort, CrowdingDistance, TournamentSelection
from evox.operators import mutation, crossover
from evox.operators.sampling import UniformSampling
from evox.utils import pairwise_euclidean_dist
from evox import Algorithm, State, jit_class
import math
from jax.experimental.host_callback import id_print


class nsga2:
    def __init__(self,  
        lb,
        ub,
        n_objs,
        pop_size,
        key = None,
        problem = None,
        mutation_op=None,
        loop_num = 10000,
        crossover_op=None,) -> None:
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.n_neighbor = jnp.ceil(self.pop_size / 10).astype(int)
        
        self.problem = problem
        self.mutation = mutation_op
        self.crossover = crossover_op
        self.key = key
        
        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary(type=2)
        if key is None:
            self.key = jax.random.PRNGKey(0)
            
        self.sample = UniformSampling(self.pop_size, self.n_objs)
        
    def fun(self):
        key, init_key, loop_key = jax.random.split(self.key,3)
        self.key = key
        population, _ = self.sample(key =init_key)
        x, FrontNo, CrowdDis = self.envSelect(population)
        for i in self.loop_num:
            x_key, mut_key = jax.random.split(loop_key,2)
            MatingPool = TournamentSelection(2,self.pop_size,FrontNo, CrowdDis)
            mating_pop = population[MatingPool]
            crossovered = self.crossover(x_key, mating_pop)
            offspring = self.mutation(mut_key, crossovered)
            population, FrontNo, CrowdDis = self.EnvSel(jnp.cat((population,offspring),0))
        return population
    
    def envSelect(self, population):
        PopObj, state = self.problem.evaluate(State(), population)
        FrontNo, MaxNo = NDSort(PopObj, self.pop_size)
        Next = FrontNo < MaxNo
        CrowDis = CrowdingDistance(PopObj, FrontNo)
        Last = jnp.nonzero(FrontNo == MaxNo)
        X, rank = jnp.sort(CrowDis[Last],descending=True)
        Next[Last[rank[:self.pop_size-Next.sum()]]] = True
        population = population[Next]
        FrontNo = FrontNo[Next]
        CrowDis = CrowDis[Next]
        return population, FrontNo, CrowDis

        
        
