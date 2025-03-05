from functools import partial

import jax
import jax.numpy as jnp
from evox.core.module import jit_class
from evox.operators import crossover, mutation
from evox.operators.sampling import UniformSampling
from evox.utils import pairwise_euclidean_dist
from evox import problems
import math
import time
from evox.metrics import IGD



@jit_class
class AggregationFunction:
    def __init__(self, function_name):
        if function_name == "pbi":
            self.function = self.pbi
        elif function_name == "tchebycheff":
            self.function = self.tchebycheff
        elif function_name == "tchebycheff_norm":
            self.function = self.tchebycheff_norm
        elif function_name == "modified_tchebycheff":
            self.function = self.modified_tchebycheff
        elif function_name == "weighted_sum":
            self.function = self.weighted_sum
        else:
            raise ValueError("Unsupported function")

    def pbi(self, f, w, z, *args):
        norm_w = jnp.linalg.norm(w, axis=1)
        f = f - z
        d1 = jnp.sum(f * w, axis=1) / norm_w
        d2 = jnp.linalg.norm(
            f - (d1[:, jnp.newaxis] * w / norm_w[:, jnp.newaxis]), axis=1
        )
        return d1 + 5 * d2

    def tchebycheff(self, f, w, z, *args):
        return jnp.max(jnp.abs(f - z) * w, axis=1)

    def tchebycheff_norm(self, f, w, z, z_max, *args):
        return jnp.max(jnp.abs(f - z) / (z_max - z) * w, axis=1)

    def modified_tchebycheff(self, f, w, z, *args):
        return jnp.max(jnp.abs(f - z) / w, axis=1)

    def weighted_sum(self, f, w, *args):
        return jnp.sum(f * w, axis=1)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class MOEAD:
    def __init__(
        self,
        lb,
        ub,
        pop_size,
        n_objs,
        num_generations,
        problem,
        key,
        uniform_init=True,
    ):
        self.lb = lb
        self.ub = ub
        self.num_variables = lb.shape[0]
        self.n_objs = n_objs
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.population = jnp.zeros((pop_size, self.num_variables))
        self.fitness = jnp.zeros((pop_size, self.n_objs))
        self.problem = problem
        self.key = key
        self.n_neighbor = 0
        self.func_name = "pbi"
        self.pf = self.problem.pf()
        self.uniform_init = uniform_init

        self.mutation = mutation.Polynomial((lb, ub))
        self.crossover = crossover.SimulatedBinary(type=1)
        self.sample = UniformSampling(self.pop_size, self.n_objs)
        self.aggregate_func = AggregationFunction(self.func_name)

    @partial(jax.jit, static_argnums=(0,))
    def initialize_population(self, key):
        initializer = jax.nn.initializers.glorot_normal()
        if self.uniform_init:
            return (
                jax.random.uniform(key, (self.pop_size, self.num_variables))
                * (self.ub - self.lb)
                + self.lb
            )
        else:
            return initializer(key, (self.pop_size, self.num_variables))

    @partial(jax.jit, static_argnums=(0,))
    def step_forloop(self, key, population, pop_obj, neighbors, w, z, state):
        for i in range(self.pop_size):
            key, subkey, cross_key, mut_key = jax.random.split(key, 4)
            p = neighbors[i]
            p = jax.random.permutation(subkey, p).astype(int)

            parent = population[p[:2]]

            crossovered = self.crossover(cross_key, parent)
            off = self.mutation(mut_key, crossovered)
            off = off[0].reshape(1, -1)
            off = jnp.clip(off, self.lb, self.ub)
            off_obj, state = self.problem.evaluate(state, off)
            z = jnp.min(jnp.vstack([z, off_obj]), axis=0)

            f_old = self.aggregate_func(pop_obj[p], w[p], z)
            f_new = self.aggregate_func(off_obj, w[p], z)

            update_condition = (f_old > f_new)[:, jnp.newaxis]
            population = population.at[p].set(
                jnp.where(
                    update_condition, jnp.tile(off, (jnp.shape(p)[0], 1)), population[p]
                )
            )
            pop_obj = pop_obj.at[p].set(
                jnp.where(
                    update_condition,
                    jnp.tile(off_obj, (jnp.shape(p)[0], 1)),
                    pop_obj[p],
                )
            )
        return key, population, pop_obj, z, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, population, pop_obj, neighbors, w, z, state):

        def body_fn(i, vals):
            key, population, pop_obj, z, state = vals
            key, subkey, cross_key, mut_key = jax.random.split(key, 4)
            p = neighbors[i]
            p = jax.random.permutation(subkey, p).astype(int)

            parent = population[p[:2]]

            crossovered = self.crossover(cross_key, parent)
            off = self.mutation(mut_key, crossovered)
            off = off[0].reshape(1, -1)
            off = jnp.clip(off, self.lb, self.ub)
            off_obj, state = self.problem.evaluate(state, off)
            z = jnp.min(jnp.vstack([z, off_obj]), axis=0)

            f_old = self.aggregate_func(pop_obj[p], w[p], z)
            f_new = self.aggregate_func(off_obj, w[p], z)

            update_condition = (f_old > f_new)[:, jnp.newaxis]
            population = population.at[p].set(
                jnp.where(
                    update_condition, jnp.tile(off, (jnp.shape(p)[0], 1)), population[p]
                )
            )
            pop_obj = pop_obj.at[p].set(
                jnp.where(
                    update_condition,
                    jnp.tile(off_obj, (jnp.shape(p)[0], 1)),
                    pop_obj[p],
                )
            )
            return key, population, pop_obj, z, state

        vals = (key, population, pop_obj, z, state)

        key, population, pop_obj, z, state = jax.lax.fori_loop(
            0, self.pop_size, body_fn, vals
        )

        return key, population, pop_obj, z, state

    def run(self):
        self.key, key = jax.random.split(self.key, 2)
        key, subkey1, subkey2 = jax.random.split(key, 3)
        state = self.problem.init(self.key)
        w, _ = self.sample(subkey1)
        self.pop_size = w.shape[0]

        self.n_neighbor = int(math.ceil(self.pop_size / 10))

        self.population = self.initialize_population(subkey2)
        self.pop_obj, state = self.problem.evaluate(state, self.population)

        neighbors = pairwise_euclidean_dist(w, w)
        neighbors = jnp.argsort(neighbors, axis=1)
        neighbors = neighbors[:, : self.n_neighbor]

        z = jnp.min(self.pop_obj, axis=0)

        igd = IGD(self.pf)
        start = time.time()
        for gen in range(self.num_generations):
            print(gen)
            key, self.population, self.pop_obj, z, state = self.step(
                key, self.population, self.pop_obj, neighbors, w, z, state
            )

            igd_value = igd(self.pop_obj)
            print(igd_value)
        end = time.time()
        print(end - start)


if __name__ == "__main__":
    pop_size = 100
    num_generations = 100
    key = jax.random.PRNGKey(42)
    n_objs = 3
    problem = problems.numerical.DTLZ2(m=3)
    lb = jnp.full(shape=(12,), fill_value=0)
    ub = jnp.full(shape=(12,), fill_value=1)

    moead = MOEAD(lb, ub, pop_size, n_objs, num_generations, problem, key)
    moead.run()
