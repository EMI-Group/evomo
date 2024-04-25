# --------------------------------------------------------------------------------------
# 1. MOEA/D algorithm is described in the following papers:
#
# Title: MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition
# Link: https://ieeexplore.ieee.org/document/4358754
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import mutation, crossover
from evox.operators.sampling import UniformSampling
from evox.utils import pairwise_euclidean_dist
from evox import Algorithm, State, jit_class
import math
from jax.experimental.host_callback import id_print

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
        d2 = jnp.linalg.norm(f - (d1[:, jnp.newaxis] * w / norm_w[:, jnp.newaxis]), axis=1)
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

@jit_class
class TensorMOEAD(Algorithm):
    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        aggregate_op=['pbi','pbi'],
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.n_neighbor = jnp.ceil(self.pop_size / 10).astype(int)

        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary(type=2)
        self.sample = UniformSampling(self.pop_size, self.n_objs)

        self.aggregate_func1 = AggregationFunction(aggregate_op[0])
        self.aggregate_func2 = AggregationFunction(aggregate_op[1])


    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        w, _ = self.sample(subkey2)
        self.pop_size = w.shape[0]
        self.n_neighbor = int(math.ceil(self.pop_size / 10))

        population = (
                jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
                * (self.ub - self.lb)
                + self.lb
        )
        neighbors = pairwise_euclidean_dist(w, w)
        neighbors = jnp.argsort(neighbors, axis=1)
        neighbors = neighbors[:, : self.n_neighbor]

        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            weight_vector=w,
            neighbors=neighbors,
            z=jnp.zeros(shape=self.n_objs),
            parent=jnp.zeros((self.pop_size, self.n_neighbor)).astype(int),
            key=key,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        z = jnp.min(fitness, axis=0)
        state = state.update(fitness=fitness, z=z)
        return state

    def ask(self, state):
        key, subkey, sel_key, mut_key = jax.random.split(state.key, 4)
        parent = jax.random.permutation(
            subkey, state.neighbors, axis=1, independent=True
        ).astype(int)
        population = state.population
        selected_p = jnp.r_[population[parent[:, 0]], population[parent[:, 1]]]

        crossovered = self.crossover(sel_key, selected_p)
        next_generation = self.mutation(mut_key, crossovered)
        next_generation = jnp.clip(next_generation, self.lb, self.ub)

        return next_generation, state.update(
            next_generation=next_generation, parent=parent, key=key
        )

    def tell(self, state, fitness):
        population = state.population
        pop_obj = state.fitness
        offspring = state.next_generation
        obj = fitness
        w = state.weight_vector
        z = state.z
        # parent = state.parent
        neighbor = state.neighbors

        z_min = jnp.minimum(z, jnp.min(obj, axis=0))
        z_max = jnp.max(pop_obj, axis=0)

        sub_pop_indices = jnp.arange(0, self.pop_size)

        def body(ind_p, ind_obj):
            g_old = self.aggregate_func1(pop_obj[ind_p], w[ind_p], z_min, z_max)
            g_new = self.aggregate_func1(jnp.tile(ind_obj, (self.n_neighbor, 1)), w[ind_p], z_min, z_max)

            neighbor_indices = jnp.where(g_old > g_new, ind_p, -1)
            sub = sub_pop_indices.at[neighbor_indices].set(
                jnp.where(neighbor_indices == -1, sub_pop_indices[neighbor_indices], -1))
            return sub

        indices_cube = jax.vmap(body, in_axes=(0, 0))(neighbor, obj)

        def body2(sub_indices, population, pop_obj):
            f = jnp.where(sub_indices[:, jnp.newaxis] == -1, obj, pop_obj)
            x = jnp.where(sub_indices[:, jnp.newaxis] == -1, offspring, population)
            idx = jnp.argmin(self.aggregate_func2(f, w, z_min, z_max))
            return x[idx], f[idx]

        population, pop_obj = jax.vmap(body2, in_axes=(1, 0, 0))(indices_cube, population, pop_obj)

        state = state.update(population=population, fitness=pop_obj, z=z_min)
        return state
