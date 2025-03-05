# --------------------------------------------------------------------------------------
# 1. NSGA-III algorithm is described in the following papers:
#
# Title: An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting
# Approach, Part I: Solving Problems With Box Constraints
# Link: https://ieeexplore.ieee.org/document/6600851
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from evox.operators import (
    non_dominated_sort,
    selection,
    mutation,
    crossover,
    sampling,
)
from evox import Algorithm, jit_class, State


@jit_class
class NSGA3(Algorithm):
    """NSGA-III algorithm

    link: https://ieeexplore.ieee.org/document/6600851
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = selection.UniformRand(1)
        if self.mutation is None:
            self.mutation = mutation.Polynomial((self.lb, self.ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()
        self.sampling = sampling.UniformSampling(self.pop_size, self.n_objs)
        subkey = jax.random.PRNGKey(0)
        self.ref = self.sampling(subkey)[0]
        self.ref = self.ref / jnp.linalg.norm(self.ref, axis=1)[:, None]

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )

        #
        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            key=key,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    def ask(self, state):
        key, sel_key1, mut_key, sel_key2, x_key = jax.random.split(state.key, 5)
        mutated = self.mutation(mut_key, state.population)
        crossovered = self.crossover(x_key, mutated)
        next_generation = jnp.clip(crossovered, self.lb, self.ub)

        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        rank = rank[order]
        ranked_pop = merged_pop[order]
        ranked_fitness = merged_fitness[order]
        last_rank = rank[self.pop_size]
        ranked_fitness = jnp.where(
            jnp.repeat((rank <= last_rank)[:, None], self.n_objs, axis=1),
            ranked_fitness,
            jnp.nan,
        )

        # Normalize
        ideal = jnp.nanmin(ranked_fitness, axis=0)
        offset_fitness = ranked_fitness - ideal

        extreme = jnp.zeros(self.n_objs, dtype=int)
        weight = jnp.eye(self.n_objs) + 1e-6
        for i in range(self.n_objs):
            extreme = extreme.at[i].set(jnp.argmin(
                jnp.max(offset_fitness / jnp.tile(weight[i], (len(offset_fitness), 1)), axis=1), axis=0))

        def extreme_point(val):
            extreme, offset_fitness = val
            plane = jnp.linalg.solve(offset_fitness[extreme, :], jnp.ones(self.n_objs))
            intercept = 1 / plane
            return intercept

        def worst_point(val):
            return jnp.nanmax(ranked_fitness, axis=0)

        nadir_point = jax.lax.cond(
            jnp.linalg.matrix_rank(extreme) == self.n_objs,
            extreme_point,
            worst_point,
            (extreme, offset_fitness),
        )
        normalized_fitness = offset_fitness / nadir_point

        # def perpendicular_distance(x, y):
        #     dist = jnp.zeros((len(x), len(y)))
        #
        #     for i in range(len(x)):
        #         proj_len = jnp.dot(x[i], y.T) / jnp.sum(y * y, axis=1)
        #         proj_vec = proj_len[:, None] * y
        #         prep_vec = x[i] - proj_vec
        #         norm = jnp.linalg.norm(prep_vec, axis=1)
        #
        #         dist = dist.at[i].set(norm)
        #
        #     return dist
        #
        # dist = perpendicular_distance(ranked_fitness, self.ref)
        def perpendicular_distance(x, y):
            dist = jnp.zeros((len(x), len(y)))

            def outer_loop(i, dist):
                def inner_loop(j, dist):
                    proj_len = jnp.dot(x[i], y[j]) / jnp.dot(y[j], y[j])
                    proj_vec = proj_len * y[j]
                    prep_vec = x[i] - proj_vec
                    norm = jnp.linalg.norm(prep_vec)
                    return dist.at[i, j].set(norm)

                # Use fori_loop for the inner loop
                return jax.lax.fori_loop(0, len(y), inner_loop, dist)

            # Use fori_loop for the outer loop
            dist = jax.lax.fori_loop(0, len(x), outer_loop, dist)
            return dist

        # def perpendicular_distance(x, y):
        #     dist = jnp.zeros((len(x), len(y)))
        #
        #     def loop_body(i, dist):
        #         proj_len = jnp.dot(x[i], y.T) / jnp.sum(y * y, axis=1)
        #         proj_vec = proj_len[:, None] * y
        #         prep_vec = x[i] - proj_vec
        #         norm = jnp.linalg.norm(prep_vec, axis=1)
        #
        #         return dist.at[i].set(norm)
        #
        #     dist = jax.lax.fori_loop(0, len(x), loop_body, dist)
        #     return dist

        dist = perpendicular_distance(ranked_fitness, self.ref)

        pi = jnp.nanargmin(dist, axis=1)
        d = dist[jnp.arange(len(normalized_fitness)), pi]

        # Niche
        def niche_loop(val):
            def nope(val):
                idx, i, rho, j = val
                rho = rho.at[j].set(self.pop_size)
                return idx, i, rho, j

            def have(val):
                def zero(val):
                    idx, i, rho, j = val
                    idx = idx.at[i].set(jnp.nanargmin(jnp.where(pi == j, d, jnp.nan)))
                    rho = rho.at[j].add(1)
                    return idx, i + 1, rho, j

                def already(val):
                    idx, i, rho, j = val
                    key = jax.random.PRNGKey(i * j)
                    temp = jax.random.randint(
                        key, (1, len(ranked_pop)), 0, self.pop_size
                    )
                    temp = temp + (pi == j) * self.pop_size
                    idx = idx.at[i].set(jnp.argmax(temp))
                    rho = rho.at[j].add(1)
                    return idx, i + 1, rho, j

                return jax.lax.cond(rho[val[3]], already, zero, val)

            idx, i, rho = val
            j = jnp.argmin(rho)
            idx, i, rho, j = jax.lax.cond(
                jnp.sum(pi == j), have, nope, (idx, i, rho, j)
            )
            return idx, i, rho

        survivor_idx = jnp.arange(self.pop_size)
        rho = jnp.bincount(
            jnp.where(rank < last_rank, pi, len(self.ref)), length=len(self.ref)
        )
        pi = jnp.where(rank == last_rank, pi, -1)
        d = jnp.where(rank == last_rank, d, jnp.nan)
        survivor_idx, _, _ = jax.lax.while_loop(
            lambda val: val[1] < self.pop_size,
            niche_loop,
            (survivor_idx, jnp.sum(rho), rho),
        )

        state = state.update(
            population=ranked_pop[survivor_idx], fitness=ranked_fitness[survivor_idx]
        )
        return state
