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
from evox.utils import cos_dist
from jax.experimental.host_callback import id_print
from jax import vmap
from jax.numpy import newaxis

# @jit_class
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
        self.uniform_init = uniform_init

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
        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        last_rank = rank[order[self.pop_size]]
        the_selected_one_idx = order[0]
        the_drop_one_idx = order[-1]
        selected_number = (rank < last_rank).sum()
        ranked_fitness = jnp.where(
            (rank <= last_rank)[:, None],
            merged_fitness,
            jnp.nan,
        )

        # Normalize
        ideal_points = jnp.nanmin(ranked_fitness, axis=0)
        ranked_fitness = ranked_fitness - ideal_points
        weight = jnp.eye(self.n_objs) + 1e-6

        def get_extreme(i):
            return jnp.nanargmin(jnp.nanmax(ranked_fitness / weight[i], axis=1))

        extreme_ind = jax.vmap(get_extreme)(jnp.arange(self.n_objs))
        extreme = ranked_fitness[extreme_ind]

        def get_intercept(val):
            # Calculate the intercepts of the hyperplane constructed by the extreme points
            _extreme = val[0]
            plane = jnp.linalg.solve(_extreme, jnp.ones(self.n_objs))
            intercept = 1 / plane
            return intercept

        def worst_intercept(val):
            _ranked_fitness = val[1]
            return jnp.nanmax(_ranked_fitness, axis=0)

        nadir_point = jax.lax.cond(
            jnp.linalg.matrix_rank(extreme) == self.n_objs,
            get_intercept,
            worst_intercept,
            (extreme, ranked_fitness),
        )

        normalized_fitness = ranked_fitness / nadir_point
        cos_distance = cos_dist(normalized_fitness, state.ref)
        dist = jnp.linalg.norm(normalized_fitness, axis=-1, keepdims=True) * jnp.sqrt(
            1 - cos_distance**2
        )
        # Associate each solution with its nearest reference point
        group_id = jnp.nanargmin(dist, axis=1)
        group_id = jnp.where(group_id == -1, len(state.ref), group_id)
        group_dist = jnp.nanmin(dist, axis=1)
        rho = jnp.bincount(
            jnp.where(rank < last_rank, group_id, len(state.ref)), length=len(state.ref)
        )
        rho_last = jnp.bincount(
            jnp.where(rank == last_rank, group_id, len(state.ref)), length=len(state.ref)
        )
        group_id = jnp.where(rank == last_rank, group_id, jnp.inf)
        group_dist = jnp.where(rank == last_rank, group_dist, jnp.inf)
        selected_number = jnp.sum(rho)
        rho = jnp.where(rho_last == 0, jnp.inf, rho)
        keys = jax.random.split(state.key, self.pop_size + 1)

        def select_loop(vals):
            num, rho_level, rho, rank, last_index = vals
            selected_rho = rho == rho_level
            dists = jnp.where((rank == last_rank)[:, jnp.newaxis], dist, jnp.inf)
            selected_idx = jnp.argmin(dists, axis=0)
            index = jnp.where(selected_rho, selected_idx, jnp.inf)
            rho_level += 1
            rho = jnp.where(selected_rho, rho_level, rho)
            real_index = jnp.where(jnp.isinf(selected_idx), the_selected_one_idx, index)

            def update_rank(rank, idx):
                rank = rank.at[idx].set(last_rank - 1)
                return rank, idx

            rank, _ = jax.lax.scan(update_rank, rank, real_index)
            last_num = selected_rho.sum()
            num += last_num
            return num, rho_level, rho, rank, index

        selected_number, rho_level, rho, rank, last_index = jax.lax.while_loop(
            lambda val: val[0] < self.pop_size,
            select_loop,
            (selected_number, 0, rho, rank, rho),
        )

        def cut_mask(rank, dif, mask_index):
            sorted_index = jnp.sort(mask_index, descending=False)
            index = jnp.where(
                jnp.arange(sorted_index.size) < dif, sorted_index, the_drop_one_idx
            )

            def _update_rank(rank, idx):
                rank = rank.at[idx].set(last_rank)
                return rank, idx

            rank, _ = jax.lax.scan(_update_rank, rank, index)
            return rank 

        dif = selected_number - self.pop_size
        rank = cut_mask(rank, dif, last_index)

        selected_idx = jnp.sort(
            jnp.where(rank < last_rank, jnp.arange(ranked_fitness.shape[0]), jnp.inf)
        )[: self.pop_size].astype(jnp.int32)
        state = state.update(
            population=merged_pop[selected_idx],
            fitness=merged_fitness[selected_idx],
            key=keys[0],
        )
        return state

    """
     def select_loop(vals):
            selected_number, rank, group_id, rho, rho_last = vals
            group = jnp.argmin(rho)
            candidates = jnp.where(group_id == group, group_dist, jnp.inf)

            def get_rand_candidate(candidates):
                order = jnp.sort(
                    jnp.where(
                        jnp.isinf(candidates), jnp.inf, jnp.arange(candidates.size)
                    )
                )
                rand_index = jax.random.randint(
                    keys[selected_number], (), 0, rho_last[group]
                )
                return order[rand_index].astype(jnp.int32)

            def get_min_candidate(candidates):
                return jnp.argmin(candidates)

            candidate = jax.lax.cond(
                (rho[group] == 0) | (rho_last[group] == 1),
                get_min_candidate,
                get_rand_candidate,
                candidates,
            )
            rank = rank.at[candidate].set(last_rank - 1)
            group_id = group_id.at[candidate].set(jnp.nan)
            rho_last = rho_last.at[group].set(rho_last[group] - 1)

            def update_(vals):
                idx, matrix = vals
                return matrix.at[idx].set(jnp.inf)

            def add_(vals):
                idx, matrix = vals
                return matrix.at[idx].set(matrix[idx] + 1)

            rho = jax.lax.cond(rho_last[group] == 0, update_, add_, (group, rho))
            selected_number += 1
            return selected_number, rank, group_id, rho, rho_last

        selected_number, rank, group_id, rho, rho_last = jax.lax.while_loop(
            lambda val: jnp.nansum(val[0]) < self.pop_size,
            select_loop,
            (selected_number, rank, group_id, rho, rho_last),
        )

    """
