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
        key, shuffled_key = jax.random.split(state.key)
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)
        shuffled_idx = jax.random.permutation(shuffled_key, merged_pop.shape[0])
        merged_pop = merged_pop[shuffled_idx]
        merged_fitness = merged_fitness[shuffled_idx]
        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        last_rank = rank[order[self.pop_size]]
        the_selected_one_idx = order[0]
        ranked_fitness = jnp.where(
            (rank <= last_rank)[:, None],
            merged_fitness,
            jnp.nan,
        )
        # Normalize
        ideal_points = jnp.nanmin(ranked_fitness, axis=0)
        ranked_fitness = ranked_fitness - ideal_points
        weight = jnp.eye(self.n_objs) + 1e-6
        def get_extreme(w):
            return jnp.nanargmin(jnp.nanmax(ranked_fitness / w, axis=1))
        extreme_ind = jax.vmap(get_extreme)(weight)
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
        # dist is matrix with shape is (merged_pop_size, ref_num)
        dist = jnp.linalg.norm(normalized_fitness, axis=-1, keepdims=True) * jnp.sqrt(
            1 - cos_distance ** 2
        )
        # Associate each solution with its nearest reference point
        group_id = jnp.nanargmin(dist, axis=1)
        group_dist = jnp.nanmin(dist, axis=1)
        group_id = jnp.where(group_id == -1, len(state.ref), group_id)
        rho = jnp.bincount(
            jnp.where(rank < last_rank, group_id, len(state.ref)), length=len(state.ref)
        )
        rho_last = jnp.bincount(
            jnp.where(rank == last_rank, group_id, len(state.ref)), length=len(state.ref)
        )
        selected_number = jnp.sum(rho)
        upper_bound = self.pop_size * 2 + len(state.ref)
        rho = jnp.where(rho_last == 0, upper_bound, rho)
        group_id = jnp.where(rank == last_rank, group_id, upper_bound)
        # for rho == 0
        rho_level = 0
        # selected_rho = rho == rho_level
        selected_rho = jnp.where(rho == rho_level, jnp.arange(rho.size), upper_bound)
        def select_from_index_by_min(idx):
            return jax.lax.cond(
                idx == upper_bound,
                lambda : upper_bound,
                lambda : jnp.argmin(jnp.where(group_id == idx, group_dist, upper_bound)),
            )
        # use vmap to vectorize the selection
        selected_idx = jax.vmap(select_from_index_by_min)(selected_rho)
        the_selected_one_idx = jnp.minimum(jnp.min(selected_idx), the_selected_one_idx)
        # selected_idx = jnp.where(rho == rho_level, selected_idx, the_selected_one_idx).astype(jnp.int32)
        selected_idx = jnp.where(selected_idx == upper_bound, the_selected_one_idx, selected_idx)
        rho_last = jnp.where(rho == rho_level, rho_last - 1, rho_last)
        last_num = jnp.sum(rho == rho_level)
        rho = jnp.where(rho == rho_level, rho_level + 1, rho)
        rho = jnp.where(rho_last == 0, upper_bound, rho)
        rank = rank.at[selected_idx].set(last_rank - 1)
        selected_number += last_num
        
        def cut_mask(vals):
            rank, dif, mask_index = vals
            dif = -dif
            mask_index = jnp.where(mask_index == the_selected_one_idx, upper_bound, mask_index)
            sorted_index = jnp.sort(mask_index)
            the_drop_one_idx = sorted_index[0]
            index = jnp.where(
                jnp.arange(sorted_index.size) < dif, sorted_index, the_drop_one_idx
            )
            rank = rank.at[index].set(last_rank)
            return rank
        
        def add_dif(vals):
            rank, dif, selected_idx = vals
            selected_idx = jnp.where(rank == last_rank, jnp.arange(rank.size), upper_bound)
            sorted_index = jnp.sort(selected_idx)
            sorted_index = jnp.where(
                jnp.arange(sorted_index.size) < dif, sorted_index, the_selected_one_idx
            )
            rank = rank.at[sorted_index].set(last_rank - 1)
            return rank

        dif = self.pop_size - selected_number
        rank = jax.lax.cond(dif < 0, cut_mask, add_dif, (rank, dif, selected_idx))
        selected_idx = jnp.sort(
            jnp.where(rank < last_rank, jnp.arange(ranked_fitness.shape[0]), upper_bound)
        )[: self.pop_size]
        state = state.update(
            population=merged_pop[selected_idx],
            fitness=merged_fitness[selected_idx],
            key=key,
        )
        return state