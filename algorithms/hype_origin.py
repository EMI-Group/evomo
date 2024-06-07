# --------------------------------------------------------------------------------------
# 1. HypE algorithm is described in the following papers:
#
# Title: HypE: An Algorithm for Fast Hypervolume-Based Many-Objective Optimization
# Link: https://direct.mit.edu/evco/article-abstract/19/1/45/1363/HypE-An-Algorithm-for-Fast-Hypervolume-Based-Many
#
# 2. This code has been inspired by PlatEMO.
# More information about PlatEMO can be found at the following URL:
# GitHub Link: https://github.com/BIMK/PlatEMO
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from functools import partial

from evox import jit_class, Algorithm, State
from evox.operators import selection, mutation, crossover, non_dominated_sort
from jax.experimental.host_callback import id_print

@partial(jax.jit, static_argnums=[0, 1])
def calculate_alpha(N, k):
    alpha = jnp.zeros(N)

    for i in range(1, k + 1):
        num = jnp.prod((k - jnp.arange(1, i)) / (N - jnp.arange(1, i)))
        alpha = alpha.at[i-1].set(num / i)
    return alpha


@partial(jax.jit, static_argnums=[2, 3])
def calculate_hv_foriloop(points, ref, k, n_sample, key):
    n, m = jnp.shape(points)
    alpha = calculate_alpha(n, k)

    f_min = jnp.min(points, axis=0)

    s = jax.random.uniform(key, shape=(n_sample, m), minval=f_min, maxval=ref)

    pds = jnp.zeros((n, n_sample), dtype=bool)
    ds = jnp.zeros((n_sample, ))

    def body_fun1(i, vals):
        pds, ds = vals
        x = jnp.sum((jnp.tile(points[i, :], (n_sample, 1)) - s) <= 0, axis=1) == m
        pds = pds.at[i].set(jnp.where(x, True, pds[i]))
        ds = jnp.where(x, ds+1, ds)
        return pds, ds

    pds, ds = jax.lax.fori_loop(0, n, body_fun1, (pds, ds))
    ds = ds - 1

    f = jnp.zeros((n,))

    def body_fun2(i, val):
        temp = jnp.where(pds[i, :], ds, -1).astype(int)
        value = jnp.where(temp!=-1, alpha[temp], 0)
        value = jnp.sum(value)
        val = val.at[i].set(value)
        return val

    f = jax.lax.fori_loop(0, n, body_fun2, f)
    f = f * jnp.prod(ref - f_min) / n_sample

    return f


def calculate_hv_for(points, bounds, k, n_sample, key):
    N, M = points.shape
    alpha = jnp.zeros(N)
    indices = jnp.arange(1, k + 1)
    for i in range(1, k + 1):
        j = jnp.arange(1, i)
        alpha = alpha.at[i - 1].set(jnp.prod((k - j) / (N - j)) / i)
    f_min = jnp.min(points, axis=0)
    s = jax.random.uniform(key, shape=(n_sample, M), minval=f_min, maxval=bounds)
    pd_s = jnp.zeros((N, n_sample), dtype=bool)
    d_s = jnp.zeros(n_sample, dtype=int)

    for i in range(N):
        x = jnp.all(points[i] - s <= 0, axis=1)
        pd_s = pd_s.at[i].set(x)
        d_s = d_s + x.astype(jnp.int32)

    f = jnp.zeros(N)
    for i in range(N):
        temp = jnp.where(pd_s[i, :], d_s, -1).astype(int)
        value = jnp.sum(jnp.where(temp != -1, alpha[temp], 0))
        f = f.at[i].set(value)
    f = f * jnp.prod(bounds - f_min) / n_sample
    return f

class HypEOrigin(Algorithm):
    """HypE algorithm

    link: https://direct.mit.edu/evco/article-abstract/19/1/45/1363/HypE-An-Algorithm-for-Fast-Hypervolume-Based-Many
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        n_sample=10000,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.n_sample = n_sample

        self.mutation = mutation_op
        self.crossover = crossover_op
        self.selection = selection.Tournament(
            n_round=self.pop_size, multi_objective=True
        )
        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()

    def setup(self, key):
        key, subkey = jax.random.split(key)
        population = (
            jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
            * (self.ub - self.lb)
            + self.lb
        )
        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            ref_point=jnp.zeros((self.n_objs,)),
            key=key,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        ref_point = jnp.zeros((self.n_objs,)) + jnp.max(fitness) * 1.2
        state = state.update(fitness=fitness, ref_point=ref_point)
        return state

    def ask(self, state):
        population = state.population
        pop_obj = state.fitness
        key, subkey, sel_key, x_key, mut_key = jax.random.split(state.key, 5)
        # hv = cal_hv(pop_obj, state.ref_point, self.pop_size, self.n_sample, subkey)
        hv = calculate_hv_foriloop(pop_obj, state.ref_point, self.pop_size, self.n_sample, subkey)

        selected, _ = self.selection(sel_key, population, -hv)
        crossovered = self.crossover(x_key, selected)
        next_generation = self.mutation(mut_key, crossovered)
        next_generation = jnp.clip(next_generation, self.lb, self.ub)

        return next_generation, state.update(next_generation=next_generation)

    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_obj = jnp.concatenate([state.fitness, fitness], axis=0)

        n = jnp.shape(merged_pop)[0]

        rank = non_dominated_sort(merged_obj)
        order = jnp.argsort(rank)
        worst_rank = rank[order[n - 1]]
        mask = rank == worst_rank

        key, subkey = jax.random.split(state.key)
        hv = calculate_hv_foriloop(merged_obj, state.ref_point, n, self.n_sample, subkey)

        dis = jnp.where(mask, hv, -jnp.inf)

        combined_indices = jnp.lexsort((-dis, rank))[: self.pop_size]

        survivor = merged_pop[combined_indices]
        survivor_fitness = merged_obj[combined_indices]

        state = state.update(population=survivor, fitness=survivor_fitness, key=key)

        return state