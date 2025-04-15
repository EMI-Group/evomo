import jax
import jax.numpy as jnp
from functools import partial

from evox import jit_class, Algorithm, State
from evox.operators import selection, mutation, crossover, non_dominated_sort


@partial(jax.jit, static_argnums=[3])
def cal_hv(points, ref, k, n_sample, key):
    n, m = jnp.shape(points)
    # hit in alpha relevant partition
    alpha = jnp.cumprod(
        jnp.r_[1, (k - jnp.arange(1, n)) / (n - jnp.arange(1, n))]
    ) / jnp.arange(1, n + 1)
    alpha = jnp.nan_to_num(alpha)

    f_min = jnp.min(points, axis=0)

    samples = jax.random.uniform(key, shape=(n_sample, m), minval=f_min, maxval=ref)

    # update hypervolume estimates
    ds = jnp.zeros((n_sample,))
    pds = jax.vmap(
        lambda x: jnp.sum((jnp.tile(x, (n_sample, 1)) - samples) <= 0, axis=1) == m,
        in_axes=0,
        out_axes=0,
    )(points)
    ds = jnp.sum(jnp.where(pds, ds + 1, ds), axis=0)
    ds = jnp.where(ds == 0, ds, ds - 1)

    def cal_f(val):
        temp = jnp.where(val, ds, -1).astype(int)
        value = jnp.where(temp != -1, alpha[temp], 0)
        value = jnp.sum(value)
        return value

    f = jax.vmap(cal_f, in_axes=0, out_axes=0)(pds)
    f = f * jnp.prod(ref - f_min) / n_sample

    return f


@jit_class
class TensorHypE(Algorithm):
    """
    The tensorized version of HypE algorithm.

    :references:
        [1] J. Bader and E. Zitzler, "HypE: An algorithm for fast hypervolume-based many-objective optimization,"
            Evolutionary Computation, vol. 19, no. 1, pp. 45-76, 2011. Available:
            https://direct.mit.edu/evco/article-abstract/19/1/45/1363/HypE-An-Algorithm-for-Fast-Hypervolume-Based-Many

        [2] Z. Liang, H. Li, N. Yu, K. Sun, and R. Cheng, "Bridging Evolutionary Multiobjective Optimization and
            GPU Acceleration via Tensorization," IEEE Transactions on Evolutionary Computation, 2025. Available:
            https://ieeexplore.ieee.org/document/10944658
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        uniform_init=True,
        n_sample=10000,
        mutation_op=None,
        crossover_op=None,
    ):
        """
        Initializes the TensorHype algorithm.

        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param n_objs: The number of objectives in the optimization problem.
        :param pop_size: The size of the population.
        :param uniform_init: Whether to initialize the population uniformly. Defaults to True.
        :param n_sample: The number of samples to generate for the random search. Defaults to 10000.
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary` if not provided (optional).
        """
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.n_sample = n_sample
        self.uniform_init = uniform_init

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
        initializer = jax.nn.initializers.glorot_normal()

        if self.uniform_init:
            population = (
                jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
                * (self.ub - self.lb)
                + self.lb
            )
        else:
            population = initializer(subkey, shape=(self.pop_size, self.dim))

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
        hv = cal_hv(pop_obj, state.ref_point, self.pop_size, self.n_sample, subkey)

        selected, _ = self.selection(sel_key, population, -hv)
        crossovered = self.crossover(x_key, selected)
        next_generation = self.mutation(mut_key, crossovered)
        next_generation = jnp.clip(next_generation, self.lb, self.ub)

        return next_generation, state.update(next_generation=next_generation)

    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_obj = jnp.concatenate([state.fitness, fitness], axis=0)

        n = self.pop_size

        rank = non_dominated_sort(merged_obj)
        order = jnp.argsort(rank)
        worst_rank = rank[order[n - 1]]
        mask = rank <= worst_rank

        key, subkey = jax.random.split(state.key)
        hv = cal_hv(merged_obj, state.ref_point, jnp.sum(mask) - n, self.n_sample, subkey)

        dis = jnp.where(mask, hv, -jnp.inf)

        combined_indices = jnp.lexsort((-dis, rank))[: self.pop_size]

        survivor = merged_pop[combined_indices]
        survivor_fitness = merged_obj[combined_indices]

        state = state.update(population=survivor, fitness=survivor_fitness, key=key)

        return state
