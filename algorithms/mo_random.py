import jax
import jax.numpy as jnp
from evox.operators import selection
from evox import Algorithm, jit_class, State


@jit_class
class MORandom(Algorithm):
    """Multiobjective random search algorithm

    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        uniform_init=False,
        selection_op=None,
    ):
        """
        Initializes the Multiobjective random search algorithm.

        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param n_objs: The number of objectives in the optimization problem.
        :param pop_size: The size of the population.
        :param uniform_init: Whether to initialize the population uniformly. Defaults to False.
        :param selection_op: The selection operation for evolutionary strategy (optional).
        """
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.uniform_init = uniform_init

        self.selection = selection_op
        if self.selection is None:
            self.selection = selection.UniformRand(0.5)

    def setup(self, key):
        key, subkey = jax.random.split(key)

        if self.uniform_init:
            population = (
                    jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
                    * (self.ub - self.lb)
                    + self.lb
            )
        else:
            population = (
                    jax.random.normal(subkey, shape=(self.pop_size, self.dim)) * 0.1
            )

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
        key, subkey = jax.random.split(state.key, 2)
        next_generation = jax.random.normal(subkey, shape=(self.pop_size, self.dim)) * 0.1
        next_generation = jnp.clip(next_generation, self.lb, self.ub)

        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        survivor, survivor_fitness = self.selection(state.key, merged_pop, merged_fitness)
        state = state.update(population=survivor, fitness=survivor_fitness)
        return state