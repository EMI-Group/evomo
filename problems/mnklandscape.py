import jax
import jax.numpy as jnp
import jax.random as random
from evox import Problem, State, jit_class

"""
This is minmazing problem which translated from maximazing problem：

Insights on properties of multiobjective MNKlandscapes
"""


@jit_class
class MNKLandscape(Problem):
    def __init__(self, m, d, k=None, key=None):
        if m is None:  # Number of objectives
            self.m = 3
        else:
            self.m = m

        if d is None:  # Length of bit string
            self.d = 8
        else:
            self.d = d

        if k is None:  # Number of epistatic interactions
            self.k = self.d // 2
        else:
            self.k = k

        if key is None:
            key = jax.random.PRNGKey(0)

        self.key = key

        self.IM, self.key = self.create_influence_matrix(self.key)
        self.FC, self.key = self.create_fitness_config(self.key)
        self.reference_point = jnp.zeros(self.m)

    def create_fitness_config(self, key):
        key, key2 = jax.random.split(key)
        return jax.random.uniform(key, (self.m, 2 ** (self.k + 1))), key2

    def create_influence_matrix(self, key):
        key1, key2 = jax.random.split(key)

        def generate_IM_row(key, index):
            prob = jnp.ones(self.d) / (self.d - 1)
            prob = prob.at[index].set(0)
            k_index = jax.random.choice(
                key, self.d, shape=(self.k,), p=prob, replace=False
            )
            IM_row = jnp.zeros(self.d)
            IM_row = IM_row.at[k_index].set(1)
            IM_row = IM_row.at[index].set(1)
            return IM_row

        def get_IM_matrix(key):
            return jax.vmap(generate_IM_row, in_axes=(0, 0))(
                jax.random.split(key, self.d), jnp.arange(self.d)
            )

        IM = jax.vmap(get_IM_matrix, in_axes=(0,))(jax.random.split(key1, self.m))
        return IM.astype(jnp.bool_), key2

    def setup(self, key):
        return State(key=key)

    def translate_to_index(self, sub_string):
        return jnp.sum(sub_string * (2 ** jnp.arange(self.k + 1)))

    """
    X: Bool solutions
    """

    def evaluate(self, state, X):
        def query_fitness(sol_row):
            def get_objectives(obj_index):
                IM = self.IM[obj_index]
                FC_row = self.FC[obj_index]

                def get_sub_fitness(IM_row):
                    final_im = sol_row & IM_row
                    index = jnp.argwhere(IM_row, size=self.k + 1, fill_value=0)
                    sub_string = final_im[index].reshape(-1)
                    index = self.translate_to_index(sub_string)
                    return FC_row[index]

                sub_fitness = jax.vmap(get_sub_fitness, in_axes=(0,))(IM)
                return jnp.mean(sub_fitness)

            fitness = jax.vmap(get_objectives, in_axes=(0,))(jnp.arange(self.m))
            return fitness

        objectives = jax.vmap(query_fitness, in_axes=(0,))(X)

        return -objectives, state


if __name__ == "__main__":
    N = 4  # Length of bit string
    K = 2  # Number of epistatic interactions
    M = 2  # Number of objectives

    key = jax.random.PRNGKey(0)
    mnk_problem = MNKLandscape(M, N, K, key)

    state = mnk_problem.setup(key)

    solutions = jax.random.randint(
        jax.random.PRNGKey(1), (100, N), 0, 2
    )  # Random binary solutions
    objectives, _ = mnk_problem.evaluate(state, solutions)

    print("Objectives values:", objectives)
