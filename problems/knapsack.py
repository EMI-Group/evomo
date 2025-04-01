import jax
import jax.numpy as jnp
from evox import Problem, State, jit_class


@jit_class
class MOKnapsack(Problem):
    """E. Zitzler and L. Thiele, Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach, IEEE Transactions on Evolutionary Computation, 1999, 3(4): 257-271."""

    def __init__(self, m=None, d=None, key=None):
        """init

        Parameters
        ----------
        num_items : int
            the number of items
        num_knapsacks : int
            the number of knapsacks (objectives)
        """
        if m is None:
            self.num_knapsacks = 2  # Default number of knapsacks, 2, 3, 4
        else:
            self.num_knapsacks = m

        if d is None:
            self.num_items = 250  # Default number of items, 250, 500, 750
        else:
            self.num_items = d

        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        # Initialize random profits and weights
        self.profits = jax.random.randint(
            key1, (self.num_knapsacks, self.num_items), 10, 100
        )
        self.weights = jax.random.randint(
            key2, (self.num_knapsacks, self.num_items), 10, 100
        )  # m * d
        self.capacities = (jnp.sum(self.weights, axis=1) / 2).astype(
            jnp.int32
        )  # Set capacities to half of the total weight
        self.reference_point = jnp.sum(
            self.profits, axis=1
        )  # Set reference point to a large value

        # Sort items by their profit-to-weight ratio
        ratios = jnp.max(self.profits / self.weights, axis=0)
        self.sorted_indices = jnp.argsort(ratios, descending=False, stable=False)  # d

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, X):
        # Calculate total profit and weight for each knapsack
        total_profits = jnp.dot(X, self.profits.T)  # n * m
        total_weights = jnp.dot(X, self.weights.T)  # n * m

        # Iterate over each solution and repair it if it violates the constraints
        def repair_soulutions(sol, profit, weight):
            def cond(vals):
                sol, profit, weight = vals
                return jnp.any(weight > self.capacities)

            def loop_fn(vals):
                sol, profit, weight = vals
                sorted_sol = sol[self.sorted_indices]
                rm_index = jnp.nonzero(sorted_sol, size=1)
                index = self.sorted_indices[rm_index]
                sol = sol.at[index].set(False)
                weight = weight - self.weights[:, index].reshape(-1)
                profit = profit - self.profits[:, index].reshape(-1)
                return sol, profit, weight

            sol, profit, weight = jax.lax.while_loop(
                cond, loop_fn, (sol, profit, weight)
            )

            return profit

        repaired_profits = jax.vmap(repair_soulutions, in_axes=(0, 0, 0))(
            X, total_profits, total_weights
        )

        # Calculate objective values as total profit - actual profit
        objectives = jnp.sum(self.profits, axis=1, dtype=jnp.float32) - repaired_profits
        return objectives, state


if __name__ == "__main__":
    # Example instantiation and usage
    dim = 12
    pop = 5
    obj = 3
    knapsack_problem = MOKnapsack(d=dim, m=obj)
    state = knapsack_problem.setup(jax.random.PRNGKey(0))
    solutions = jnp.array(
        jax.random.randint(jax.random.PRNGKey(1), (pop, dim), 0, 2)
    ).astype(jnp.bool_)  # Random binary solutions
    objectives, _ = knapsack_problem.evaluate(state, solutions)
    print("Capacities:", knapsack_problem.capacities)
    print("Objective values:", objectives)
    print("sorted_indices:", knapsack_problem.sorted_indices)
