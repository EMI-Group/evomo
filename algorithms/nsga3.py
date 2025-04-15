import jax
import jax.numpy as jnp
from evox.operators import (
    non_dominated_sort,
    selection,
    mutation,
    crossover,
    sampling,
)
from evox import Algorithm, State
from evox.utils import cos_dist

class NSGA3(Algorithm):
    """
    An implementation of the tensorized NSGA-III for many-objective optimization problems.

    :references:
        [1] K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
            Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints," IEEE Transactions on Evolutionary
            Computation, vol. 18, no. 4, pp. 577-601, 2014. Available: https://ieeexplore.ieee.org/document/6600851

        [2] H. Li, Z. Liang, and R. Cheng, "GPU-accelerated Evolutionary Many-objective Optimization Using Tensorized
            NSGA-III," IEEE Congress on Evolutionary Computation, 2025.
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
        data_type=None,
    ):
        """
        Initializes the NSGA-III algorithm.

        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param n_objs: The number of objective functions in the optimization problem.
        :param pop_size: The size of the population.
        :param uniform_init: Whether to use uniform initialization (default: True).
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary` if not provided (optional).
        :param data_type: The data type for the decision variables, either 'float' or 'bool' (optional).
        """
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.uniform_init = uniform_init
        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op
        self.dype = data_type
        self.sampling = sampling.UniformSampling(self.pop_size, self.n_objs)

        if self.selection is None:
            self.selection = selection.UniformRand(1)
        if self.mutation is None:
            self.mutation = mutation.Polynomial((self.lb, self.ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()
        if self.dype is None:
            self.dype = "float"

        self.limit_num = jnp.maximum(50, self.pop_size//100)

    def setup(self, key):
        """
        Initialize the state of the algorithm.

        :param key: Random key for reproducibility.
        """
        key, subkey = jax.random.split(key)
        initializer = jax.nn.initializers.glorot_normal()
        if self.uniform_init:
            if self.dype == "float":
                population = (
                    jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
                    * (self.ub - self.lb)
                    + self.lb
                )
            else:
                population = jax.random.uniform(subkey, shape=(self.pop_size, self.dim))
                population = population > 0.5
        else:
            population = initializer(subkey, shape=(self.pop_size, self.dim))
        ref = self.sampling(subkey)[0]
        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            key=key,
            ref=ref,
        )

    def init_ask(self, state):
        """
        Return the initial population and state.

        :param state: Current state of the algorithm.
        """
        return state.population, state

    def init_tell(self, state, fitness):
        """
        Update the state with the fitness values.

        :param state: Current state of the algorithm.
        :param fitness: Fitness values for the initial population.
        """
        state = state.update(fitness=fitness)
        return state

    def ask(self, state):
        """
        Generate the next population using crossover and mutation.

        :param state: Current state of the algorithm.
        """
        key, mut_key, x_key = jax.random.split(state.key, 3)
        crossovered = self.crossover(x_key, state.population)
        next_generation = self.mutation(mut_key, crossovered)
        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        key, shuffled_key, shuffle_key1 = jax.random.split(state.key, 3)
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)
        shuffled_idx = jax.random.permutation(shuffled_key, merged_pop.shape[0])
        merged_pop = merged_pop[shuffled_idx]
        merged_fitness = merged_fitness[shuffled_idx]
        rank = non_dominated_sort(merged_fitness)
        order = jnp.argsort(rank)
        last_rank = rank[order[self.pop_size]]
        the_selected_one_idx = order[0]
        the_drop_one_idx = order[-1]
        ranked_fitness = jnp.where(
            (rank <= last_rank)[:, None],
            merged_fitness,
            jnp.nan,
        )

        # Normalize
        ideal_points = jnp.min(merged_fitness, axis=0)
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
        shuffled_idx = jax.random.permutation(shuffle_key1, state.ref.shape[0])
        ref = state.ref[shuffled_idx]
        cos_distance = cos_dist(normalized_fitness, ref)
        # dist is matrix with shape is (merged_pop_size, ref_num)
        dist = jnp.linalg.norm(normalized_fitness, axis=-1, keepdims=True) * jnp.sqrt(
            1 - cos_distance**2
        )
        # Associate each solution with its nearest reference point
        group_id = jnp.nanargmin(dist, axis=1)
        group_dist = jnp.nanmin(dist, axis=1)
        group_id = jnp.where(group_id == -1, len(state.ref), group_id)
        # count the number of individuals for each group id, the id = len(state.ref) will be dropped due to the set "length=len(state.ref)"
        rho = jnp.bincount(
            jnp.where(rank < last_rank, group_id, len(state.ref)), length=len(state.ref)
        )
        rho_last = jnp.bincount(
            jnp.where(rank == last_rank, group_id, len(state.ref)),
            length=len(state.ref),
        )
        selected_number = jnp.sum(rho)
        upper_bound = jnp.iinfo(jnp.int32).max
        rho = jnp.where(rho_last == 0, upper_bound, rho)
        group_id = jnp.where(rank == last_rank, group_id, upper_bound)

        # Handle assignment of solutions to reference points
        rho_level = 0
        _selected_rho = rho == rho_level
        selected_rho = jnp.where(_selected_rho, jnp.arange(rho.size), upper_bound)

        def select_from_index_by_min(idx):
            return  jnp.argmin(jnp.where(group_id == idx, group_dist, jnp.inf))
        # use vmap to vectorize the selection
        candi_idx = jax.vmap(select_from_index_by_min)(selected_rho)
        temp_selected_idx = jnp.nonzero(_selected_rho, fill_value=upper_bound, size=1)[0]
        the_selected_one_idx = jnp.where(temp_selected_idx==upper_bound, the_selected_one_idx, temp_selected_idx)
        candi_idx = jnp.where(_selected_rho, candi_idx, the_selected_one_idx)
        rank = rank.at[candi_idx].set(last_rank - 1)
        group_id = group_id.at[candi_idx].set(upper_bound)
        
        bool_ref_candidates = jnp.arange(len(state.ref))[:, None] == group_id[None, :]
        ref_candidates = jax.vmap(lambda x: jnp.nonzero(x, fill_value=upper_bound, size=self.pop_size)[0])(bool_ref_candidates)
        ref_cand_idx = jnp.zeros_like(rho)

        rho_last = jnp.where(_selected_rho, rho_last - 1, rho_last)
        rho = jnp.where(_selected_rho, rho_level + 1, rho)
        rho = jnp.where(rho_last == 0, upper_bound, rho)
        selected_number += jnp.sum(_selected_rho)

        # for rho > 0
        def select_loop(vals):
            num, rho, rho_last, rank, candi_idx, _selected_rho, ref_cand_idx = vals
            rho_level = jnp.min(rho)
            _selected_rho = rho == rho_level
 
            candi_idx = ref_candidates[jnp.arange(len(state.ref)), ref_cand_idx]
            candi_idx = jnp.where(_selected_rho, candi_idx, the_selected_one_idx)
            
            ref_cand_idx = jnp.where(_selected_rho, ref_cand_idx+1, ref_cand_idx)
            rho_last = jnp.where(_selected_rho, rho_last - 1, rho_last)
            rho = jnp.where(_selected_rho, rho_level + 1, rho)
            rho = jnp.where(rho_last == 0, upper_bound, rho)
            rank = rank.at[candi_idx].set(last_rank - 1)
            num += jnp.sum(_selected_rho)
           
            return num, rho, rho_last, rank,  candi_idx, _selected_rho, ref_cand_idx

        selected_number, rho, rho_last, rank, candi_idx, _selected_rho, ref_cand_idx = (
            jax.lax.while_loop(
                lambda val: (val[0] < self.pop_size),
                select_loop,
                (selected_number, rho, rho_last, rank, candi_idx, _selected_rho, ref_cand_idx),
            )
        )

        dif = selected_number - self.pop_size
        candi_idx = jnp.where(_selected_rho, candi_idx, upper_bound)
        sorted_index = jnp.sort(candi_idx, stable=False)
        index = jnp.where(
            jnp.arange(sorted_index.size) < dif, sorted_index, the_drop_one_idx
        )
        rank = rank.at[sorted_index[index]].set(last_rank)

        candi_idx = jnp.nonzero(rank < last_rank, size=self.pop_size, fill_value=0)[0]
        state = state.update(
            population=merged_pop[candi_idx],
            fitness=merged_fitness[candi_idx],
            key=key,
        )
        return state
