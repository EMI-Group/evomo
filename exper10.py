# import os

# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
from evox import problems, metrics
from evox.workflows import StdWorkflow
from algorithms import MOEADOrigin, PMOEAD, HypEOrigin, HypE, NSGA3Origin, NSGA3, TensorMOEAD
from jax import random
import jax
import jax.numpy as jnp
import time
import json
from tqdm import tqdm
from evox.operators import non_dominated_sort
import os


def run(algorithm_name, problem, key, num_iter=100, d=5000):
#     try:
        algorithm = {
            "MOEADOrigin": MOEADOrigin(lb=jnp.zeros((d,)), ub=jnp.ones((d,)), n_objs=3, pop_size=10000, problem=problem, key=key, num_generations=100),
            "PMOEAD": PMOEAD(lb=jnp.zeros((d,)), ub=jnp.ones((d,)), n_objs=3, pop_size=10000),
            "HypEOrigin": HypEOrigin(lb=jnp.zeros((d,)), ub=jnp.ones((d,)), n_objs=3, pop_size=10000),
            "HypE": HypE(lb=jnp.zeros((d,)), ub=jnp.ones((d,)), n_objs=3, pop_size=10000),
            "NSGA3Origin": NSGA3Origin(lb=jnp.zeros((d,)), ub=jnp.ones((d,)), n_objs=3, pop_size=10000),
            "NSGA3": NSGA3(lb=jnp.zeros((d,)), ub=jnp.ones((d,)), n_objs=3, pop_size=10000),
            "TensorMOEAD": TensorMOEAD(lb=jnp.zeros((d,)), ub=jnp.ones((d,)), n_objs=3, pop_size=10000),
        }.get(algorithm_name)
        
        if algorithm_name == "MOEADOrigin":
            obj, run_time = algorithm.run()
            return jnp.array(obj), jnp.array(run_time)

        workflow = StdWorkflow(
            algorithm,
            problem,
        )
        state = workflow.init(key)
        step_func = jax.jit(workflow.step).lower(state).compile()
        state = step_func(state)
        run_time = []
        obj = []
#         pop = []
        start = time.time()
        for k in range(num_iter):
            state = step_func(state)
            jax.block_until_ready(state)
            now = time.time()
            duration = now - start
            run_time.append(duration)
            obj.append(state.get_child_state("algorithm").fitness)
#             pop.append(state.get_child_state("algorithm").population)
        return jnp.array(obj), jnp.array(run_time)

#     except Exception as e:
#         import traceback
#         traceback.print_stack()
#         print("Error occurred:", e)
#         return float("nan"), float("nan"), float("nan")


def evaluate(f, key, pf, num_iter=100):
    m = jnp.shape(pf)[1]
    ref = jnp.ones((m, ))
    igd = []

    ind1 = metrics.IGD(pf)

    history_data = []
    for i in range(num_iter):
        key, subkey = jax.random.split(key)
#         current_pop = x[i]
        current_obj = f[i]
        current_obj = current_obj[~jnp.isnan(current_obj).any(axis=1)]
#         current_pop = current_pop[~jnp.isnan(current_pop).any(axis=1)]
        rank = non_dominated_sort(current_obj)
        pf = rank == 0
        pf_fitness = current_obj[pf]
#         pf_solutions = current_pop[pf]
        fmax = jnp.max(pf, axis=0)

        igd.append(ind1(pf_fitness))

        if i == num_iter - 1:
            data = {"raw_obj":current_obj.tolist(), "pf_fitness":pf_fitness.tolist()}
            history_data.append(data)

    return history_data, jnp.array(igd)


if __name__ == "__main__":

    jax.config.update("jax_default_prng_impl", "rbg")
    num_iter = 100

    algorithm_names = ["MOEADOrigin", "TensorMOEAD", "HypEOrigin", "HypE", "NSGA3Origin", "NSGA3"]
#     algorithm_names = ["TensorMOEAD", "HypEOrigin", "HypE", "NSGA3Origin", "NSGA3"]
    algorithm_names = ["TensorMOEAD", "MOEADOrigin", "NSGA3Origin"]
#     algorithm_names = ["HypE", "MOEADOrigin"]
    algorithm_names = ["NSGA3Origin"]
    problem_list = [
            problems.numerical.LSMOP1(m=3),
            problems.numerical.LSMOP2(m=3),
            problems.numerical.LSMOP3(m=3),
            problems.numerical.LSMOP4(m=3),
            problems.numerical.LSMOP5(m=3),
            problems.numerical.LSMOP6(m=3),
            problems.numerical.LSMOP7(m=3),
            problems.numerical.LSMOP8(m=3),
            problems.numerical.LSMOP9(m=3),
    ]
#     alpha_list = [1.5, 1.5, 50, 1.5, 5, 5, 5, 5, 5]
    num_runs = 31
    num_pro = 9

    experiment_stats = []
    key = random.PRNGKey(42)
    pro_keys = random.split(key, num_pro)

    directory = f"data/effi_scal"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

#     skip_task = [0, 1, 2, 3, 4, 5, 6, 7]
    skip_task = [0]
    for algorithm_name in algorithm_names:
        for j, problem in enumerate(problem_list):
#             d = 7 if j == 0 else 12  # Dimension of the decision variables
            if j in skip_task:
                continue
            print(f"Running {algorithm_name} on LSMOP{j + 1} with dimension 500")

            pro_key = pro_keys[j]
            run_keys = random.split(pro_key, num_runs)
            pf = problem.pf()
            for exp_id in tqdm(range(num_runs), desc=f"{algorithm_name} - Problem {j + 1}"):
                run_key = run_keys[exp_id]
                obj, t = run(algorithm_name, problem, run_key, num_iter=num_iter)

                history_data, igd = evaluate(
                    obj, run_key, pf, num_iter=num_iter
                )

                data = {"history_data": history_data, "igd": igd.tolist(), "time": t.tolist()}
                with open(f"{directory}/{algorithm_name}_LSMOP{j + 1}_exp{exp_id}.json", "w") as f:
                    json.dump(data, f)