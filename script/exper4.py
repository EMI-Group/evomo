from evox import metrics
from evox.workflows import StdWorkflow
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from algorithms import TensorNSGA3
from problems import MOKnapsack, MNKLandscape
from jax import random
import jax
import jax.numpy as jnp
import time
import json
from tqdm import tqdm
from evox.operators import non_dominated_sort
import os
from evox.operators import mutation, crossover

# only for discrete problems
def run(algorithm_name, problem, key, objects=3, num_iter=100, d=500, data_type="bool"):
    lb = jnp.zeros((d,), dtype=jnp.bool if data_type == "bool" else jnp.float32)
    ub = jnp.ones((d,), dtype=jnp.bool if data_type == "bool" else jnp.float32)
    mutation_op = mutation.Bitflip(prob=0.1)
    crossover_op = crossover.UniformRand()
    algorithm = {
        "TensorNSGA3": TensorNSGA3(
            lb=lb,
            ub=ub,
            n_objs=objects,
            pop_size=1000,
            mutation_op=mutation_op,
            crossover_op=crossover_op,
            data_type=data_type,
        ),
    }.get(algorithm_name)

    workflow = StdWorkflow(
        algorithm,
        problem,
    )
    # state = workflow.init(key)
    # step_func = jax.jit(workflow.step).lower(state).compile()
    # state = step_func(state)
    workflow = StdWorkflow(
        algorithm, problem, opt_direction="min", jit_step=True
    )
    state = workflow.init(key)
    step_func = workflow.step
    run_time = []
    obj = []
    start = time.perf_counter()
    for k in range(num_iter):
        state = step_func(state)
        jax.block_until_ready(state)
        now = time.perf_counter()
        duration = now - start
        run_time.append(duration)
        obj.append(state.get_child_state("algorithm").fitness)
    return jnp.array(obj), jnp.array(run_time)


def evaluate(f, pf, num_iter=100):
    igd = []
    ind1 = metrics.IGD(pf)

    history_data = []
    for i in range(num_iter):
        current_obj = f[i]
        current_obj = current_obj[~jnp.isnan(current_obj).any(axis=1)]
        rank = non_dominated_sort(current_obj)
        pf = rank == 0
        pf_fitness = current_obj[pf]
        igd.append(ind1(pf_fitness))

        if i == num_iter - 1:
            data = {
                "raw_obj": current_obj.tolist(),
                "pf_fitness": pf_fitness.tolist(),
            }
            history_data.append(data)

    return history_data, jnp.array(igd)


if __name__ == "__main__":
    jax.config.update("jax_default_prng_impl", "rbg")
    num_iter = 100

    algorithm_names = [
        "TensorNSGA3",
    ]

    obj = 3
    dim = 500
    key = random.PRNGKey(42)
    problem_key1, problem_key2 = random.split(key, 2)
    problem_list = [
        MNKLandscape(m=obj, d=dim, k=10, key=problem_key1),
        MOKnapsack(m=obj, d=dim, key=problem_key2),
    ]
    num_runs = 31
    num_pro = 9

    experiment_stats = []
   
    pro_keys = random.split(key, num_pro)

    directory = f"../data/effi_scal"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for algorithm_name in algorithm_names:
        for j, problem in enumerate(problem_list):
            
            print(
                f"Running {algorithm_name} on {problem.__class__.__name__} with dimension 500"
            )

            pro_key = pro_keys[j]
            run_keys = random.split(pro_key, num_runs)
            
            for exp_id in tqdm(
                range(num_runs),
                desc=f"{algorithm_name} - Problem {problem.__class__.__name__}",
            ):
                run_key = run_keys[exp_id]
                obj, t = run(algorithm_name, problem, run_key, num_iter=num_iter)

               
                data = {
                    "obj": obj.tolist(),
                    "time": t.tolist(),
                }
                with open(
                    f"{directory}/{algorithm_name}_{problem.__class__.__name__}_exp{exp_id}.json",
                    "w",
                ) as f:
                    json.dump(data, f)
