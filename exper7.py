from algorithms import PMOEAD, HypE, NSGA3, MORandom
from evox.workflows import StdWorkflow
from problems import MoBrax, Obs_Normalizer
from jax import random
import jax
import jax.numpy as jnp
import time
from flax import linen as nn
import json
from tqdm import tqdm
from evox.utils import TreeAndVector
from evox.operators import non_dominated_sort
from tensorboardX import SummaryWriter
from evox.metrics import HV



def get_algorithm(algorithm_name, center, n_objs, pop_size):
    bounds = jnp.full_like(center, -5), jnp.full_like(center, 5)
    return {
        "Random": MORandom(*bounds, n_objs=n_objs, pop_size=pop_size),
        "NSGAIII": NSGA3(*bounds, n_objs=n_objs, pop_size=pop_size, uniform_init=False,),
        "HYPE": HypE(*bounds, n_objs=n_objs, pop_size=pop_size, uniform_init=False,),
        "PMOEAD": PMOEAD(*bounds, n_objs=n_objs, pop_size=pop_size, uniform_init=False,),
    }.get(algorithm_name, None)


def run_workflow(
    algorithm, problem, key, adapter, num_iter, ref_point, Al_env_exp_struct, log_dir
):
    workflow = StdWorkflow(
        algorithm,
        problem,
        pop_transform=adapter.batched_to_tree,
        opt_direction="max",
    )
    state = workflow.init(key)
    step_func = jax.jit(workflow.step).lower(state).compile()
    times = []
    pop = []
    obj = []
    start = time.perf_counter()
    writer = SummaryWriter(log_dir=f"logs/{log_dir}/{Al_env_exp_struct}_log1")
    metric_key = random.PRNGKey(42)
    writer.add_scalar("HV", 0, 0)
    for iter in range(num_iter):
        state = step_func(state)
        times.append(time.perf_counter() - start)
        temp = -state.get_child_state("algorithm").fitness
        temp1 = state.get_child_state("algorithm").population
        current_obj = -temp[~jnp.isnan(temp).any(axis=1)]
        current_pop = temp1[~jnp.isnan(temp1).any(axis=1)]

        if current_obj.ndim < 2:
            print(
                f"Shape issue at iteration {num_iter}: current_obj shape {current_obj.shape}"
            )
        rank = non_dominated_sort(current_obj)
        pf_indices = rank == 0
        current_pf = -current_obj[pf_indices]
        if iter == num_iter - 1:
            pop.append(current_pop.tolist())
        obj.append({"pf": current_pf.tolist()})

        current_pf = current_pf[jnp.all(current_pf >= ref_point, axis=1)]
        if current_pf.shape[0] == 0:
            writer.add_scalar("HV", 0, iter+1)
            continue
        hv_metric = HV(ref=(-1 * ref_point))
        current_hv = hv_metric(random.split(metric_key)[1], -current_pf)
        writer.add_scalar("HV", current_hv, iter+1)

    # return jnp.array(pop), jnp.array(obj), jnp.array(times)
    writer.close()
    return pop, obj, jnp.array(times)


def main():
    jax.config.update("jax_default_prng_impl", "rbg")
    num_iter, num_runs = 100, 10
    algorithm_list = ["Random", "NSGAIII", "PMOEAD", "HYPE"]
    algorithm_list = ["PMOEAD"]
    envs = [
#         {
#             "name": "mo_halfcheetah",
#             "observation_shape": 17,
#             "action_shape": 6,
#             "num_obj": 2,
#             "type": "continuous",
#             "ref": jnp.array([0, -291]),
#             "scale": 1,
#             "layer": 2,
#             "node": 16,
#         },
#         {
#             "name": "mo_hopper_m3",
#             "observation_shape": 11,
#             "action_shape": 3,
#             "num_obj": 3,
#             "type": "continuous",
#             "ref": jnp.array([0, 0, -881]),
#             "scale": 1,
#             "layer": 2,
#             "node": 16,
#         },
#         {
#             "name": "mo_swimmer",
#             "observation_shape": 8,
#             "action_shape": 2,
#             "num_obj": 2,
#             "type": "continuous",
#             "ref": jnp.array([0, -0.1]),
#             "scale": 1,
#             "layer": 2,
#             "node": 16,
#         },
#         {
#             "name": "mo_ant",
#             "observation_shape": 27,
#             "action_shape": 8,
#             "num_obj": 2,
#             "type": "continuous",
#             "ref": jnp.array([0, -315]),
#             "scale": 1,
#             "layer": 2,
#             "node": 16,
#         },
        {
            "name": "mo_humanoid",
            "observation_shape": 244,
            "action_shape": 17,
            "num_obj": 2,
            "type": "continuous",
            "ref": jnp.array([0, 0]),
            "scale": 0.4,
            "layer": 2,
            "node": 16,
        },
        {
            "name": "mo_humanoidstandup",
            "observation_shape": 244,
            "action_shape": 17,
            "num_obj": 2,
            "type": "continuous",
            "ref": jnp.array([0, -200]),
            "scale": 0.4,
            "layer": 2,
            "node": 16,
        },
        {
            "name": "mo_inverted_double_pendulum",
            "observation_shape": 8,
            "action_shape": 1,
            "num_obj": 2,
            "type": "continuous",
            "ref": jnp.array([0, 0]),
            "scale": 1,
            "layer": 2,
            "node": 16,
        },
        {
            "name": "mo_walker2d",
            "observation_shape": 17,
            "action_shape": 6,
            "num_obj": 2,
            "type": "continuous",
            "ref": jnp.array([0, 0]),
            "scale": 1,
            "layer": 2,
            "node": 16,
        },
        {
            "name": "mo_pusher",
            "observation_shape": 23,
            "action_shape": 7,
            "num_obj": 3,
            "type": "continuous",
            "ref": jnp.array([0, 0, 0]),
            "scale": 1,
            "layer": 2,
            "node": 16,
        },
        {
            "name": "mo_reacher",
            "observation_shape": 11,
            "action_shape": 2,
            "num_obj": 2,
            "type": "continuous",
            "ref": jnp.array([0, 0]),
            "scale": 1,
            "layer": 2,
            "node": 16,
        },
    ]
    """
    humanoid  ac_shape:17  ob_shape:244
    humanoidStandup as below
    InvertedDoublePendulum  ac_shape:1  ob_shape:11
    walker2D  ac_shape:6  ob_shape:17
    ant  ac_shape:8  ob_shape:27
    hopper  ac_shape:3  ob_shape:11
    swimmer  ac_shape:2  ob_shape:8
    halfcheetah  ac_shape:6  ob_shape:17
    """
    alg_key = random.PRNGKey(43)

    for algorithm_name in tqdm(algorithm_list, desc="Algorithms"):
#         algorithm_name = "TensorRVEA"
        for env in tqdm(envs, desc="Environments"):
            run_keys = random.split(alg_key, num_runs)
            for exp_id in range(num_runs):
                run = "run"
                name = f"{algorithm_name}_{env['name']}_exp{exp_id}_{run}"
                log_dir = f"{algorithm_name}_{env['name']}_{run}"
                model_key, workflow_key = jax.random.split(run_keys[exp_id])

                """
                1*16, 32, 64
                2*16, 32, 64
                1,2 -- the num of the hidden layer
                16,32,64 -- the num of the agent in each hidden layer
                """

#                     if env["layer"] == 1:
#                         class PolicyModel(nn.Module):
#                             @nn.compact
#                             def __call__(self, x):
#                                 x = nn.Dense(env["node"])(x)
#                                 x = nn.tanh(x)
#                                 x = nn.Dense(env["action_shape"])(x)
#                                 x = nn.tanh(x) * env["scale"]
#                                 return x
#                     else:
#                         class PolicyModel(nn.Module):
#                             @nn.compact
#                             def __call__(self, x):
#                                 x = nn.Dense(env["node"])(x)
#                                 x = nn.tanh(x)
#                                 x = nn.Dense(env["node"])(x)
#                                 x = nn.tanh(x)
#                                 x = nn.Dense(env["action_shape"])(x)
#                                 x = nn.tanh(x) * env["scale"]
#                                 return x

                class PolicyModel(nn.Module):
                    @nn.compact
                    def __call__(self, x):
                        x = nn.Dense(env["node"])(x)
                        x = nn.tanh(x)
                        x = nn.Dense(env["action_shape"])(x)
                        x = nn.tanh(x) * env["scale"]
                        return x

                model = PolicyModel()
                params = model.init(model_key, jnp.zeros((env["observation_shape"],)))
                adapter = TreeAndVector(params)
                center = adapter.to_vector(params)
                obs_norm = Obs_Normalizer(observation_shape=env["observation_shape"], useless=False)
                problem = MoBrax(
                    policy=jax.jit(model.apply),
                    env_name=env["name"],
                    cap_episode=1000,  # 500
                    obs_norm=obs_norm,
                    num_obj=env["num_obj"],
                )

                algorithm = get_algorithm(algorithm_name, center, env["num_obj"], 10000)
                if algorithm is None:
                    raise ValueError(f"Algorithm {algorithm_name} not recognized")

                pop, objs, times = run_workflow(
                    algorithm,
                    problem,
                    workflow_key,
                    adapter,
                    num_iter,
                    env["ref"],
                    name,
                    log_dir,
                )

                raw_data = {"pop": pop, "objs": objs, "time": times.tolist()}

                with open(f"data/mul_neuro/{name}.json", "w") as f:
                    json.dump(raw_data, f)



if __name__ == "__main__":
    main()
