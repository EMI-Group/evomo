from algorithms.nsga2_origin import NSGA2Origin
# from algorithms.old_nsga3_origin import NSGA3Origin2 as NSGA3Origin
from algorithms.nsga3_evox_ori import NSGA3Origin as NSGA3_EVOX_ORI
from algorithms.nsga3_evox import NSGA3 as NSGA3_EVOX
from algorithms.nsga3_origin import NSGA3Origin2 as NSGA3Origin_evox_version
import evox
import jax
import jax.numpy as jnp
from evox import algorithms, problems, workflows, monitors, State
from evox.metrics import IGD
import time
from jax import jit
from types import SimpleNamespace

n_obj = 3
pop_size = 100
dim = 5
# lb = jnp.full(shape=(dim,), fill_value=0)
# ub = jnp.full(shape=(dim,), fill_value=1)
problem = problems.numerical.DTLZ2
key = jax.random.PRNGKey(1)
loop_num = 10


def test_evox_algo(name, algo, config_dict):
    # evox nsga2 1000 loop
    # time: 3.1723 s
    # igd 0.06418476
    lb = jnp.full(shape=(config_dict.dim,), fill_value=0)
    ub = jnp.full(shape=(config_dict.dim,), fill_value=1)
    problem = config_dict.problem(d=config_dict.dim, m=config_dict.n_obj)
    print("start evox {}".format(name))
    start = time.time()
    algo_instance = algo(
        lb=lb,
        ub=ub,
        n_objs=config_dict.n_obj,
        pop_size=config_dict.pop_size,
    )
    # nsga3 = algorithms.NSGA3(lb=lb, ub=ub, n_objs=n_obj, pop_size=pop_size)
    workflow = workflows.NonJitWorkflow(algo_instance, problem)
    state = workflow.init(key)
    # run the workflow for 100 steps
    for i in range(config_dict.loop_num):
        state = workflow.step(state)
    end = time.time()
    print("time: {}".format(end-start))

    fit = state.get_child_state("algorithm").fitness
    # pop = state.get_child_state("algorithm").population
    non_nan_rows = fit[~jnp.isnan(fit).any(axis=1)]

    true_pf = problem.pf()
    igd = IGD(true_pf)

    print("igd: {}".format(igd(non_nan_rows)))


def test_ori_algo(name, algo, config_dict):
    print("---- start {}----".format(name))
    lb = jnp.full(shape=(config_dict.dim,), fill_value=0)
    ub = jnp.full(shape=(config_dict.dim,), fill_value=1)
    problem = config_dict.problem(d=config_dict.dim, m=config_dict.n_obj)
    start = time.time()
    algo_instance = algo(
        lb=lb,
        ub=ub,
        n_objs=n_obj,
        pop_size=pop_size,
        key=key,
        problem=problem,
        num_generations=loop_num,
    )
    df = algo_instance.run()
    end = time.time()
    print("time: {}".format(end - start))
    true_pf = problem.pf()
    pf, _ = problem.evaluate(State(), df)
    igd = IGD(true_pf)
    print("igd: {}".format(igd(pf)))
    print("df shape: {}".format(df.shape))


def tests():
    config_dict = {
        "n_obj": n_obj,
        "pop_size": pop_size,
        "dim": dim,
        "problem": problem,
        "key": key,
        "loop_num": loop_num
    }

    config = SimpleNamespace(**config_dict)
    test_evox_algo("nsga3_evox", NSGA3Origin_evox_version, config)
    # test_evox_algo("nsga3_evox_1", evox.algorithms.NSGA3, config)
    # config.pop_size = 1000
    # config.dim = 10
    # config.loop_num = 1000
    # test_evox_algo("nsga3_improve_2", NSGA3_EVOX, config)
    # test_evox_algo("nsga3_evox_2", evox.algorithms.NSGA3, config)
    

if __name__ == "__main__":
    tests()
    # from jax import config

    # # 禁用 JIT 编译
    # config.update("jax_disable_jit", True)

    # 现在所有 jit 装饰的函数都会以非 JIT 模式执行，便于调试

    # name = "nsga3"
    # algo = NSGA3_EVOX
    # test_evox_algo(name, algo)
