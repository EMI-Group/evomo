from algorithms.nsga2_origin import NSGA2Origin
from algorithms.nsga3_origin import NSGA3Origin2 as NSGA3Origin
from algorithms.nsga3_evox_ori import NSGA3Origin as NSGA3_EVOX_ORI
from algorithms.nsga3_evox import NSGA3 as NSGA3_EVOX
import evox
import jax
import jax.numpy as jnp
from evox import algorithms, problems, workflows, monitors, State
from evox.metrics import IGD
import time
from jax import jit


n_obj = 3
pop_size = 10
dim = 5
lb = jnp.full(shape=(dim,), fill_value=0)
ub = jnp.full(shape=(dim,), fill_value=1)
problem = problems.numerical.DTLZ2(d=dim, m=n_obj)
key = jax.random.PRNGKey(1)
loop_num = 100

def test_evox_algo(name, algo):
    # evox nsga2 1000 loop
    # time: 3.1723 s
    # igd 0.06418476
    print("start evox {}".format(name))
    start = time.time()
    algo_instance = algo(lb=lb, ub=ub, n_objs=n_obj, pop_size=pop_size)
    # nsga3 = algorithms.NSGA3(lb=lb, ub=ub, n_objs=n_obj, pop_size=pop_size)
    workflow = workflows.StdWorkflow(algo_instance, problem)
    state = workflow.init(key)
    # run the workflow for 100 steps
    for i in range(loop_num):
        state = workflow.step(state)
    end = time.time()
    print(end-start)

    fit = state.get_child_state("algorithm").fitness
    # pop = state.get_child_state("algorithm").population
    non_nan_rows = fit[~jnp.isnan(fit).any(axis=1)]

    true_pf = problem.pf()
    igd = IGD(true_pf)

    print("igd", igd(non_nan_rows))


def test_ori_algo(name, algo):
    print("---- start {}----".format(name))
    start = time.time()
    algo_instance = algo(
        lb=lb,
        ub=ub,
        n_objs=n_obj,
        pop_size=pop_size,
        key=key,
        problem=problem,
        num_generation=loop_num,
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
    test_ori_algo("nsga3", NSGA3Origin)
    test_evox_algo("nsga3", algorithms.NSGA3)
    # names = ["nsga3", "nsga3_1", "nsga3_2", "nsga3_3", "nsga3_4"]
    # algos = [NSGA3Origin, NSGA3Origin_1, NSGA3Origin_2, NSGA3Origin_3, NSGA3Origin_4]
    # for i in range(5):
    #     test_ori_algo(names[i], algos[i])


if __name__ == "__main__":
    from jax import config

    # 禁用 JIT 编译
    config.update("jax_disable_jit", True)

    # 现在所有 jit 装饰的函数都会以非 JIT 模式执行，便于调试

    name = "nsga3"
    algo = NSGA3_EVOX
    test_evox_algo(name, algo)
