from nsga2_origin import NSGA2Origin
from nsga3_origin import NSGA3Origin 
from nsga3_evox_ori import NSGA3 as NSGA3_EVOX_ORI
import evox
import jax
import jax.numpy as jnp
from evox import algorithms, problems, workflows, monitors, State
from evox.metrics import IGD
import time
from jax import jit


n_obj = 3
pop_size = 100
dim = 500
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
    print(end - start)
    true_pf = problem.pf()
    pf, _ = problem.evaluate(State(), df)
    igd = IGD(true_pf)
    print(igd(pf))
    print(df.shape)
    print(pf)


def test_oris():
    names = ["nsga3", "nsga3_1", "nsga3_2", "nsga3_3", "nsga3_4"]
    algos = [NSGA3Origin, NSGA3Origin_1, NSGA3Origin_2, NSGA3Origin_3, NSGA3Origin_4]
    for i in range(5):
        test_ori_algo(names[i], algos[i])
    test_evox_algo("nsga3", algorithms.NSGA3)


if __name__ == "__main__":
    # test_evox_algo()
    # test_nsga3_evox_ori()
    test_oris()
    # test_evox_algo("nsga3", algorithms.NSGA3)
