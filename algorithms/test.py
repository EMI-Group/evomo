from nsga2_origin import NSGA2Origin
from nsga3_origin import NSGA3Origin
import evox
import jax
import jax.numpy as jnp
from evox import algorithms, problems, workflows, monitors, State
from evox.metrics import IGD
import time
from jax import jit

lb = jnp.full(shape=(3,), fill_value=0)
ub = jnp.full(shape=(3,), fill_value=1)
n_obj = 3
pop_size = 100
problem = problems.numerical.DTLZ2(m=n_obj)
key = jax.random.PRNGKey(1)

def test_nsga2():
    # origin nsga2 1000 loop
    # time: 4324.2117 s
    # igd: 0.06821167
    print("start original nsga2")
    start = time.time()
    ori_ns2 = NSGA2Origin(lb=lb, ub=ub, n_objs=n_obj, pop_size=pop_size, key=key, problem=problem, num_generation=100)
    df = ori_ns2.run()
    end = time.time()
    print(end-start)
    true_pf = problem.pf()
    pf, _ = problem.evaluate(State(), df)
    igd = IGD(true_pf)
    print(igd(pf))
    print(df.shape)
    print(pf)

def test_nage2_evox():
    # evox nsga2 1000 loop
    # time: 3.1723 s
    # igd 0.06418476
    print("start evox nsga2")
    start = time.time()
    nsga2 = algorithms.NSGA2(lb=lb, ub=ub, n_objs=n_obj, pop_size=pop_size)
    workflow = workflows.StdWorkflow(nsga2, problem)
    state = workflow.init(key)
    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)
    end = time.time()
    print(end-start)

    fit = state.get_child_state("algorithm").fitness
    # pop = state.get_child_state("algorithm").population
    non_nan_rows = fit[~jnp.isnan(fit).any(axis=1)]

    true_pf = problem.pf()
    igd = IGD(true_pf)

    print("igd", igd(non_nan_rows))


def test_nsga3():
    # origin nsga2 100 loop
    # time: 1055.1837 s
    # igd: 0.05381546
    print("start original nsga3")
    start = time.time()
    ori_ns3 = NSGA3Origin(
        lb=lb,
        ub=ub,
        n_objs=n_obj,
        pop_size=pop_size,
        key=key,
        problem=problem,
        num_generation=100,
    )
    df = ori_ns3.run()
    end = time.time()
    print(end - start)
    true_pf = problem.pf()
    pf, _ = problem.evaluate(State(), df)
    igd = IGD(true_pf)
    print(igd(pf))
    print(df.shape)
    print(pf)


def test_nage3_evox():
    # evox nsga2 100 loop
    # time: 4.2556774616 s
    # igd 0.053480785
    print("start evox nsga3")
    start = time.time()
    nsga3 = algorithms.NSGA3(lb=lb, ub=ub, n_objs=n_obj, pop_size=pop_size)
    workflow = workflows.StdWorkflow(nsga3, problem)
    state = workflow.init(key)
    # run the workflow for 100 steps
    for i in range(100):
        state = workflow.step(state)
    end = time.time()
    print(end - start)

    fit = state.get_child_state("algorithm").fitness
    # pop = state.get_child_state("algorithm").population
    non_nan_rows = fit[~jnp.isnan(fit).any(axis=1)]

    true_pf = problem.pf()
    igd = IGD(true_pf)

    print("igd", igd(non_nan_rows))


if __name__ == "__main__":
    test_nsga3()
    test_nage3_evox()
