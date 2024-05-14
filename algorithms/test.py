from algorithms.nsga2_origin import nsga2
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
    # 0.06821167
    print("start original nsga2")
    start = time.time()
    ori_ns2 = nsga2(lb=lb, ub=ub, n_objs=n_obj, pop_size=pop_size, key=key, problem=problem, loop_num=100)
    df = ori_ns2.fun()
    end = time.time()
    print(end-start)
    true_pf = problem.pf()
    pf, _ = problem.evaluate(State(), df)
    igd = IGD(true_pf)
    print(igd(pf))
    print(df.shape)
    print(pf)

def test_evox():
    # start evox nsga2 1000 loop
    # 3.1723215579986572
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


test_evox()
test_nsga2()

