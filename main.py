import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from evox import workflows, problems
import evox
from evox.monitors import StdMOMonitor, PopMonitor
from evox.metrics import IGD as IGD1
from evox.metrics import HV as HV1
import jax
import jax.numpy as jnp
import numpy as np
import time
from algorithms import TensorMOEAD, MOEAD1, PMOEAD, MOEADOrigin, HypEOrigin, NSGA3
from evox.utils import cos_dist
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax.experimental.host_callback import id_print
from evox.operators import non_dominated_sort


def run_moea(algorithm, key):
    monitor = PopMonitor()
    # monitor = StdMOMonitor()

    problem = problems.numerical.DTLZ1(m=3)
    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
    )
    # workflow = workflows.NonJitWorkflow(
    #     algorithm=algorithm,
    #     problem=problem,
    #     monitor=monitor,
    # )

    state = workflow.init(key)

    true_pf = problem.pf()

    igd = IGD1(true_pf)
    ref = jnp.array([1., 1., 1.])
    ind = HV1(ref=ref)

    for i in range(100):
        print(i)
        key, subkey = jax.random.split(key)
        state = workflow.step(state)

        fit = state.get_child_state("algorithm").fitness
        pop = state.get_child_state("algorithm").population

        non_nan_rows = fit[~np.isnan(fit).any(axis=1)]
        non_nan_rows_pop = pop[~jnp.isnan(pop).any(axis=1)]
        print("igd", igd(non_nan_rows))

    # fig = monitor.plot()
    # fig.show()


if __name__ == '__main__':
    print("NSGA3")

    lb = jnp.full(shape=(12,), fill_value=0)
    ub = jnp.full(shape=(12,), fill_value=1)

    algorithm = NSGA3(
        lb=lb,
        ub=ub,
        n_objs=3,
        pop_size=5000,
    )
    # algorithm = evox.algorithms.HypE(
    #     lb=lb,
    #     ub=ub,
    #     n_objs=3,
    #     pop_size=10000,
    # )
    key = jax.random.PRNGKey(42)

    for i in range(1):
        start = time.time()
        run_moea(algorithm, key)
        end = time.time()
        print(end-start)
        key, subkey = jax.random.split(key)
