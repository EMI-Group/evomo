from evox import workflows, problems
from evox.metrics import IGD
from legacy.algorithms import TensorMOEAD
import jax
import jax.numpy as jnp
import time


def run_moea(algorithm, key):
    problem = problems.numerical.DTLZ2(m=3)
    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
    )

    state = workflow.init(key)

    true_pf = problem.pf()
    igd = IGD(true_pf)

    for i in range(100):
        print("Generation", i + 1)
        key, subkey = jax.random.split(key)
        state = workflow.step(state)
        fit = state.get_child_state("algorithm").fitness
        print("IGD:", igd(fit))


if __name__ == "__main__":
    print("TensorMOEAD")

    lb = jnp.full(shape=(12,), fill_value=0)
    ub = jnp.full(shape=(12,), fill_value=1)

    algorithm = TensorMOEAD(
        lb=lb,
        ub=ub,
        n_objs=3,
        pop_size=100,
    )

    key = jax.random.PRNGKey(42)

    for i in range(1):
        start = time.time()
        run_moea(algorithm, key)
        end = time.time()
        print("Time:", end - start)
        key, subkey = jax.random.split(key)
