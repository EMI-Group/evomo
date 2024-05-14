from nsga2 import nsga2
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

print("start")
start = time.time()
problem = problems.numerical.DTLZ2(m=n_obj)
key = jax.random.PRNGKey(0)
ori_ns2 = nsga2(lb=lb, ub=ub, n_objs=n_obj, pop_size=pop_size, key=key, problem=problem, loop_num=100)
df = ori_ns2.fun()
end = time.time()
print(end-start)
true_pf = problem.pf()
pf, _ = problem.evaluate(State(), df)
igd = IGD(true_pf)

print(igd(pf))


