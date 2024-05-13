from nsga2 import nsga2
import evox
import jax
import jax.numpy as jnp

lb = jnp.full(shape=(2,), fill_value=-32)
ub = jnp.full(shape=(2,), fill_value=32)

key = jax.random.PRNGKey(0)
ori_ns2 = nsga2()

ori_ns2
