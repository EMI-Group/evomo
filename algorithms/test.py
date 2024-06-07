import jax
import jax.numpy as jnp

a = jnp.arange(10)
c = a.at[-1].set(jnp.inf) 


print(c)