import jax
import jax.numpy as jnp

a = jnp.arange(10)
b = a.at[jnp.nan].set(-1)
c = a.at[-1].set(-1) 
d = a.at[100].set(-1)


print