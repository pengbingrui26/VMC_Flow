import numpy as np
import jax
import jax.numpy as jnp

from ED_dimer_flow import dimer

model = dimer(1, 6)

"""
for g in jnp.arange(0.01, 1., 0.1):
    qgt = model.qgt(g)
    beta = 9.
    print('g, qgt, ln_qgt:', g, qgt, 1/(2*beta) * jnp.log(qgt)) 
"""

beta = 1.

F, Z = model.Free_energy(beta)
print(F)
