import numpy as np
import jax 
import jax.numpy as jnp

from logpsi import log_psi
from free_hamiltonian import H_free


def test_make_logpsi():
    Lsite = 6
    N = int(Lsite/2)
    t = 1.
    U = 4.
    g = 0.5

    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()
   
    make_psi_ratio, make_logpsi, grad_g_logpsi = log_psi(psi, Lsite, N, g)

    make_psi_ratio_vmapped = jax.vmap(make_psi_ratio, in_axes = (0, 0), out_axes = 0)
    make_logpsi_vmapped = jax.vmap(make_logpsi, in_axes = 0, out_axes = 0)

    state = jnp.array([ 0, 1, 2, 6, 7, 8 ])
    #print(make_logpsi(state))
    states = jnp.array([ [ 0, 1, 2, 6, 7, 8 ], \
                         [ 0, 1, 2, 6, 8, 9 ] ])
    #print(make_logpsi_vmapped(states))
    #print(make_psi_ratio_vmapped(states, states))


test_make_logpsi()


 
