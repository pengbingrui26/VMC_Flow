import jax 
import jax.numpy as jnp
from functools import partial

from free_hamiltonian import H_free
#from logpsi import all_hop, count_double_occ, jump, random_init, log_psi
from logpsi import log_psi
#from metropolis import make_eloc, walk, make_Veff
from metropolis import random_init, make_E, make_QGT, make_QGT_ED, make_Veff, make_loss
from sampler import mixed_Gaussian_sampler, flow_fn


#def test_make_eloc():


def test_make_E():
    batch = 500
    t = 1.
    U = 6.
    Lsite = 2
    #g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    states_init = random_init(batch, Lsite)

    nthermal = 100
    beta = 2.5

    make_psi_ratio, make_logpsi, grad_g_logpsi = log_psi(psi, Lsite, N) 

    gs = jnp.array([ 0.28, 0.3, 0.32 ])
    W = gs.size

    make_E_vmapped = jax.vmap(make_E, in_axes = (None, None, None, None, None, 0, None), out_axes = 0)
 
    #gs = jnp.arange(0.37, 0.67, 0.1)
    E_mean, grad = make_E_vmapped(t, U, states_init, make_psi_ratio, make_logpsi, gs, nthermal)
    print(E_mean)
    print(grad)
    print(E_mean/Lsite)


def test_qgt():
    batch = 3000
    t = 1.
    U = 10.
    Lsite = 6
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    states_init = random_init(batch, Lsite)

    nthermal = 200
    beta = 2.5

    make_psi_ratio, make_logpsi, grad_g_logpsi = log_psi(psi, Lsite, N) 

    gs = jnp.array([0.01, 0.02, 0.03, 0.04])
    make_QGT_vmapped = jax.vmap(make_QGT, in_axes = (None, None, None, None, None, 0, None), out_axes = 0)
    qgts = make_QGT_vmapped(t, U, states_init, make_psi_ratio, grad_g_logpsi, gs, nthermal)
    print(qgts)


def test_make_QGT_ED():
    t = 1.
    U = 4.
    gs = np.array([ 0.4, 0.5 ])
    
    qgts = jax.vmap(make_QGT_ED, in_axes = (None, None, 0), out_axes = 0)(t, U, gs)
    #print(qgts)
    for qgt in qgts:
        print(qgt)
        print(type(qgt))


def test_make_Veff():
    batch = 10
    t = 1.
    U = 10.
    Lsite = 2
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    states_init = random_init(batch, Lsite)

    nthermal = 100
    beta = 2.5

    make_psi_ratio, make_logpsi, make_grad_logpsi = log_psi(psi, Lsite, N) 

    gs = jnp.array([ 0.1, 0.3, 0.5 ])
 
    Veff, E, double_occ = jax.vmap(make_Veff, in_axes = (None, None, None, None, None, None, 0, None), \
                     out_axes = (0, 0, 0))(beta, t, U, states_init, make_psi_ratio, make_grad_logpsi, gs, nthermal)

    print(Veff)
    print(E)
    print(double_occ)



def test_make_loss():
    batch = 10
    t = 1.
    U = 6.
    Lsite = 2
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    states_init = random_init(batch, Lsite)

    nthermal = 100
    beta = 2.5

    batch_g = 30
    Lsize = 20
    shape = (batch_g, )
    mu = 0.
    sigma = 1.

    key = jax.random.PRNGKey(42)

    params_init, loss_fn = make_loss(beta, psi, Lsite, N, t, U, nthermal, \
                                       log_psi, flow_fn, batch_g, mu, sigma, Lsize, key)
    print('params_init:')
    print(params_init)   

    #exit()
 
    #params = jnp.array([0.7, 0.1, 0.56])
    gs, grad, loss, E_mean, double_occ_mean = loss_fn(states_init, params_init)

    print('grad:')
    print(grad)
    print('loss:')
    print(loss)
    print('E_mean:')
    print(E_mean)
    print('double_occ_mean:') 
    print(double_occ_mean)


def test_final_Veff():
    batch = 10
    t = 1.
    U = 10.
    Lsite = 2
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    states_init = random_init(batch, Lsite)

    nthermal = 100
    beta = 2.5

    make_psi_ratio, make_logpsi, make_grad_logpsi = log_psi(psi, Lsite, N) 

    gs = jnp.array([ 0.1, 0.3, 0.5 ])
 
    Veff, E, double_occ = jax.vmap(make_Veff, in_axes = (None, None, None, None, None, None, 0, None), \
                     out_axes = (0, 0, 0))(beta, t, U, states_init, make_psi_ratio, make_grad_logpsi, gs, nthermal)

    print(Veff)
    print(E)
    print(double_occ)





# run ==========================================================

#test_make_E()
#test_qgt()
#test_make_QGT_ED()
#test_make_Veff()

test_make_loss()


