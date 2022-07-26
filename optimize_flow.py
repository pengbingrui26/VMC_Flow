import jax 
import jax.numpy as jnp
from functools import partial
import haiku as hk

from free_hamiltonian import H_free
from logpsi import log_psi
from metropolis import random_init, make_E, make_QGT, make_QGT_ED, make_loss
from sampler import flowNet, flow_fn


def optimize_flow():
    batch = 500
    t = 1.
    U = 6.
    Lsite = 2
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    states_init = random_init(batch, Lsite)

    nthermal = 100
    beta = 1.

    opt_nstep = 1000
    learning_rate = 1e-2

    batch_g = 500
    mu = 0.
    sigma = 1.
    Lsize = 50

    #
    def forward_fn_flow(x):
        module = flowNet(Lsize)
        return module(x)

    make_flow_forward = hk.transform(forward_fn_flow)

    key_dummy = jax.random.PRNGKey(21)
    x_dummy = jax.random.uniform(key_dummy, minval=0., maxval=1.)
    params_init = make_flow_forward.init(key_dummy, x_dummy)
    # 

    #exit()

    make_psi_ratio, make_logpsi, make_grad_logpsi = log_psi(psi, Lsite, N) 
 
    params = params_init
    #print('params:')
    #print(params)

    #exit()

    import optax
    optimizer = optax.adam(learning_rate = learning_rate)
    opt_state = optimizer.init(params)

    #exit()

    def step(params, opt_state, key):
        loss_fn = make_loss(beta, psi, Lsite, N, t, U, nthermal, \
                              log_psi, flow_fn, batch_g, mu, sigma, Lsize, key)
       
        gs, log_p, grad, loss, E_mean, double_occ_mean = loss_fn(states_init, params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, gs, log_p, grad, loss, E_mean, double_occ_mean

    key = jax.random.PRNGKey(42)

    grad_all = []
    loss_all = []
    Emean_all = []
    gs_all = []

    gs_final, logp_final = '', ''

    for istep in range(opt_nstep):
        key_old, key = jax.random.split(key, 2)
        params, opt_state, gs, log_p, grad, loss, E_mean, double_occ_mean = step(params, opt_state, key)
        print('istep:', istep)
        print('grad:')
        print(grad) 
        print('params:')
        print(params) 
        print('loss, E_mean, double_occ_mean:', loss, E_mean, double_occ_mean)
        #grad_all.append(grad)
        loss_all.append(loss)
        Emean_all.append(E_mean)
        gs_all.append(gs)
        if istep == (opt_nstep - 1):
            gs_final = gs
            logp_final = log_p
        print('\n') 

    #print('gs:')
    #print(gs)
    datas = { 'U': U, 'beta': beta, 'batch_x': batch, 'batch_g': batch_g, 'Lsize': Lsize, \
              'opt_nstep': opt_nstep, 'learning_rate': learning_rate, \
              'gs': gs_all, 'loss': loss_all, 'E_mean': Emean_all, 'gs_final': gs_final, 'logp_final': logp_final}

    import pickle as pk
    fd = open('./results_optimize_flow', 'wb')
    pk.dump(datas, fd)
    fd.close()


# run ================================================================

optimize_flow()

