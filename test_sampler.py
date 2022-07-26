import numpy as np
import jax 
import jax.numpy as jnp
import haiku as hk
from functools import partial

from sampler import flowNet, flow_fn


def test_flowNet():
    L = 10
    
    def forward_fn_flow(x):
        module = flowNet(L)
        return module(x)

    forward_flow = hk.transform(forward_fn_flow)

    batch = 12
    key_dummy = jax.random.PRNGKey(21)
    x_dummy = jax.random.uniform(key_dummy, minval=0., maxval=1.)
    print('x_dummy:', x_dummy)

    params = forward_flow.init(key_dummy, x_dummy)
    print('params:')
    print(params)

    out_fn = forward_flow.apply

    key_z = jax.random.PRNGKey(51)

    z = jax.random.uniform(key_z, minval=0., maxval=1.)
    mu = 0.3
    z += mu
    #print('z:', z)
    g = out_fn(params, None, z)
    #print('g:', g)

    #exit()

    def grad(x):
        return jax.grad(out_fn, argnums = -1)(params, None, x)

    def grad_params(params, x):
        return jax.jacrev(out_fn, argnums = 0)(params, None, x)

    #print('grad(z):', grad(z))
    #print('grad_params:', grad_params(params, z))

    out_fn_vmapped = jax.vmap(out_fn, in_axes = (None, None, 0), out_axes = 0)

    batch = 10
    #zs = jax.random.uniform(key_z, shape = (batch, ), minval = -1., maxval = 1.)
    #zs = jnp.array([0.2]*batch)
    zs = jax.random.normal(key = jax.random.PRNGKey(11), shape = (batch, ))
    mu = 0.3
    zs += mu
    print('zs:', zs)
    gs = out_fn_vmapped(params, None, zs)
    print('gs:', gs)
    
    #exit()

    grad_vmapped = jax.vmap(grad, in_axes = 0, out_axes = 0)

    grad_params_vmapped = jax.vmap(grad_params, in_axes = (None, 0), out_axes = 0)

    print(grad_vmapped(zs))
 
    #def grads(x):
    #    return jax.jacrev(out_fn, argnums = -1)(params, None, x)
    #print(grads(zs))


def test_flow_fn():
    batch_g = 5 
    mu = 0.3
    sigma = 1.
    Lsize = 1
    key = jax.random.PRNGKey(42)
    zs, params_init, make_g, log_prob, grad_logprob = flow_fn(batch_g, mu, sigma, Lsize, key)

    print('params_init:')
    print(params_init)

    params = params_init['flow_net']
    w1 = params['w1']
    w2 = params['w2']
    b1 = params['b1']

    gs = make_g(params_init)
    print('gs:', gs) 
    ln_p = log_prob(params_init)
    print('ln_p:', ln_p)

    #jac_hand = w2 * (1 - jnp.power(jnp.tanh(w1 * zs + b1), 2)) * w1
    #print('jac_hand:')
    #print(jac_hand)

    #print('grad_w2:')
    #print(jnp.tanh(w1 * zs + b1))

    
    grad_logp = grad_logprob(params_init)
    print('grad_logp:')
    print(grad_logp)

    #gs_min = min(gs)
    #gs_max = max(gs)
    #dg = max(gs) - min(gs)
    #dg = dg / 25
    
    #nums = []
    #for gg in jnp.arange(gs_min, gs_max, dg):
    #    num = [ g for g in gs if (g > gg and g < (gg+dg)) ]
    #    nums.append(len(num))
    #print(nums)


def sample_mixed_Gaussian(k1, k2, mu1, sigma1, mu2, sigma2, N):
    N1 = int(N * k1)
    N2 = int(N * k2)

    x1 = np.random.normal(mu1, sigma1, N1)
    x2 = np.random.normal(mu2, sigma2, N2)

    x = jnp.hstack((x1, x2))
    print(x)






# run ==========================================================

#test_flowNet()
test_flow_fn()
#sample_mixed_Gaussian(0.7, 0.3, 0., 1., 0., 1., 100)
#test_mixed_Gaussian()



