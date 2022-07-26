import numpy as np
import jax 
import jax.numpy as jnp
import haiku as hk
from functools import partial

from sampler import Gaussian_prob, flowNet, flow_fn


def mixed_Gaussian_sampler(k1, k2, mu1, sigma1, mu2, sigma2, N):
    N1 = int(N * k1)
    N2 = int(N * k2)

    x1 = np.random.normal(mu1, sigma1, N1)
    x2 = np.random.normal(mu2, sigma2, N2)

    x = jnp.hstack((x1, x2))
    print(x)


def flow_rev_fn(batch_g, k1, k2, mu1, m2, sigma1, sigma2, Lsize):
    shape = (batch_g, )
    key = jax.random.PRNGKey(21) 
    g = mixed_Gaussian_sampler(k1, k2, mu1, sigma1, mu2, sigma2, batch_g)

    def forward_fn_flow(x):
        module = flowNet(Lsize)
        return module(x)

    make_flow_forward = hk.transform(forward_fn_flow)

    key_dummy = jax.random.PRNGKey(21)
    x_dummy = jax.random.uniform(key_dummy, minval=0., maxval=1.)
    params_init = make_flow_forward.init(key_dummy, x_dummy)
    flow_forward = make_flow_forward.apply

    def flow_grad(x, params):
        return jax.grad(flow_forward, argnums = -1)(params, None, x)

    flow_forward_vmapped = jax.vmap(flow_forward, in_axes = (None, None, 0), out_axes = 0)
    flow_grad_vmapped = jax.vmap(flow_grad, in_axes = (0, None), out_axes = 0)

    def make_g(params):
        return flow_forward_vmapped(params, None, g)
 
    def log_prob(params):
        pz = Gaussian_prob(z, mu, sigma)
        ln_pz = jnp.log(pz)
   
        jac = flow_grad_vmapped(z, params)
        jac = jnp.linalg.norm(jac[:,None])
    
        return ln_pz - jnp.log(jac)

    return params_init, make_g, log_prob


