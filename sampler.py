import jax
import jax.numpy as jnp
import haiku as hk

def Gaussian_sampler(shape, key, mu, sigma):
    x = jax.random.normal(key = key, shape = shape)
    return sigma * x + mu


def Gaussian_prob(z, mu, sigma):
    return 1/(jnp.sqrt(2*jnp.pi)*sigma) * jnp.exp(-(z-mu)**2/(2*sigma**2))
 

def masker(x):
    return jnp.array([ xx for xx in x if xx < 1. and xx > 0. ])


def flow_fn_naive(batch_g, mu, sigma):
    shape = (batch_g, )
    key = jax.random.PRNGKey(21) 

    z = Gaussian_sampler(shape, key, mu, sigma)

    def fn(params):
        w1, w2, b1, b2 = params[0], params[1], params[2], params[3]
        #return w2 * jnp.tanh(w1 * z + b1) 
        return jnp.tanh( w2 * jnp.tanh(w1 * z + b1) + b2 )

    def log_prob(params):
        w1, w2, b1, b2 = params[0], params[1], params[2], params[3]
 
        pz = Gaussian_prob(z, mu, sigma)
        ln_pz = jnp.log(pz)
        #g = w2 * jnp.tanh(w1*z + b1) 
   
        def trans(x):
            #return w2 * jnp.tanh(w1 * x + b1)  
            return jnp.tanh( w2 * jnp.tanh(w1 * x + b1) + b2 )

        #jac = jax.jacrev(lambda x: w2 * jnp.tanh(w1 * x + b1))(z)
        jac = jax.jacrev(trans)(z)
        jac = jnp.linalg.norm(jac)
    
        return ln_pz - jnp.log(jac)

    return fn, log_prob



class flowNet(hk.Module):

    def __init__(self, Lsize, init_stddev = 0.01):
        super().__init__()
        self.Lsize = Lsize
        self.init_stddev = init_stddev

    def __call__(self, x):
        w1_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        #w1_init = -jnp.ones(self.Lsize)
        #print('w1_init:')
        #print(w1_init)
        w1 = hk.get_parameter("w1", shape=(self.Lsize,), dtype=x.dtype, init=w1_init)
        w1 = jax.nn.softplus(w1-2.)
        #print('w1:')
        #print(w1)
        b1_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        b1 = hk.get_parameter("b1", shape=(self.Lsize,), dtype=x.dtype, init=b1_init)
          
        x1 = jnp.tanh(w1 * x + b1)

        w2_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        w2 = hk.get_parameter("w2", shape=(self.Lsize,), dtype=x.dtype, init=w2_init)
        w2 = jax.nn.softplus(w2-2.)
        #print('w2:')
        #print(w2)

        x2 = jnp.dot(w2, x1)
        #print('x2:')
        #print(x2)

        return x2      
        #return x + x2



def flow_fn(batch_g, mu, sigma, Lsize, key):
    shape = (batch_g, )
    #key = jax.random.PRNGKey(42)
    z = Gaussian_sampler(shape, key, mu, sigma)
    print('z_mean, z_std:', z.mean(), z.std())

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
        return flow_forward_vmapped(params, None, z)
 
    def log_prob(params):
        pz = Gaussian_prob(z, mu, sigma)
        ln_pz = jnp.log(pz)
   
        jac = flow_grad_vmapped(z, params)
        #print('jac:')
        #print(jac)
        jac = jnp.linalg.norm(jac[:,None], axis = -1)
        #print('jac_normed:')
        #print(jac)    

        return ln_pz - jnp.log(jac)

    def grad_logprob(params):
        return jax.jacrev(log_prob)(params)

    #return params_init, make_g, log_prob
    return z, params_init, make_g, log_prob, grad_logprob



