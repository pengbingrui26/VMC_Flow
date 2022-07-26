import jax
import jax.numpy as jnp
import haiku as hk
import scipy as sp



class flowNet(hk.Module):

    def __init__(self, Lsize, init_stddev = 0.1):
        super().__init__()
        self.Lsize = Lsize
        self.init_stddev = init_stddev

    def __call__(self, x):
        w1_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        w1 = hk.get_parameter("w1", shape=(self.Lsize,), dtype=float, init=w1_init)
        #w1 = jax.nn.softplus(w1)

        b1_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        b1 = hk.get_parameter("b1", shape=(self.Lsize,), dtype=float, init=b1_init)
          
        x1 = jnp.tanh(w1 * x + b1)

        w2_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        w2 = hk.get_parameter("w2", shape=(self.Lsize,), dtype=float, init=w2_init)
        #w2 = jax.nn.softplus(w2)

        x2 = jnp.dot(w2, x1)
        #return x2      
        return jax.nn.sigmoid(x2)



class flowNet_2layer(hk.Module):

    def __init__(self, Lsize, init_stddev = 0.1):
        super().__init__()
        self.Lsize = Lsize
        self.init_stddev = init_stddev

    def __call__(self, x):
        w1_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        w1 = hk.get_parameter("w1", shape=(self.Lsize,), dtype=float, init=w1_init)

        b1_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        b1 = hk.get_parameter("b1", shape=(self.Lsize,), dtype=x.dtype, init=b1_init)
          
        x1 = jax.nn.relu(w1 * x + b1)

        w2_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        w2 = hk.get_parameter("w2", shape=(self.Lsize, self.Lsize), dtype=float, init=w2_init)

        b2_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        b2 = hk.get_parameter("b2", shape=(self.Lsize,), dtype=float, init=b2_init)
 
        x2 = jax.nn.relu(jnp.dot(w2, x1) + b2)

        w3_init = hk.initializers.RandomNormal(stddev=self.init_stddev)
        w3 = hk.get_parameter("w3", shape=(self.Lsize, ), dtype=float, init=w3_init)

        x3 = jnp.dot(w3, x2)

        return jax.nn.sigmoid(x3)




class dimer(object):

    def __init__(self, t, U, flow_fn):
        self.t = t
        self.U = U
        self.H = jnp.array([ [ 0, 0, -t, -t ], \
               [ 0, 0, -t, -t ], \
               [ -t, -t, U, 0 ], \
               [ -t, -t, 0, U ] ])
        self.H1 = jnp.array([ [ -U/2, 0, -t, -t ], \
               [ 0, -U/2, -t, -t ], \
               [ -t, -t, U/2, 0 ], \
               [ -t, -t, 0, U/2 ] ])
 
        self.flow_fn = flow_fn

    def eigs(self):
        E, V = jnp.linalg.eigh(self.H)
        return E, V

    def GS(self):
        E, V = self.eigs()
        idx = jnp.argsort(E)
        return V[:, idx[0]]

    def Free_energy(self, beta):
        Es, Vs = self.eigs()
        Z = 0.
        F = 0.
        for E in Es:
            Z += jnp.exp(-beta*E)
        F = -1/beta * jnp.log(Z)
        return Z, F

    def gwf(self, g):
        t = self.t
        H_free = jnp.array([ [ 0, 0, -t, -t ], \
               [ 0, 0, -t, -t ], \
               [ -t, -t, 0, 0 ], \
               [ -t, -t, 0, 0 ] ])
        E_free, V_free = jnp.linalg.eigh(H_free)
        idx = jnp.argsort(E_free)
        GS_WF = V_free[:, idx[0]]
        Gutz_weight = jnp.array([ 1, 1, g, g ] )
        Gutz_WF = jnp.multiply(Gutz_weight, GS_WF)
        return Gutz_WF

    def make_g(self, a, params):
        return self.flow_fn(params, None, a)

    def grad_g(self, a, params):
        return jax.grad(self.flow_fn, argnums = -1)(params, None, a)

    def gwf_flow(self, a, params):
        t = self.t
        H_free = jnp.array([ [ 0, 0, -t, -t ], \
               [ 0, 0, -t, -t ], \
               [ -t, -t, 0, 0 ], \
               [ -t, -t, 0, 0 ] ])
        E_free, V_free = jnp.linalg.eigh(H_free)
        idx = jnp.argsort(E_free)
        GS_WF = V_free[:, idx[0]]

        g = self.flow_fn(params, None, a)
        #print('g', g)
        Gutz_weight = jnp.array([ 1, 1, g, g ] )
        Gutz_WF = jnp.multiply(Gutz_weight, GS_WF)
        return Gutz_WF

    def qgt(self, g):
        matr_double_occ = jnp.array([ [0,0,0,0], \
                                     [0,0,0,0], \
                                     [0,0,1,0], \
                                     [0,0,0,1] ])
        grad_g = jnp.power(g, -1) * matr_double_occ
        grad_g_square = jnp.dot(grad_g, grad_g)
        gwf = self.gwf(g)
        basis = jnp.array([1,1,1,1])
        A = jnp.dot(gwf, jnp.dot(grad_g_square, gwf)) / jnp.dot(gwf, gwf)
        b = jnp.dot(gwf, jnp.dot(grad_g, gwf)) / jnp.dot(gwf, gwf)
        B = b * b
        qgt = A - B
        return qgt    

    def qgt_autograd(self, g):
        gwf = self.gwf(g)
        grad_gwf = jax.jacrev(self.gwf)(g)    
        A = jnp.dot(grad_gwf, grad_gwf) / jnp.dot(gwf, gwf)
        b = jnp.dot(gwf, grad_gwf) / jnp.dot(gwf, gwf)
        B = b * b
        qgt = A - B
        return qgt    

    def qgt_flow(self, a, params):
        psi = self.gwf_flow(a, params)
        grad_psi = jax.jacrev(self.gwf_flow, argnums = 0)(a, params)
        #print('grad_psi:')
        #print(grad_psi)
        qgt = jnp.dot(grad_psi, grad_psi) / jnp.dot(psi, psi) \
                 - jnp.dot(grad_psi, psi) * jnp.dot(psi, grad_psi) / (jnp.dot(psi, psi) **2 )       
        return qgt

    def boltz(self, g, beta):
        gwf = self.gwf(g)
        dis = sp.linalg.expm(-beta * self.H)
        return jnp.dot(gwf, jnp.dot(dis, gwf)) / jnp.dot(gwf, gwf)

    def Veff(self, g, beta):
        gwf = self.gwf(g)
        E = jnp.dot(gwf, jnp.dot(self.H, gwf)) / jnp.dot(gwf, gwf)
        qgt = self.qgt(g)
        return E - 1/(2*beta) * jnp.log(qgt)

    def Veff_flow(self, a, params, beta):
        gwf = self.gwf_flow(a, params)
        E = jnp.dot(gwf, jnp.dot(self.H, gwf)) / jnp.dot(gwf, gwf)
        qgt = self.qgt_flow(a, params)
        Veff = E - 1/(2*beta) * jnp.log(qgt)
        return Veff, E, qgt



