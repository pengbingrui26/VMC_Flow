import jax
import jax.numpy as jnp
#import sympy as sp

class dimer(object):

    def __init__(self, t, U):
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

    def eigs(self):
        E, V = jnp.linalg.eigh(self.H)
        return E, V

    def GS(self):
        E, V = self.eigs()
        idx = jnp.argsort(E)
        return V[:, idx[0]]

    def gwf(self, g):
        t = self.t
        H_free = jnp.array([ [ 0, 0, -t, -t ], \
               [ 0, 0, -t, -t ], \
               [ -t, -t, 0, 0 ], \
               [ -t, -t, 0, 0 ] ])
        E_free, V_free = jnp.linalg.eigh(H_free)
        idx = jnp.argsort(E_free)
        GS_WF = V_free[:, idx[0]]
        #print('GS_WF:', GS_WF)
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

"""           
model = dimer(1, 10)
gwf = model.gwf(0.18)
#print(model.eigs()[0])
#print(gwf)
for g in jnp.arange(0.01, 1., 0.05):
    qgt = model.qgt(g)
    print(qgt, jnp.log(qgt), jnp.log(1e-6+qgt))
exit()


gs = model.GS()
print(gs)
h = model.H
print(jnp.dot( gwf, jnp.dot(h, gwf) ) / jnp.linalg.norm(gwf)**2)
print(jnp.dot( gs, jnp.dot(h, gs) ))
"""


