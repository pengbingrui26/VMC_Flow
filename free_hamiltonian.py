import jax 
import jax.numpy as jnp

class H_free(object):
    def __init__(self, Lsite, N, t):
        self.Lsite = Lsite
        self.N = N
        self.t = t

    def get_matr(self):
        H_free_half = -self.t*jnp.eye(self.Lsite, k = 1) - self.t*jnp.eye(self.Lsite, k = -1)
        H_free_half = H_free_half.at[0, self.Lsite-1].set(-self.t)
        H_free_half = H_free_half.at[self.Lsite-1, 0].set(-self.t)  
       
        H_free = jnp.zeros((self.Lsite*2, self.Lsite*2))
        H_free = H_free.at[ :self.Lsite, :self.Lsite ].set(H_free_half)
        H_free = H_free.at[ self.Lsite:, self.Lsite: ].set(H_free_half)
        return H_free
        
    def get_eig(self):
        eigvals, eigvecs = jnp.linalg.eigh(self.get_matr())
        return eigvals, eigvecs
   
    def get_psi0(self):  
        # < x | Psi >, where Psi is the non-interacting ground state
        E, U = self.get_eig()
        sort_indice = jnp.argsort(E)
        U_new = U[:, sort_indice[:self.N*2]] # its k-th column represents the k-th lowest eigenstate
        return U_new
 
