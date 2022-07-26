import jax 
import jax.numpy as jnp


def log_psi(psi, Lsite, N):

    def _make_logpsi0(state):
        U = psi[list(state),:]
        phase, logabsdet = jnp.linalg.slogdet(U)
        #print('phase:', phase) 
        logpsi = logabsdet + jnp.log(phase)    
        return jnp.array( [ logpsi.real, logpsi.imag ] )

    def _make_psi0_ratio_fast(x, x1):
        """
        W_{K,l} = \sum_{a} U_{K,a} * \hat{U}_{a, l}  
        """
        U_x = psi[x, :]
        W = jnp.dot( psi, jnp.linalg.inv(U_x) )
        """
        | x > = |n_1, ... , n_k, ... , n_l, n_L >
        | x1 > = |n_1, ... , n_k + 1, ... , n_l - 1, n_L >
        """
        dx = x1 - x
        x_test = jnp.arange(N*2) + 1
        idx = jnp.where(dx == 0, 0, x_test)
        l = jnp.sort(idx)[-1] - 1
        K = x1[l]
        ratio = W[K][l] 
        return ratio

    def _make_gutzwiller_factor(state, g):
        """
        compute the Gutzwiller factor g^{D(x)} for a single state x, 
        where D(x) is the number of double occupation
        """
        up = state[None, :N]
        down = state[N:, None]
        diff = up + Lsite - down
        double_occ_state = jnp.sum( jnp.where(diff == 0, 1, 0) )
        #g = jnp.sqrt(g)
        return jnp.power( g, double_occ_state ) 

    def _make_gutzwiller_ratio(state0, state1, g):
        up1 = state1[None, :N]
        down1 = state1[N:, None]
        diff1 = up1 + Lsite - down1
        double_occ_state1 = jnp.sum( jnp.where(diff1 == 0, 1, 0) )
     
        up0 = state0[None, :N]
        down0 = state0[N:, None]
        diff0 = up0 + Lsite - down0
        double_occ_state0 = jnp.sum( jnp.where(diff0 == 0, 1, 0) )

        #g = jnp.sqrt(g)
        return jnp.power( g, double_occ_state1 ) / jnp.power( g, double_occ_state0 )

    def make_psi_ratio(state0, state1, g):
        psi0_ratio = _make_psi0_ratio_fast(state0, state1)
        gutz_ratio = _make_gutzwiller_ratio(state0, state1, g)
        return jnp.multiply(psi0_ratio, gutz_ratio)

    def make_logpsi(state, g):
        """
        ln |Psi(x)|^2, Psi(x) = Psi_0(x) * g^{D(x)}
        """
        logpsi0 = _make_logpsi0(state) 
        gutzwiller_factor = _make_gutzwiller_factor(state, g)

        #return logpsi0 + jnp.array( [jnp.log(gutzwiller_factor), 0 ] )
        return logpsi0[0] + jnp.log(gutzwiller_factor)

    def grad_g_logpsi(state, g):
        up = state[None, :N]
        down = state[N:, None]
        diff = up + Lsite - down
        double_occ = jnp.sum( jnp.where(diff == 0, 1, 0) )

        #g = jnp.sqrt(g)
        return jnp.power(g, -1) * double_occ

    return make_psi_ratio, make_logpsi, grad_g_logpsi

