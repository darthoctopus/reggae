from .utils import beta, normal
from dataclasses import dataclass, astuple
import jax.numpy as jnp

@dataclass
class ThetaReg:
    '''
    Parameters for Reggae (asymptotic description of l = 1 modes)

    If θ is a numpy array, generate an instance as
    theta = ThetaReg(*θ)
    '''

    dPi0: float
    p_L: float
    p_D: float
    epsilon_g: float
    log_omega_core: float
    d01: float
    alpha_g: float
    inclination: float

    dims = 8

    def asarray(self):
        return jnp.array(astuple(self))

    bounds = list([[3.7, 3.8],
                   [0, 7],
                   [0, 7],
                   [0, 1],
                   [-2, .5],
                   [-.1, .1],
                   [-.01, .01],
                   [0, jnp.pi/2],
                  ])
    normalisation = 1

    beta12 = beta(a=1, b=2)
    normal = normal(mu=0, sigma=1)

    #TODO: jax this up
    @staticmethod
    def prior_transform(u, bounds=None):
        if bounds is None:
            bounds = ThetaReg.bounds
        θ = [a + (b-a)*t for (a, b), t in zip(bounds, u)]
        θ[1] = ThetaReg.beta12.ppf(u[1]) * bounds[1][1] + bounds[1][0]
        θ[2] = ThetaReg.normal.ppf(u[2]) * bounds[2][1] + bounds[2][0]
        θ[7] = jnp.arccos(u[7])
        #θ[-1] = ThetaReg.normal.ppf(u[-1]) * bounds[-1][1] + bounds[-1][0]
        
        return jnp.array(θ)

    @staticmethod
    def inv_prior_transform(θ, bounds=None):
        if bounds is None:
            bounds = ThetaReg.bounds
        x = jnp.array([(ξ - a) / (b - a) for (a, b), ξ in zip(bounds, θ)])
        x[1] = scipy.stats.beta.cdf((θ[1] - bounds[1][0]) / bounds[1][1], 1, 2)
        x[2] = scipy.stats.norm.cdf((θ[2] - bounds[2][0]) / bounds[2][1])
        x[7] = jnp.cos(θ[7])
        #x[-1] = scipy.stats.norm.cdf((θ[-1] - bounds[-1][0]) / bounds[-1][1])
        
        return x

@dataclass
class ThetaAsy:
    log_numax: float
    log_dnu: float
    eps: float
    log_d02: float
    log_alpha: float
    log_hmax: float
    log_env_width: float
    log_mode_width: float

    dims = 8

    def nmax(self):
        numax = 10.**self.log_numax
        dnu = 10.**self.log_dnu
        eps = self.eps
        return numax / dnu - eps

    def n_p(self, n_orders):
        nmax = self.nmax()
        below = jnp.floor(nmax - jnp.floor(n_orders/2)).astype(int)
        enns = jnp.arange(n_orders) + below
        return enns

    def nu_0(self, n_orders):
        nmax = self.nmax()
        dnu = 10.**self.log_dnu
        eps = self.eps
        alpha = 10.**self.log_alpha
        n_p = self.n_p(n_orders)
        return dnu * (n_p + eps + alpha/2*(n_p - nmax)**2)

    def asarray(self):
        return jnp.array(astuple(self))

@dataclass
class ThetaBkg:
    hsig1 : float
    dhnu1 : float
    exp1 : float
    hsig2 : float
    dhnu2 : float
    exp2 : float
    hsig3 : float
    hnu3 : float
    exp3 : float
    white : float

    dims = 10
    def asarray(self):
        return jnp.array(astuple(self))

@dataclass
class ThetaObs:
    Teff : float
    bprp : float      

    dims = 2