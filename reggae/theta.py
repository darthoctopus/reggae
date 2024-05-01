from .utils import beta, normal
from dataclasses import dataclass, astuple
import jax.numpy as jnp
import scipy

@dataclass
class ThetaReg:
    '''
    Parameters for Reggae (asymptotic description of l = 1 modes)

    If theta is a numpy array, generate an instance as
    theta = ThetaReg(*θ)
    '''

    dPi0: float # Period spacing
    p_L: float # Coupling parameter
    p_D: float # Coupling parameter
    epsilon_g: float # g-mode offset
    log_omega_core: float # log10 core splitting
    d01: float # l=0,1 frequency splitting
    alpha_g: float # g-mode curvature term
    inclination: float # inclination stellar rotation axis
    log_omega_env: float # log10 envelope splitting

    dims = 9

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
                   [-2, .5],
                  ])
    normalisation = 1

    beta12 = beta(a=1, b=2)
    normal = normal(mu=0, sigma=1)

    
    @staticmethod
    def prior_transform(u, bounds=None):
        """ Prior transform for inverse sampling

        Evalues the ppf (quantile function) given a set of quantile values u
        drawn from the n-dimensional hypercube. 

        Parameters
        ----------
        u : np.array
            Array of values between 0 and 1 drawn uniformly from the 
            n-dimensional hypercube.
        bounds : np.array
            Array of bounds for the distributions in case they need to be
            truncated.
        """

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
        """ Prior transform for inverse sampling

        Evalues the ppf (quantile function) given a set of quantile values u
        drawn from the n-dimensional hypercube. 

        Parameters
        ----------
        u : np.array
            Array of values between 0 and 1 drawn uniformly from the 
            n-dimensional hypercube.
        bounds : np.array
            Array of bounds for the distributions in case they need to be
            truncated.
        """
        
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
    """ Parameters for the asymptotic description of l = 0,2 modes
    """

    log_numax: float # log10 numax
    log_dnu: float # log10 dnu
    eps: float # p-mode phase offset
    log_d02: float # log10  l=0,2 frequency spacing
    log_alpha: float # log10 p-mode curvature
    log_hmax: float # log10 envelope height
    log_env_width: float # log10 envelope width
    log_mode_width: float # log10 mode width

    dims = 8

    def nmax(self):
        """Compute nmax"""

        numax = 10.**self.log_numax
        dnu = 10.**self.log_dnu
        eps = self.eps
        return numax / dnu - eps

    def n_p(self, n_orders):
        """Build array of radial orders"""
        nmax = self.nmax()
        below = jnp.floor(nmax - jnp.floor(n_orders/2)).astype(int)
        enns = jnp.arange(n_orders) + below
        return enns

    def nu_0(self, n_orders):
        """Compute frequencies of the radial orders"""
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
    """ Parameters for the Harvey-like background terms of the model
    """

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
    """ Parameters to compare with additional observational parameters
    """

    Teff : float
    bprp : float      

    dims = 2