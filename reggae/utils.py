#!/usr/bin/env python3
from functools import partial
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax
import numpy as np
import lightkurve as lk
import os
jax.config.update('jax_enable_x64', True)


def readBkgFits(ID):
    """ Read background samples for bog prior
    
    Read in the previosly fit background model parameters to use as a prior
    on the bog sampling. 

    This will be removed once the background parameters are put into the prior
    and become part of the DR.
    """  
    
    fname = os.path.join(*['Examples', ID+'_bkg_samples.npz'])
    
    samples = np.load(fname)['samples']
    
    percs = np.percentile(samples, [16, 50, 84], axis=0)
    
    mu = percs[1, :]

    err = np.mean(np.diff(percs, axis = 0), axis=0)
    
    out = {}
    
    keys = ['hsig1', 'dhnu1', 'hexp1', 
            'hsig2', 'dhnu2', 'hexp2', 
            'hsig3', 'dhnu3', 'hexp3'] # assumes ordered dictionary, stick to this order
    
    for i, key in enumerate(keys):
        out[key] = (mu[i], err[i])
        
    out['w'] = (mu[-1], err[-1])
    
    return out



# Jaxxed
class jaxInterp1D():
    """ Replacement for scipy.interpolate.interp1d in jax"""

    def __init__(self, xp, fp, left=None, right=None, period=None):
        self.xp = xp
        self.fp = fp
        self.left = left
        self.right = right
        self.period = period

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):

        return jnp.interp(x, self.xp, self.fp, self.left, self.right, self.period)


# Jaxxed
class scalingRelations():
    """ Container for scaling relations

    This is a helper class which contains methods for the various scaling
    relations.

    """

    def __init_(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def dnuScale(self, nu, gamma=0.0, p=[0.79101684, -0.63285292]):
        """ Compute dnu from numax

        Computes an estimate of the large separation from a given value of numax,
        assuming the two parameters scale as a polynomial in log(dnu) and
        log(numax).

        The default is a linear function in log(dnu) and log(numax), estimated
        based on a polynomial fit performed on a set of main-sequence and sub-
        giant stars in the literature.

        The output may be scaled by a factor gamma, e.g., for setting credible
        intervals.

        Parameters
        ----------
        nu : float, array
            Value(s) at which to compute dnu, assuming nu corresponds to numax.
        gamma : float
            Scaling factor to apply.
        p : array-like, optional
            Polynomial coefficients to use for log(dnu), log(numax), starting with
            the coefficient of the Nth order term and ending at the bias term.

        Returns
        -------
        dnu : float, array
            Estimate of the large separation dnu
        """
        dnu = 10**(jnp.polyval(jnp.array(p), jnp.log10(nu))+gamma)
        return dnu

    @partial(jax.jit, static_argnums=(0,))
    def envWidth(self, numax):
        """ Scaling relation for the envelope width

        Computest he full width at half maximum of the p-mode envelope based
        on numax and Teff (optional).

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope.
        Teff : float, optional
            Effective surface temperature of the star.
        Teff0 : float, optional
            Solar effective temperature in K. Default is 5777 K.

        Returns
        -------
        width : float
            Envelope width in muHz
        """

        width = 0.66*numax**0.88 # Mosser et al. 201??

        return width

# Jaxxed


@jax.jit
def to_log10(x, xerr):
    """ Transform to value to log10

    Takes a value and related uncertainty and converts them to logscale.
    Approximate.

    Parameters
    ----------
    x : float
        Value to transform to logscale
    xerr : float
        Value uncertainty

    Returns
    -------
    logval : list
        logscaled value and uncertainty

    """



    L = jax.lax.cond(xerr > 0,
                     lambda A: jnp.array([jnp.log10(A[0]), A[1]/A[0]/jnp.log(10.0)]),
                     lambda A: A,
                     jnp.array([x, xerr]))

    return L

# Jaxxed
class normal():

    def __init__(self, **kwargs):
        """ normal distribution class

        Create instances a probability density which follows the normal
        distribution.

        Parameters
        ----------

        mu : float
            The mean of the normal distribution.
        sigma : float
            The standard deviation of the normal distribution.
        """

        allowed_keys = {'mu', 'sigma'}

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.norm = 1 / (jnp.sqrt(2*jnp.pi) * self.sigma)

        self.lognorm = jnp.log(self.norm)

        self.fac = -0.5 / self.sigma**2



    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x, norm=True):
        """ Return PDF

        Returns the normal distribution at x. The distribution is normalized to
        unit integral by default so that it may be used as a PDF.

        In some cases the normalization is not necessary, and since it's
        marginally slower it may as well be left out.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized normal distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the normal distribution at x.

        """
        y = jnp.exp( self.fac * (x - self.mu)**2)

        Y = jax.lax.cond(norm,
                         lambda y: y * self.norm,
                         lambda y: y ,
                         y)

        return Y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x, norm=True):
        """ Return log-PDF

        Returns the log of the normal distribution at x. The distribution is
        normalized to unit integral (in linear units) by default so that it
        may be used as a PDF.

        In some cases the normalization is not necessary, and since it's
        marginally slower it may as well be left out.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized normal distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the logarithm of the normal distribution at x.
        """

        y = self.fac * (x - self.mu)**2

        Y = jax.lax.cond(norm,
                     lambda y: y + self.lognorm,
                     lambda y: y,
                     y)

        return Y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):

        y = 0.5 * (1 + jsp.erf((x-self.mu)/(jnp.sqrt(2)*self.sigma)))

        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):

        x = self.mu + self.sigma*jnp.sqrt(2)*jsp.erfinv(2*y-1)

        return x



class uniform():

    def __init__(self, **kwargs):
        allowed_keys = {'loc', 'scale'}

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.a = self.loc

        self.b = self.loc + self.scale

        self.mu = 0.5 * (self.a + self.b)

    def pdf(self, x):

        x = jnp.array(x)

        if isinstance(x, (jnp.ndarray, )):

            idx = (self.a <= x) & (x <= self.b)

            y = jnp.zeros_like(x)

            y = y.at[idx].set(1/self.scale)

        elif isinstance(x, (float, int)):
            if (x <= self.a) or (self.b <= x):
                y = 0

            else:
                y = 1/self.scale

        return y

    def logpdf(self, x):

        x = jnp.array(x)

        if isinstance(x, (jnp.ndarray, )):

            y = jnp.zeros_like(x) - jnp.inf

            idx = (self.a <= x) & (x <= self.b)

            y = y.at[idx].set(-jnp.log(self.scale))

        elif isinstance(x, (float, int)):
            if (x <= self.a) or (self.b <= x):
                y = -jnp.inf

            else:
                y = -jnp.log(self.scale)

        return y

    def cdf(self, x):

        x = jnp.array(x)

        if isinstance(x, (jnp.ndarray, )):

            y = jnp.zeros_like(x)

            idx = (self.a <= x) & (x <= self.b)

            _y = (x[idx] - self.a) / (self.b - self.a)

            y = y.at[idx].set(_y)

            y = y.at[x>self.b].set(1)

        elif isinstance(x, (float, int)):
            if x == self.a:
                y = 0

            elif x > self.b:
                y = 1

            else:
                y = (x - self.a) / (self.b - self.a)

        return y

    #@partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):

        #y = jnp.array(y)

        x = y * (self.b - self.a) + self.a

        return x


class beta():
    def __init__(self, **kwargs):
        """ beta distribution class

        Create instances a probability density which follows the beta
        distribution.

        Parameters
        ----------
        a : float
            The first shape parameter of the beta distribution.
        b : float
            The second shape parameter of the beta distribution.
        loc : float
            The lower limit of the beta distribution. The probability at this
            limit and below is 0.
        scale : float
            The width of the beta distribution. Effectively sets the upper
            bound for the distribution, which is loc+scale.
        eps : float, optional
            Small fudge factor to avoid dividing by 0 etc.
        """

        allowed_keys = {'a', 'b', 'loc', 'scale', 'eps'}
        
        self.loc = 0
        self.scale = 1

        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.fac = jnp.exp(jsp.gammaln(self.a + self.b)) / (jnp.exp(jsp.gammaln(self.a)) * jnp.exp(jsp.gammaln(self.b))) / self.scale

        self.logfac = jnp.log(self.fac)

        self.am1 = self.a-1

        self.bm1 = self.b-1

    def _transformx(self, x):
        """ Transform x

        Translates and scales the input x to the unit interval according to
        the loc and scale parameters.

        Parameters
        ----------
        x : array
            Input support for the probability density.

        Returns
        -------
        _x : array
            x translated and scaled to the range 0 to 1.

        """
        return (x - self.loc) / self.scale

    def _inverse_transform(self, x):

        return x * self.scale + self.loc

    def pdf(self, x, norm=True):
        """ Return PDF

        Returns the beta distribution at x. The distribution is normalized to
        unit integral by default so that it may be used as a PDF.

        In some cases the normalization is not necessary, and since it's
        marginally slower it may as well be left out.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized beta distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the beta distribution at x.

        """

        x = jnp.array(x)

        _x = self._transformx(x)

        if isinstance(x, (jnp.ndarray, )):

            y = jnp.zeros_like(_x)

            idx = (0 < _x) & (_x < 1)

            _y = _x[idx]**self.am1 * (1 - _x[idx])**self.bm1

            y = y.at[idx].set(_y)


        elif isinstance(x, (float, int)):
            if _x <=0 or 1 <= _x:
                y = -jnp.inf

            else:
                y = _x**self.am1 * (1 - _x)**self.bm1

        if norm:
            return y * self.fac

        else:
            return y

    def logpdf(self, x, norm=True):
        """ Return log-PDF

        Returns the log of the beta distribution at x. The distribution is
        normalized to unit integral (in linear units) by default so that it
        may be used as a PDF.

        In some cases the normalization is not necessary, and since it's
        marginally slower it may as well be left out.

        Parameters
        ----------
        x : array
            Input support for the probability density.
        norm : bool, optional
            If true, returns the normalized beta distribution. The default is
            True.

        Returns
        -------
        y : array
            The value of the logarithm of the beta distribution at x.
        """

        x = jnp.array(x)

        _x = self._transformx(x)

        if isinstance(x, (jnp.ndarray)):

            y = jnp.zeros_like(_x) - jnp.inf

            idx = (0 < _x) & (_x < 1)

            _y = self.am1 * jnp.log(_x[idx]) + self.bm1 * jnp.log(1-_x[idx])

            y = y.at[idx].set(_y)

        elif isinstance(x, (float, int)):
            if _x <=0 or _x>=1:

                y = -jnp.inf

            else:
                y = self.am1 * jnp.log(_x) + self.bm1 * jnp.log(1-_x)

        if norm:
            return y + self.logfac
        else:
            return y

    def cdf(self, x):

        _x = self._transformx(x)

        y = jsp.betainc(self.a, self.b, _x)

        y = y.at[_x<=0].set(0)

        y = y.at[_x>=1].set(1)

        return y

    def ppf(self, y):

        _x = betaincinv(self.a, self.b, y)

        x = self._inverse_transform(_x)

        return x

@jax.jit
def update_x(x, a, b, p, a1, b1, afac):
    err = jsp.betainc(a, b, x) - p
    t = jnp.exp(a1 * jnp.log(x) + b1 * jnp.log(1.0 - x) + afac)
    u = err/t
    tmp = u * (a1 / x - b1 / (1.0 - x))
    t = u/(1.0 - 0.5 * jnp.clip(tmp, a_max=1.0))
    x -= t
    x = jnp.where(x <= 0., 0.5 * (x + t), x)
    x = jnp.where(x >= 1., 0.5 * (x + t + 1.), x)

    return x, t

@jax.jit
def func_1(a, b, p):
    pp = jnp.where(p < .5, p, 1. - p)
    t = jnp.sqrt(-2. * jnp.log(pp))
    x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
    x = jnp.where(p < .5, -x, x)
    al = (jnp.power(x, 2) - 3.0) / 6.0
    h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
    w = (x * jnp.sqrt(al + h) / h)-(1.0 / (2.0 * b - 1) - 1.0/(2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h))
    return a / (a + b * jnp.exp(2.0 * w))

@jax.jit
def func_2(a, b, p):
    lna = jnp.log(a / (a + b))
    lnb = jnp.log(b / (a + b))
    t = jnp.exp(a * lna) / a
    u = jnp.exp(b * lnb) / b
    w = t + u

    return jnp.where(p < t/w, jnp.power(a * w * p, 1.0 / a), 1. - jnp.power(b *w * (1.0 - p), 1.0/b))

@jax.jit
def compute_x(p, a, b):
    return jnp.where(jnp.logical_and(a >= 1.0, b >= 1.0), func_1(a, b, p), func_2(a, b, p))

@jax.jit
def betaincinv(a, b, p):
    a1 = a - 1.0
    b1 = b - 1.0

    ERROR = 1e-8

    p = jnp.clip(p, a_min=0., a_max=1.)

    x = jnp.where(jnp.logical_or(p <= 0.0, p >= 1.), p, compute_x(p, a, b))

    afac = - jsp.betaln(a, b)
    stop  = jnp.logical_or(x == 0.0, x == 1.0)
    for i in range(10):
        x_new, t = update_x(x, a, b, p, a1, b1, afac)
        x = jnp.where(stop, x, x_new)
        stop = jnp.where(jnp.logical_or(jnp.abs(t) < ERROR * x, stop), True, False)

    return x
