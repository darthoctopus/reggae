import jax.numpy as jnp
import jax
from functools import partial
jax.config.update('jax_enable_x64', True)

class asymptotic():

    def __init__(self):

        pass

    @partial(jax.jit, static_argnums=(0,))
    def _pair(self, nu0, h, w, d02):
        """Define a pair as the sum of two Lorentzians.

        A pair is assumed to consist of an l=0 and an l=2 mode. The widths are
        assumed to be identical, and the height of the l=2 mode is scaled
        relative to that of the l=0 mode. The frequency of the l=2 mode is the
        l=0 frequency minus the small separation.

        Parameters
        ----------
        nu0 : float
            Frequency of the l=0 (muHz).
        h : float
            Height of the l=0 (SNR).
        w : float
            The mode width (identical for l=2 and l=0) (log10(muHz)).
        d02 : float
            The small separation (muHz).
        hfac : float, optional
            Ratio of the l=2 height to that of l=0 (unitless).

        Returns
        -------
        pair_model : array
            The SNR as a function of frequency of a mode pair.

        """

        pair_model = self._lor(nu0, h, w) + self._lor(nu0 - d02, h * self.V20, w)

        return pair_model


    @partial(jax.jit, static_argnums=(0,))
    def _lor(self, nu0, h, w):
        """ Lorentzian to describe a mode.

        Parameters
        ----------
        nu0 : float
            Frequency of lorentzian (muHz).
        h : float
            Height of the lorentizan (SNR).
        w : float
            Full width of the lorentzian (muHz).

        Returns
        -------
        mode : ndarray
            The SNR as a function frequency for a lorentzian.

        """

        return h / (1.0 + 4.0/w**2*(self.f[self.sel] - nu0)**2)


    # no jax on this
    def _get_nmax(self, dnu, numax, eps):
        """Compute radial order at numax.

        Compute the radial order at numax, which in this implimentation of the
        asymptotic relation is not necessarily integer.

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz).
        dnu : float
            Large separation of l=0 modes (muHz).
        eps : float
            Epsilon phase term in asymptotic relation (muHz).

        Returns
        -------
            nmax : float
                non-integer radial order of maximum power of the p-mode envelope

        """

        nmax = numax / dnu - eps

        return nmax


    @partial(jax.jit, static_argnums=(0,))
    def _get_n_p(self, nmax):
        """Compute radial order numbers.

        Get the enns that will be included in the asymptotic relation fit.
        These are all integer.

        Parameters
        ----------
        nmax : float
            Frequency of maximum power of the p-mode envelope
        norders : int
            Total number of radial orders to consider

        Returns
        -------
        enns : ndarray
                Numpy array of norders radial orders (integers) around nu_max
                (nmax).

        """

        below = jnp.floor(nmax - jnp.floor(self.norders/2)).astype(int)

        enns = jnp.arange(self.norders) + below

        #above = jnp.floor(nmax + jnp.ceil(self.norders/2)).astype(int)

        return enns #jnp.arange(below, above)


    @partial(jax.jit, static_argnums=(0,))
    def _P_envelope(self, nu, hmax, numax, width):
        """ Power of the seismic p-mode envelope

        Computes the power at frequency nu in the p-mode envelope from a Gaussian
        distribution. Used for computing mode heights.

        Parameters
        ----------
        nu : float
            Frequency (in muHz).
        hmax : float
            Height of p-mode envelope (in SNR).
        numax : float
            Frequency of maximum power of the p-mode envelope (in muHz).
        width : float
            Width of the p-mode envelope (in muHz).

        Returns
        -------
        h : float
            Power at frequency nu (in SNR)

        """

        h = hmax * jnp.exp(- 0.5 * (nu - numax)**2 / width**2)

        return h


    @partial(jax.jit, static_argnums=(0,))
    def _get_freq_range(self, numax):
        """ Get frequency range around numax for model
        """

        dnu = self.dnuScale(numax)

        eps = 1.5

        nmax = self._get_nmax(dnu, numax, eps)

        enns = self._get_n_p(nmax)

        lfreq = (enns.min() + 1 + eps) * dnu

        ufreq = (enns.max() + 1 + eps) * dnu

        idx = (lfreq < self.f) & (self.f < ufreq)

        return jnp.array(idx, dtype=bool)

    @partial(jax.jit, static_argnums=(0,))
    def _asymptotic_relation(self, numax, dnu, eps, alpha):
        """ Compute the l=0 mode frequencies from the asymptotic relation for
        p-modes

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz).
        dnu : float
            Large separation of l=0 modes (muHz).
        eps : float
            Epsilon phase term in asymptotic relation (unitless).
        alpha : float
            Curvature factor of l=0 ridge (second order term, unitless).

        Returns
        -------
        nu0s : ndarray
            Array of l=0 mode frequencies from the asymptotic relation (muHz).

        """

        nmax = self._get_nmax(dnu, numax, eps)

        self.n_p = self._get_n_p(nmax)

        return (self.n_p + eps + alpha/2*(self.n_p - nmax)**2) * dnu

    @partial(jax.jit, static_argnums=(0,))
    def asymptotic_model(self, theta_asy):
        """ Constructs a spectrum model from the asymptotic relation.

        The asymptotic relation for p-modes with angular degree, l=0, is
        defined as:

        $nu_nl = (n + \epsilon + \alpha/2(n - nmax)^2) * \log{dnu}$ ,

        where nmax = numax / dnu - epsilon.

        We separate the l=0 and l=2 modes by the small separation d02.

        Parameters
        ----------
        dnu : float
            Large separation log10(muHz)
        lognumax : float
            Frequency of maximum power of the p-mode envelope log10(muHz)
        eps : float
            Phase term of the asymptotic relation (unitless)
        alpha : float
            Curvature of the asymptotic relation log10(unitless)
        d02 : float
            Small separation log10(muHz)
        loghmax : float
            Gaussian height of p-mode envelope log10(SNR)
        logenvwidth : float
            Gaussian width of the p-mode envelope log10(muHz)
        logmodewidth : float
            Width of the modes (log10(muHz))
        *args : array-like
            List of additional parameters (Teff, bp_rp) that aren't actually
            used to construct the spectrum model, but just for evaluating the
            prior.

        Returns
        -------
        model : ndarray
            spectrum model around the p-mode envelope

        """

        numax, dnu, eps, d02, alpha, hmax, env_width, mode_width = theta_asy

        nu0s = self._asymptotic_relation(10**numax, 10**dnu, eps, 10**alpha)

        Hs = self._P_envelope(nu0s, 10**hmax, 10**numax, 10**env_width)

        mod = jnp.zeros(len(self.f[self.sel]))

        for n in range(len(nu0s)):

            mod += self._pair(nu0s[n], Hs[n], 10**mode_width, 10**d02)

        return mod * self.eta[self.sel]


    # No jax on this one
    def __get_enns(self, nmax):
        """Compute radial order numbers.

        This is not jaxxed.

        Get the enns that will be included in the asymptotic relation fit.
        These are all integer.

        Parameters
        ----------
        nmax : float
            Frequency of maximum power of the p-mode envelope
        norders : int
            Total number of radial orders to consider

        Returns
        -------
        enns : ndarray
                Numpy array of norders radial orders (integers) around nu_max
                (nmax).

        """

        below = jnp.floor(nmax - jnp.floor(self.norders/2)).astype(int)

        above = jnp.floor(nmax + jnp.ceil(self.norders/2)).astype(int)

        # Handling of single input (during fitting), or array input when evaluating
        # the fit result
        if type(below) != jnp.ndarray:
            return jnp.arange(below, above)
        else:
            out = jnp.concatenate([jnp.arange(x, y) for x, y in zip(below, above)])
            return out.reshape(-1, self.norders)