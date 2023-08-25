from functools import partial
import numpy as np
from astropy import units as u
from lightkurve.periodogram import SNRPeriodogram

# optimisation
from scipy.optimize import minimize
from yabox import DE
import dynesty
import jax
import jax.numpy as jnp

#pbjam
import pbjam

# self library imports
from .theta import ThetaAsy, ThetaReg
from .psdmodel import PSDModel
from .reggae import reggae

class DipoleStar:

    labels=[r'$\Delta\Pi_0$ (relative)', '$L_0$', '$D_0$', r'$\epsilon_g$',
                r'$\log \omega_\mathrm{core}/\mu$Hz', r'$\delta\epsilon_{p1}$',
                r'$\alpha_g$', r'$i$/rad',r'$\log \omega_\mathrm{env}/\mu$Hz',
                r'Normalisation'
               ]

    def __init__(self, s=None, f=None):
        self.s = s
        self.f = f

    @classmethod
    def from_pbjam(klass, star):

        if isinstance(star, pbjam.star):

            # divide out l = 0,2 modes
            pbjam_model = klass.make_pbjam_model(star)

            self = klass(np.array(star.s / pbjam_model), star.f)
            self.s_raw = np.array(star.s)

            self.pg = star.pg
            self.pg.power = u.Quantity(self.s)

        elif isinstance(star, pbjam.modeID.modeIDsampler):

            # divide out only background
            pbjam_model = klass.make_pbjam_model(star)

            self = klass(np.array(star.s / pbjam_model), star.f)
            self.s_raw = self.s

            self.pg = SNRPeriodogram(self.f * u.uHz, u.Quantity(self.s))

        # theta_asy

        self.nu0, self.nu2, self.theta_asy = self._prepare_theta_asy(star)
        self.norders = len(self.nu0)
        self.nmax = 10**(self.theta_asy.log_numax - self.theta_asy.log_dnu) - self.theta_asy.eps
        self.l1model = PSDModel(self.f, self.norders)

        self.bounds = self.get_bounds()
        self.l1model.n_g = self.select_n_g()

        self.soften = 1

        self.ID = getattr(star, 'ID', None)

        return self

    def __call__(self, dynamic=False, **kwargs):

        kwargs = {**dict(periodic=[3], reflective=[7]), **kwargs}
        ndim = ThetaReg.dims + 1
        if dynamic:
            sampler = dynesty.DynamicNestedSampler(self.ln_like, self.ptform, ndim, **kwargs) 
        else:
            sampler = dynesty.NestedSampler(self.ln_like, self.ptform, ndim, **kwargs)

        self.sampler = sampler
        sampler.run_nested()
        return self.summarise()

    def summarise(self, sampler=None):
        if sampler is None:
            sampler = getattr(self, "sampler", None)

        if sampler is not None:
            sampler = self.sampler
            self.DYresult = sampler.results
            samples = self.DYresult.samples
            weights = np.exp(self.DYresult.logwt - self.DYresult.logz[-1])
            mean, cov = dynesty.utils.mean_and_cov(samples, weights)
            new_samples = dynesty.utils.resample_equal(samples, weights)

            return {
                'mean': mean,
                'cov': cov,
                'new_samples': new_samples,
                'sampler': sampler,
                'weights': weights
            }

    @partial(jax.jit, static_argnums=(0,))
    def ptform(self, u):
        '''
        Turns coordinates on the unit cube into θ_reg object
        '''
        θ_reg = ThetaReg.prior_transform(u[:ThetaReg.dims], bounds=self.bounds)
        norm = jnp.array([self.bounds[-1][0] + u[-1] * (self.bounds[-1][1] - self.bounds[-1][0])])
        return jnp.concatenate((θ_reg, norm))

    @partial(jax.jit, static_argnums=(0,))
    def inv_ptform(self, θ_reg, norm):
        '''
        Turns θ_reg object into coordinates on the unit cube
        '''
        u_reg = ThetaReg.inv_prior_transform(u[:ThetaReg.dims], bounds=self.bounds)
        u_norm = jnp.array([(norm - self.bounds[-1][0]) / (self.bounds[-1][1] - self.bounds[-1][0])])
        return jnp.concatenate((u_reg, u_norm))

    @partial(jax.jit, static_argnums=(0,))
    def model(self, θ):
        return self.l1model.l1model(self.theta_asy, ThetaReg(*θ[:9])) * θ[9] + 1

    @partial(jax.jit, static_argnums=(0,))
    def ln_like(self, θ):
        m = self.model(θ)
        return -jnp.sum(jnp.log(m) + self.s / m) / self.soften

    def get_bounds(self):
        bounds = {'deltaPi0'   : (0.875, 0.925), # 0
          'p_L'        : (1., 3.), # 1
          'p_D'        : (0., 1.), # 2
          'epsilon_g'  : (0., 1.), # 3
          'omega'      : (-1, -0.25), # 4
          'd01'        : (0.15, 0.25), # 5
          'alpha_g'    : (-0.0025, 0.0025),
          'inclination': (0, jnp.pi/2),
          'phi'        : (10, 200)}
        return list(bounds.values())

    def select_n_g(self):
        numax = 10**self.theta_asy.log_numax
        dnu = 10**self.theta_asy.log_dnu
        env_width = 10**self.theta_asy.log_env_width

        n = self.norders // 2 + 1
        width = max((n + 1) * dnu, 3*env_width)
        freq_lims = (numax - width, numax + width)

        dPi0_lims = (self.bounds[0][0], self.bounds[0][1])
        eps_lims = (self.bounds[3][0], self.bounds[3][1])
        return reggae.select_n_g(numax, freq_lims, dPi0_lims, eps_lims)

    @staticmethod
    def _prepare_theta_asy(star):

        if isinstance(star, pbjam.star):

            nu_0 = np.array([row['mean'] for label, row in star.peakbag.summary.iterrows() if 'l0' in label])
            nu_2 = np.array([row['mean'] for label, row in star.peakbag.summary.iterrows() if 'l2' in label])
            d02 = np.median(nu_0 - nu_2)
            dnu = np.median(np.diff(nu_0))

            mean = lambda x: float(star.asy_fit.summary.loc[x]['mean'])
            return nu_0, nu_2, ThetaAsy(
                    log_numax=mean('numax'),
                    log_dnu=np.log10(dnu),
                    eps=mean('eps'),
                    log_d02=np.log10(d02),
                    log_alpha=mean('alpha'),
                    log_hmax=mean('env_height'),
                    log_env_width=mean('env_width'),
                    log_mode_width=mean('mode_width')
                )

        elif isinstance(star, pbjam.modeID.modeIDsampler):
            s = lambda x: float(star.result['summary'][x][0])
            θ = ThetaAsy(
                    log_numax=np.log10(s('numax')),
                    log_dnu=np.log10(s('dnu')),
                    eps=s('eps_p'),
                    log_d02=np.log10(s('d02')),
                    log_alpha=np.log10(s('alpha_p')),
                    log_hmax=np.log10(s('env_height')),
                    log_env_width=np.log10(s('env_width')),
                    log_mode_width=np.log10(s('mode_width'))
                )

            nu_0 = star.AsyFreqModel.asymptotic_nu_p(s('numax'), s('dnu'), s('eps_p'), s('alpha_p'))[0]
            nu_2 = nu_0 - s('d02')
            return nu_0, nu_2, θ

    @staticmethod
    def make_pbjam_model(star, n_samples=50):

        if isinstance(star, pbjam.star): # PBjam star object
            peakbag = star.peakbag
            freq = star.pg.frequency.value

            peakbag.ladder_f = np.array(freq)[None, :]
            n = peakbag.ladder_s.shape[0]
            par_names = ['l0', 'l2', 'width0', 'width2', 'height0', 'height2',
                         'back']

            acc = np.zeros((n_samples, peakbag.ladder_f.shape[1]))
            for i in range(-n_samples, 0):
                z = peakbag.model(*[peakbag.traces[x][i] for x in par_names])
                bg = np.min(z, axis=1)
                acc[i] = np.sum(z - bg[:, None], axis=0) +1

            return np.mean(acc, axis=0)

        elif isinstance(star, pbjam.modeID.modeIDsampler):
            return star.result['background']

    # optimisation tasks

    def simplex(self, θ_reg, norm, **kwargs):

        def fun(θ):
            return -self.ln_like(θ)

        self.simplex_results = minimize(fun, np.concatenate([θ_reg.asarray(), [norm]]), **{'method': 'Nelder-Mead', **kwargs})

    def genetic_algorithm(self, solve_kwargs=None, **kwargs):

        if solve_kwargs is None:
            solve_kwargs = {}

        bounds = [[0, 1]] * (ThetaReg.dims + 1)

        def fun(u):
            return -self.ln_like(self.ptform(u))

        self.de = DE(fun, bounds, **kwargs)
        self.de_results = self.de.solve(show_progress=True, **solve_kwargs)