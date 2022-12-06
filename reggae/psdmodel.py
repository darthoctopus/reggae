import numpy as np
import jax.numpy as jnp
from .theta import ThetaReg

from .reggae import reggae
from .asymptotic import asymptotic

class PSDModel(reggae, asymptotic):

    def __init__(self, f, n_orders, lw=1/200, *args, **kwargs):
        self.f = f
        self.n_orders = n_orders
        self.lw = lw
        self.n_g = None

    # def init_psdmodel(self, f, numax, nu_0, d02, env_width, alpha_p=0, lw=1/50):
    #     self.f = f

    #     self.numax = numax
    #     self.dnu = np.median(np.diff(nu_0))
    #     self.nu_0 = nu_0
    #     self.d02 = d02

    #     s = np.median(np.sin(2 * np.pi * nu_0 / self.dnu))
    #     c = np.median(np.cos(2 * np.pi * nu_0 / self.dnu))
    #     self.eps0 = np.arctan2(s, c) / 2 / np.pi

    #     self.n_p = nu_0 / self.dnu - self.eps0
    #     self.nmax = numax / self.dnu - self.eps0
    #     self.env_width = env_width

    #     self.alpha_p = alpha_p
    #     self.lw = lw
    #     self.n_g = None

    def _l1model(self, nu1s, zeta, numax, dnu, env_width, theta_reg, Pmax=0.1, amps=None,
        **kwargs):

        H = self._P_envelope(nu1s, Pmax, numax, env_width)
        if amps is not None:
            H = H * amps

        return super().l1model_rot(self.f, nu1s, zeta, dnu, 10.**theta_reg.log_omega_rot, inc=theta_reg.inclination,
                               amps=H, **kwargs)
    
    def getl1(self, theta_asy, theta_reg):

        numax = 10.**(theta_asy.log_numax)
        dnu = 10.**(theta_asy.log_dnu)
        d02 = 10.**(theta_asy.log_d02)
        alpha = 10.**(theta_asy.log_alpha)

        nu_0 = theta_asy.nu_0(self.n_orders)
        nmax = theta_asy.nmax()
        n_p = theta_asy.n_p(self.n_orders)

        return super().getl1(self.n_g,
            nu_0, numax, dnu, d02, alpha, nmax, n_p,
            theta_reg.d01, theta_reg.dPi0, theta_reg.p_L, theta_reg.p_D, theta_reg.epsilon_g,
            theta_reg.alpha_g)

    def l1model(self, theta_asy, theta_reg, update_n_g=False):

        nu_0 = theta_asy.nu_0(self.n_orders)

        numax = 10.**(theta_asy.log_numax)
        dnu = 10.**(theta_asy.log_dnu)
        env_width = 10.**(theta_asy.log_env_width)
        
        if update_n_g or self.n_g is None:
            self.n_g = self.select_n_g(numax,
                [nu_0[0] - 2*dnu, nu_0[-1] + 2*dnu],
                [theta_reg.dPi0/1.2, theta_reg.dPi0*1.2],
                [0, 1]
                )

        nu_1, zeta = self.getl1(theta_asy, theta_reg)
        lw = jnp.sqrt((zeta*0+dnu*self.lw)**2 + (10.**(theta_asy.log_mode_width))**2)

        return self._l1model(nu_1, zeta, numax, dnu, env_width, theta_reg, lw=lw, Pmax=theta_reg.normalisation)
