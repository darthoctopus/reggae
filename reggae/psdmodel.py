import numpy as np
import jax.numpy as jnp
from .theta import ThetaReg

from .reggae import reggae
from .asymptotic import asymptotic

class PSDModel(reggae, asymptotic):

    def __init__(self, f, n_orders, lw=1/200, nu_0=None, nu_2=None, *args, **kwargs):
        self.f = f
        self.n_orders = n_orders
        self.lw = lw
        self.n_g = None
        self.nu_0 = nu_0
        self.nu_2 = nu_2

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

    def get_nu_0(self, theta_asy):
        if self.nu_0 is not None:
            return self.nu_0
        return theta_asy.nu_0(self.n_orders)

    def get_nu_2(self, theta_asy):
        if self.nu_2 is not None:
            return self.nu_2

    def get_d02(self, theta_asy):
        if self.nu_2 is not None and self.nu_0 is not None:
            return self.get_nu_0(None) - self.get_nu_2(None)
        return 10.**(theta_asy.log_d02)

    def getl1(self, theta_asy, theta_reg, **kwargs):

        numax = 10.**(theta_asy.log_numax)
        dnu = 10.**(theta_asy.log_dnu)
        d02 = self.get_d02(theta_asy)
        alpha = 10.**(theta_asy.log_alpha)

        nu_0 = self.get_nu_0(theta_asy)
        nmax = theta_asy.nmax()
        n_p = theta_asy.n_p(self.n_orders)

        return super().getl1(self.n_g,
            nu_0, numax, dnu, d02, alpha, nmax, n_p,
            theta_reg.d01, theta_reg.dPi0, theta_reg.p_L, theta_reg.p_D, theta_reg.epsilon_g,
            theta_reg.alpha_g, **kwargs)

    def _l1model(self, theta_asy, theta_reg, update_n_g=False, amps=None, dnu_p=0, dnu_g=0):

        nu_0 = self.get_nu_0(theta_asy)

        numax = 10.**(theta_asy.log_numax)
        dnu = 10.**(theta_asy.log_dnu)
        env_width = 10.**(theta_asy.log_env_width)
        
        if update_n_g or self.n_g is None:
            self.n_g = self.select_n_g(numax,
                [nu_0[0] - 2*dnu, nu_0[-1] + 2*dnu],
                [theta_reg.dPi0/1.2, theta_reg.dPi0*1.2],
                [0, 1]
                )

        nu_1, zeta = self.getl1(theta_asy, theta_reg, dnu_p=dnu_p, dnu_g=dnu_g)
        H = self._P_envelope(nu_1, theta_reg.normalisation, numax, env_width)
        if amps is not None:
            H = H * amps
        lw = jnp.sqrt((zeta*0+dnu*self.lw)**2 + (10.**(theta_asy.log_mode_width))**2)

        return super().l1model(self.f, nu_1, zeta, dnu, amps=H, lw=lw)

    def l1model(self, theta_asy, theta_reg, **kwargs):
        dnu_g = 10.**theta_reg.log_omega_core /  reggae.nu_to_omega
        inc = theta_reg.inclination
        return (
            self._l1model(theta_asy, theta_reg, dnu_g=0, **kwargs) * jnp.cos(inc)**2
            + self._l1model(theta_asy, theta_reg, dnu_g=-dnu_g, **kwargs) * jnp.sin(inc)**2 / 2
            + self._l1model(theta_asy, theta_reg, dnu_g= dnu_g, **kwargs) * jnp.sin(inc)**2 / 2
        )
