#!/usr/bin/env python3

## Author: Joel Ong <joel.ong@yale.edu>
## Yale University Dept. of Astronomy

'''
2D polynomial fits; some parts adapted from EXPRES pipeline
'''

import jax.numpy as jnp
from .coupling import ν_to_ω
from .poly2d import polyfit2d

def asymptotic_ν_g(n_g, ΔΠ0, max_N2, ε_g, α=0, ν_max=0, l=1):
    """Compute asymptotic g-mode frequencies

    Uses the asymptotic g-mode relation to compute the frequencies of a 
    given set of radial orders n_g and angular degree l, using a fundamental 
    period offset (defined by the maximum Brunt-Vaisala frequency) given
    period spacing, phase offset and curvature.

    Parameters
    ----------
    n_g: array-like
        Array of radial orders at which the g-mode frequencies are computed
    ΔΠ0: float
        The period spacing of the g-modes
    max_N2: float
        The maximum of the Brunt-Vaisala frequency, used to define the fundamental
        period offset.
    ε_g: float 
        Phase offset for the g-modes.
    α: float, optional
        Curvature factor for the g-mode periods. Default is 0.
    ν_max: float, optional 
        numax of the envelope. Default is 0
    l=1: int
        Angular degree of the g-modes. Default is 1.

    Returns
    -------
    nu : array-like
        An array of g-mode frequencies. 
    """

    ΔΠ_l = ΔΠ0 / jnp.sqrt(l * (l + 1))

    P0 = 1 / (jnp.sqrt(max_N2) / ν_to_ω)
    
    if ν_max:
        P_max = 1 / ν_max
    
    else:
        P_max = 0
    
    n_max = (P_max - P0) / ΔΠ_l - ε_g
    
    P = P0 + ΔΠ_l * (n_g + ε_g + α * (n_g - n_max)**2)
    
    return 1/P

def fit_poly(n_p, n_g, ν_p, ν_g, L, D, n_low=None, order=2):
    """
    Fit upper-triangular matrix coefficients describing 2D polynomials
    of degree n for the scaled coupling and overlap matrices.

    Parameters
    ----------
    n_p: array-like
        Array of radial orders for the p-modes.
    n_g: array-like
        Array of radial order for the g-modes.
    nu_p: array-like
        Array of frequencies for the p-modes
    nu_g: array-like
        Array of frequencies for the g-modes
    L: array-like
        2D array of coupling terms for each combination of p and g modes.
    D: array-like
        2D array of coupling terms for each combination of p and g modes.
    n_low: int, optional
        Lowest g-mode radial order. Default is None.
    order: int, optional
        Polynomial order. Default is 2.
    
    Returns
    -------
    res: dict
        Dictionary of coupling polynomial coefficients p_L and p_D and 
        scaled coupling matrices.
    """

    assert len(n_p) == len(ν_p) and len(n_g) == len(ν_g)
    
    assert len(n_p) + len(n_g) == len(L)
    
    assert len(L) == len(D)

    N_π = len(n_p)

    L_scaled = L[:N_π, N_π:] / (ν_g * ν_to_ω)**2
    D_scaled = D[:N_π, N_π:] / (ν_g[None,:]) * (ν_p[:,None])

    if n_low is None:
        n_low = min(int(jnp.max(n_g)/10), 10)

    if n_low == 0:
        s = slice(None, None)
    else:
        s = slice(None, -1 * n_low)

    j_L = polyfit2d(n_p[:, None], n_g[None, s], L_scaled[:,s], order)
    j_D = polyfit2d(n_p[:, None], n_g[None, s], D_scaled[:,s], order)

    p_L = j_L['x']
    p_D = j_D['x']

    return {'p_L': p_L, 'p_D': p_D,
            'L_scaled': L_scaled,
            'D_scaled': D_scaled
            }
