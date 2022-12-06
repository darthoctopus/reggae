#!/usr/bin/env python3

## Author: Joel Ong <joel.ong@yale.edu>
## Yale University Dept. of Astronomy

'''
2D polynomial fits; some parts adapted from EXPRES pipeline
'''

#import numpy as lib
import jax.numpy as lib

from .coupling import ν_to_ω
from .poly2d import polyfit2d

def asymptotic_ν_g(n_g, ΔΠ0, max_N2, ε_g, α=0, ν_max=0, l=1):
    '''
    Asymptotic relation for the g-mode frequencies
    in terms of a fundamental period offset (defined by the
    maximum Brunt-Vaisala frequency), the asymptotic g-mode period
    spacing, the g-mode phase offset, and an optional curvature term.
    '''
    ΔΠ_l = ΔΠ0 / lib.sqrt(l * (l + 1))
    P0 = 1 / (lib.sqrt(max_N2) / ν_to_ω)
    if ν_max:
        P_max = 1 / ν_max
    else:
        P_max = 0
    n_max = (P_max - P0) / ΔΠ_l - ε_g
    P = P0 + ΔΠ_l * (n_g + ε_g + α * (n_g - n_max)**2)
    return 1/P

def fit_poly(n_p, n_g, ν_p, ν_g, L, D, n_low=None, order=2):
    '''
    Fit upper-triangular matrix coefficients describing 2D polynomials
    of degree n for the scaled coupling and overlap matrices.
    '''

    assert len(n_p) == len(ν_p) and len(n_g) == len(ν_g)
    assert len(n_p) + len(n_g) == len(L)
    assert len(L) == len(D)

    N_π = len(n_p)

    L_scaled = L[:N_π, N_π:] / (ν_g * ν_to_ω)**2
    D_scaled = D[:N_π, N_π:] / (ν_g[None,:]) * (ν_p[:,None])

    if n_low is None:
        n_low = min(int(lib.max(n_g)/10), 10)

    if n_low == 0:
        s = slice(None, None)
    else:
        s = slice(None, -1 * n_low)

    j_L = polyfit2d(n_p[:, None], n_g[None, s], L_scaled[:,s], order)
    j_D = polyfit2d(n_p[:, None], n_g[None, s], D_scaled[:,s], order)

    p_L = j_L['x']
    p_D = j_D['x']

    return {
        'p_L': p_L, 'p_D': p_D,
        'L_scaled': L_scaled,
        'D_scaled': D_scaled
    }
