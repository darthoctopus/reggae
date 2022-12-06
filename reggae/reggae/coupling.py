#!/usr/bin/env python3

## Author: Joel Ong <joel.ong@yale.edu>
## Yale University Dept. of Astronomy

'''
Manipulation of the coupling matrices. Directly lifted from the utils.coupling
module of Joel's gitlab repository.
'''

import numpy as np
import scipy.linalg as la
from .poly2d import wrap_polyval2d

ν_to_ω = 2 * np.pi / 1e6

def zip_n(n_p, n_g, ν_p, ν_g):
    '''
    Produce a set of n_p and n_g so that the usual Eckert-Scuflaire-Osaki quantity
    n_p - n_g behaves as expected. This however does NOT reproduce the correct n_p or n_g
    separately! In order to do this we have to make use of the mixing coefficients
    ζ somehow.
    '''

    q = np.argsort(ν_g)
    ν_g = ν_g[q]
    n_g = n_g[q]

    q = np.argsort(ν_p)
    ν_p = ν_p[q]
    n_p = n_p[q]

    ν = np.concatenate((ν_p, ν_g))
    q = np.argsort(ν)
    m = np.concatenate((np.ones_like(ν_p), np.zeros_like(ν_g))).astype(bool)

    out_n_p = np.ones_like(ν) * np.nan
    out_n_g = np.ones_like(ν) * np.nan

    out_n_p[m[q]] = n_p
    out_n_g[~m[q]] = n_g

    # Everything after this line is fake

    if np.isnan(out_n_p[0]):
        out_n_p[0] = np.min(n_p) - 1
    if np.isnan(out_n_g[0]):
        out_n_g[0] = np.max(n_g) + 1

    for i in range(len(out_n_p)):
        if np.isnan(out_n_p[i]):
            out_n_p[i] = out_n_p[i-1]
        if np.isnan(out_n_g[i]):
            out_n_g[i] = out_n_g[i-1]

    return out_n_p.astype(int), out_n_g.astype(int)

def new_modes(A, D, N_π, *, l=1, ξr=None, ξh=None, M=None):
    r'''
    Given the matrices A and D such that we have eigenvectors

    A cᵢ = -ωᵢ² D cᵢ,

    with ω in Hz, we solve for the frequencies ν (μHz), mode mixing coefficient ζ, and
    (if the vector coefficients ξ_r, ξ_h at a specified radius R are provided) the
    inertiae E of the modes of the coupled system. The inertiae are evaluated assuming unit
    normalisation of the eigenfunctions, so that

    Eᵢ = 4π \int dm |ξᵢ(m)|² / (M \int dΩ |ξᵢ(R)|²) === 4π / (M \int dΩ |ξᵢ(R)|²).

    Note that this differs by 4π from some other definitions; this is for consistency
    with GYRE. Overall constant factors shouldn't really matter in any case.
    '''

    Λ, U = la.eigh(A, D)

    Dγ = np.identity(len(D))
    Dγ[:N_π, :N_π] = 0

    new_ω2 = -Λ
    ζ = np.diag(U.T @ Dγ @ U)

    E = None
    if ξr is not None and ξh is not None and M is not None:
        Λ2 = l*l + l
        ηr = ξr.T @ U
        ηh = ξh.T @ U
        E = 1 / (ηr**2 + Λ2 * ηh**2) / M

    m = np.argsort(new_ω2)
    return np.sqrt(new_ω2)[m] / ν_to_ω, ζ[m], E[m] if E is not None else E

def generate_matrices(n_p, n_g, ν_p, ν_g, p_L, p_D):
    '''
    Generate a coupling matrix given a set of p- and g-mode frequencies,
    their radial orders n_p and n_g, and some polynomial coefficients for the
    scaled dimensionless coupling strengths and overlap integrals.

    Inputs:
    n_p: ndarray containing π-mode radial orders, of length N_π
    n_g: ndarray containing γ-mode radial orders, of length N_γ
    ν_p: ndarray containing p-mode frequencies, of length N_π
    ν_g: ndarray containing g-mode frequencies, of length N_γ
    p_L: parameter vector describing 2D polynomial coefficients for coupling strengths
    p_D: parameter vector describing 2D polynomial coefficients for overlap integrals
    '''

    assert len(n_p) == len(ν_p) and len(n_g) == len(ν_g)
    N_π = len(n_p)
    N_γ = len(n_g)

    L = np.zeros((N_π + N_γ, N_π + N_γ))
    D = np.identity(N_π + N_γ)

    # set diagonal elements of wave operator matrix

    L[:N_π, :N_π] = np.diag( -(ν_p * ν_to_ω)**2 )
    L[N_π:, N_π:] = np.diag( -(ν_g * ν_to_ω)**2 )

    # evaluate cross terms and reintroduce appropriate scaling

    L_cross = wrap_polyval2d(n_p[:, None], n_g[None, :], p_L) * (ν_g * ν_to_ω)**2
    D_cross = wrap_polyval2d(n_p[:, None], n_g[None, :], p_D) * (ν_g[None,:]) / (ν_p[:,None])

    L[:N_π, N_π:] = L_cross
    L[N_π:, :N_π] = L_cross.T

    D[:N_π, N_π:] = D_cross
    D[N_π:, :N_π] = D_cross.T

    return L, D
