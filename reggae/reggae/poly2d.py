#!/usr/bin/env python3

## Author: Joel Ong <joel.ong@yale.edu>
## Yale University Dept. of Astronomy

"""
2D polynomial fits; some parts adapted from EXPRES pipeline
"""

import numpy as np
from numpy.polynomial.polynomial import polyval2d
from scipy.optimize import least_squares

def wrap_polyval2d(x, y, p):
    """
    Evaluate 2D polynomial, broadcasting shapes of x and y where appropriate.

    Parameters
    ----------
    x: array-like
        x-coordinate for 2D polynomial.
    y: array-like
        y-coordinate for 2D polynomial.
    p: array-like
        Polynomial coefficients.
        
    Returns
    -------
    o: array-like
        Result of evaluating the 2D polynomial.
    """
    
    xx = (x + 0 * y)

    yy = (y + 0 * x)
    
    shape = xx.shape
    
    o = polyval2d(xx.flatten(), yy.flatten(), poly_params_2d(p))
    
    return o.reshape(shape)

def poly_params_2d(p):
    """
    Turn a parameter vector into an upper triangular coefficient matrix.

    Parameters
    ----------
    p: array-like
        1D array of polynomial coefficients to be broadcast to upper triangular matrix.

    Returns
    -------
    C: array-like
        2D array consisting of a upper triangular matrix with entries corresponding to p.

    """

    # is len(p) a triangular number?
    n = (np.sqrt(1 + 8 * len(p)) - 1) / 2

    assert n == np.floor(n), "length of parameter vector must be a triangular number"
    
    n = int(n)
    
    C = np.zeros((n, n))

    # populate matrix

    for i in range(n):
        # ith triangular number as offset
        n_i = int(i * (i + 1) / 2)
    
        for j in range(i + 1):
            C[i - j, j] = p[n_i + j]
    
    return C

def polyfit2d(x, y, z, n_max, e_z=None):
    """Fit a 2D polynomial in x and y against z
    
    We apply the constraint that the coefficient matrix be upper triangular. 
    As in the EXPRES pipeline we condition the fit by iteratively increasing 
    the order of the polynomial.

    Parameters
    ----------
    x: array-like
        x-coordinate for 2D polynomial.
    y: array-like
        y-coordinate for 2D polynomial.
    z: array-like
        Values to fit to.
    n_max: int
        Maximum number of radial orders.
    e_z: array-like
        Array to errors on z. 
    Returns
    -------
    """

    p = []
    
    if e_z is None:
        e_z = np.ones_like(z)

    cost = lambda p: ((z - wrap_polyval2d(x, y, p)) / e_z).flatten()

    for i in range(n_max + 1):
        p = np.concatenate((p, [0] * (i+1)))

        j = least_squares(cost, p)
        
        p = j['x']

    return j
