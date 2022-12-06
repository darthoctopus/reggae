#!/usr/bin/env python3

## Author: Joel Ong <joel.ong@yale.edu>
## Yale University Dept. of Astronomy

'''
2D polynomial fits; some parts adapted from EXPRES pipeline
'''

#import numpy as lib
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as lib

from scipy.optimize import least_squares

#from numpy.polynomial.polynomial import polyval2d

@jax.jit
def polyval2d(x, y, c):
    
    N0, N1 = c.shape
    
    p = lib.ones(len(x))
       
    for k, _x in enumerate(x):
           
        xx = x[k]**lib.arange(N0)

        yy = y[k]**lib.arange(N1)

        p = p.at[k].set(sum(sum(c * lib.multiply(xx[:, None], yy[:, None].T))))

    return p

#def wrap_polyval2d(x, y, p):
#    '''
#    Evaluate 2D polynomial, broadcasting shapes of x and y where appropriate.
#    '''
#    xx = (x + 0 * y)
#    yy = (y + 0 * x)
#    shape = xx.shape
#    o = polyval2d(xx.flatten(), yy.flatten(), poly_params_2d(p))
#    return o.reshape(shape)

@jax.jit
def wrap_polyval2d(x, y, p):
    '''
    Evaluate 2D polynomial, broadcasting shapes of x and y where appropriate.
    '''
    xx = (x + 0 * y)
    yy = (y + 0 * x)
    shape = xx.shape
    o = polyval2d(xx.flatten(), yy.flatten(), poly_params_2d(p))
    return o.reshape(shape)



#def poly_params_2d(p):
#    '''
#    Turn a parameter vector into an upper triangular coefficient matrix.
#    '''
#
#    # is len(p) a triangular number?
#
#    n = (np.sqrt(1 + 8 * len(p)) - 1) / 2
#    assert n == np.floor(n), "length of parameter vector must be a triangular number"
#    n = int(n)
#    C = np.zeros((n, n))
#
#    # populate matrix
#
#    for i in range(n):
#        # ith triangular number as offset
#        n_i = int(i * (i + 1) / 2)
#        for j in range(i + 1):
#            C[i - j, j] = p[n_i + j]
#    return C


@jax.jit
def poly_params_2d(p):
    '''
    Turn a parameter vector into an upper triangular coefficient matrix.
    '''

    # is len(p) a triangular number?
    # n = int((lib.sqrt(1 + 8 * len(p)) - 1) / 2)
    if len(p) == 1:
      n = 1  
    if len(p) == 3:
      n = 2
    if len(p) == 6:
      n = 3
      
    #assert n == lib.floor(n), "length of parameter vector must be a triangular number"
    #n = int(n)
    C = lib.zeros((n, n))

    # populate matrix

    for i in range(n):
        # ith triangular number as offset
        n_i = i * (i + 1) // 2
        
        for j in range(i + 1):
            
            C = C.at[i - j, j].set(p[n_i + j])
            
    return C

def polyfit2d(x, y, z, n_max, e_z=None):
    '''
    Fit a 2D polynomial in x and y against z, under the constraint that
    the coefficient matrix be upper triangular. As in the EXPRES pipeline we
    condition the fit by iteratively increasing the order of the polynomial.
    '''
    p = []
    if e_z is None:
        e_z = lib.ones_like(z)

    cost = lambda p: ((z - wrap_polyval2d(x, y, p)) / e_z).flatten()
    for i in range(n_max + 1):
        p = lib.concatenate((p, [0] * (i+1)))
        j = least_squares(cost, p)
        p = j['x']

    return j
