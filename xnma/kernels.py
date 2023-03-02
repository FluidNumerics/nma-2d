#!/usr/bin/env python3

from numba import njit
from numba import stencil

@stencil
def LapZ_stencil( s, dxc, dyc, dxg, dyg, raz ):
    
    a = dyc[0,0]/dxg[0,0]
    b = dyc[0,-1]/dxg[0,-1]
    c = dxc[0,0]/dyg[0,0]
    d = dxc[-1,0]/dyg[-1,0]

    return ( a*s[0,1] + b*s[0,-1] + c*s[1,0] + d*s[-1,0] - (a+b+c+d)*s[0,0] )/raz[0,0]


@njit(parallel=True,cache=True)
def LapZ( s, dxc, dyc, dxg, dyg, raz  ):
    return LapZ_stencil( s, dxc, dyc, dxg, dyg, raz )

@njit(parallel=True,cache=True)
def LapZ_Residual( s, b, mask, dxc, dyc, dxg, dyg, raz ):
    """Calculates the residual for the laplacian"""

    r = b - LapZ( s, dxc, dyc, dxg, dyg, raz )  
    r = r*mask
    return np.sum(r*r)


# //////////////// Jacobi Method  //////////////// # 

@stencil
def LapZ_JacobiLU_stencil( s, dxc, dyc, dxg, dyg, raz ):
    
    a = dyc[0,0]/dxg[0,0]
    b = dyc[0,-1]/dxg[0,-1]
    c = dxc[0,0]/dyg[0,0]
    d = dxc[-1,0]/dyg[-1,0]

    return ( a*s[0,1] + b*s[0,-1] + c*s[1,0] + d*s[-1,0] )/raz[0,0]

@njit(parallel=True,cache=True)
def LapZ_JacobiLU( s, dxc, dyc, dxg, dyg, raz  ):
    return LapZ_JacobiLU_stencil( s, dxc, dyc, dxg, dyg, raz )

@stencil(neighborhood = ((-1, 1),(-1,1),))
def LapZ_JacobiDinv_stencil( s, dxc, dyc, dxg, dyg, raz ):
    
    a = dyc[0,0]/dxg[0,0]
    b = dyc[0,-1]/dxg[0,-1]
    c = dxc[0,0]/dyg[0,0]
    d = dxc[-1,0]/dyg[-1,0]

    return -raz[0,0]*s[0,0]/(a+b+c+d)

@njit(parallel=True,cache=True)
def LapZ_JacobiDinv( s, dxc, dyc, dxg, dyg, raz ):
    return LapZ_JacobiDinv_stencil( s, dxc, dyc, dxg, dyg, raz )





# //////////////// ///////////// ///////////////// # 
