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
