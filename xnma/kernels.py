#!/usr/bin/env python3

from numba import njit
from numba import stencil
import numpy as np


@stencil
def vr_stencil( psi, dxg ):
    """Stencil for calculating v-velocity from the rotational component"""

    return ( psi[0,1]-psi[0,0] )/dxg[0,0]

@njit(parallel=True,cache=True)
def vr( psi, dxg  ):
    return vr_stencil( psi, dxg )

@stencil
def ur_stencil( psi, dyg ):
    """Stencil for calculating u-velocity from the rotational component"""

    return ( psi[0,0]-psi[1,0] )/dyg[0,0]

@njit(parallel=True,cache=True)
def ur( psi, dyg  ):
    return ur_stencil( psi, dyg )

@stencil
def vorticity_stencil( u, v, dxc, dyc, raz ):
    """Stencil for calculating vorticity on an Arakawa C-grid"""

    return ( v[0,0]*dyc[0,0] - u[0,0]*dxc[0,0] - v[0,-1]*dyc[0,-1] + u[-1,0]*dxc[-1,0] )/raz[0,0]

@njit(parallel=True,cache=True)
def vorticity( u, v, dxc, dyc, raz  ):
    return vorticity_stencil( u, v, dxc, dyc, raz )

@stencil
def divergence_stencil( u, v, dxg, dyg, hfacw, hfacs, rac ):
    """Stencil for calculating divergence on an Arakawa C-grid"""

    return ( u[0,1]*dyg[0,1]*hfacw[0,1] - u[0,0]*dyg[0,0]*hfacw[0,0] + v[1,0]*dxg[1,0]*hfacs[1,0] - v[0,0]*dxg[0,0]*hfacs[0,0] )/rac[0,0]

@njit(parallel=True,cache=True)
def divergence( u, v, dxg, dyg, hfacw, hfacs, rac  ):
    return divergence_stencil( u, v, dxg, dyg, hfacw, hfacs, rac )

@stencil
def LapZ_stencil( s, dxc, dyc, dxg, dyg, raz ):
    """Stencil for the laplacian on vorticity points"""
    
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

    return ( b - LapZ( s, dxc, dyc, dxg, dyg, raz ) )*mask

@stencil
def LapC_stencil( s, dxc, dyc, dxg, dyg, rac ):
    """Stencil for the laplacian on tracer points"""
    
    a = dyg[0,1]/dxc[0,1]
    b = dyg[0,0]/dxc[0,0]
    c = dxg[1,0]/dyc[1,0]
    d = dxg[0,0]/dyc[0,0]

    return ( a*s[0,1] + b*s[0,-1] + c*s[1,0] + d*s[-1,0] - (a+b+c+d)*s[0,0] )/rac[0,0]


@njit(parallel=True,cache=True)
def LapC( s, dxc, dyc, dxg, dyg, rac  ):
    return LapC_stencil( s, dxc, dyc, dxg, dyg, rac )

@njit(parallel=True,cache=True)
def LapC_Residual( s, b, mask, dxc, dyc, dxg, dyg, rac ):
    """Calculates the residual for the laplacian on tracer points"""
    return ( b - LapC( s, dxc, dyc, dxg, dyg, rac ) )*mask

# //////////////// Jacobi Method  //////////////// # 

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
