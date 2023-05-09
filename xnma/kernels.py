#!/usr/bin/env python3

from numba import njit
from numba import stencil
import numpy as np

#eShift = 1e-2 # factor to shift the diagonal by for the neumann mode operator

@stencil
def vorticityToTracer_stencil( f ):
    """Stencil for interpolating vorticity data onto tracer points"""
    return 0.25*( f[0,0]+f[1,0]+f[1,1]+f[0,1] )

@njit(parallel=True,cache=True)
def vorticityToTracer( f  ):
    return vorticityToTracer_stencil( f )


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
def vd_stencil( phi, dyc ):
    """Stencil for calculating v-velocity from the divergent component"""
    return ( phi[1,0]-phi[0,0] )/dyc[0,0]

@njit(parallel=True,cache=True)
def vd( phi, dyc  ):
    return vd_stencil( phi, dyc )

@stencil
def ud_stencil( phi, dxc ):
    """Stencil for calculating u-velocity from the divergent component"""
    return ( phi[0,1]-phi[0,0] )/dxc[0,0]

@njit(parallel=True,cache=True)
def ud( phi, dxc  ):
    return ud_stencil( phi, dxc )

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
def LapC_stencil( s, dxc, dyc, dxg, dyg, hfacw, hfacs, rac, eShift=0 ):
    """Stencil for the laplacian on tracer points"""
    
    a = hfacw[0,1]*dyg[0,1]/dxc[0,1]
    b = hfacw[0,0]*dyg[0,0]/dxc[0,0]
    c = hfacs[1,0]*dxg[1,0]/dyc[1,0]
    d = hfacs[0,0]*dxg[0,0]/dyc[0,0]

    return ( a*s[0,1] + b*s[0,-1] + c*s[1,0] + d*s[-1,0] - (a+b+c+d)*s[0,0] )/rac[0,0] - eShift*s[0,0]


@njit(parallel=True,cache=True)
def LapC( s, dxc, dyc, dxg, dyg, hfacw, hfacs, rac, eShift=0  ):
    return LapC_stencil( s, dxc, dyc, dxg, dyg, hfacw, hfacs, rac, eShift )

@njit(parallel=True,cache=True)
def LapC_Residual( s, b, mask, dxc, dyc, dxg, dyg, hfacw, hfacs, rac, eShift=0 ):
    """Calculates the residual for the laplacian on tracer points"""
    return ( b - LapC( s, dxc, dyc, dxg, dyg, hfacw, hfacs, rac, eShift ) )*mask

# //////////////// Jacobi Method  //////////////// # 

@stencil(neighborhood = ((-1,1),(-1,1),))
def LapZ_JacobiDinv_stencil( s, dxc, dyc, dxg, dyg, raz ):
    
    a = dyc[0,0]/dxg[0,0]
    b = dyc[0,-1]/dxg[0,-1]
    c = dxc[0,0]/dyg[0,0]
    d = dxc[-1,0]/dyg[-1,0]

    return -raz[0,0]*s[0,0]/(a+b+c+d)

@njit(parallel=True,cache=True)
def LapZ_JacobiDinv( s, dxc, dyc, dxg, dyg, raz ):
    return LapZ_JacobiDinv_stencil( s, dxc, dyc, dxg, dyg, raz )

@stencil(neighborhood = ((-1,1),(-1,1),))
def LapC_JacobiDinv_stencil( s, dxc, dyc, dxg, dyg, hfacw, hfacs, rac, eShift=0 ):
    
    a = hfacw[0,1]*dyg[0,1]/dxc[0,1]
    b = hfacw[0,0]*dyg[0,0]/dxc[0,0]
    c = hfacs[1,0]*dxg[1,0]/dyc[1,0]
    d = hfacs[0,0]*dxg[0,0]/dyc[0,0]

    return -rac[0,0]/((a+b+c+d)+eShift*rac[0,0])*s[0,0]

@njit(parallel=True,cache=True)
def LapC_JacobiDinv( s, dxc, dyc, dxg, dyg, hfacw, hfacs, rac, eShift=0 ):
    return LapC_JacobiDinv_stencil( s, dxc, dyc, dxg, dyg, hfacw, hfacs, rac, eShift )





# //////////////// ///////////// ///////////////// # 
