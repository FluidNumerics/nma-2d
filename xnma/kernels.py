#!/usr/bin/env python3

from numba import njit
from numba import stencil
import numpy as np

# eShift = 1e-2 # factor to shift the diagonal by for the neumann mode operator


def prolongTracerY(f):
    """Prolongs tracer data in the y-direction to satisfy homogeneous neumann boundary conditions"""
    locf = f

    locf[0, :] = locf[1, :]
    locf[-2, :] = locf[-3, :]
    locf[-1, :] = locf[-2, :]

    return locf


def prolongTracerX(f):
    """Prolongs tracer data in the x-direction to satisfy homogeneous neumann boundary conditions"""
    locf = f

    locf[:, 0] = locf[:, 1]
    locf[:, -2] = locf[:, -3]
    locf[:, -1] = locf[:, -2]

    return locf


def prolongTracer(f):
    """Prolongs tracer data in the x and y directions to satisfy homogeneous neumann boundary conditions"""
    locf = prolongTracerX(f)
    locf = prolongTracerY(locf)
    return locf


@stencil
def TtoU_stencil(f):
    """Stencil for interpolating tracer point data onto U-velocity points"""
    return (f[0, 0] + f[0,-1]) * 0.5


@njit(parallel=True, cache=True)
def TtoU(f):
    return TtoU_stencil(f)


@stencil
def TtoV_stencil(f):
    """Stencil for interpolating tracer point data onto V-velocity points"""
    return (f[0, 0] + f[-1,0]) * 0.5


@njit(parallel=True, cache=True)
def TtoV(f):
    return TtoV_stencil(f)


@stencil
def vorticityToTracer_stencil(f):
    """Stencil for interpolating vorticity data onto tracer points"""
    return 0.25 * (f[0, 0] + f[1, 0] + f[1, 1] + f[0, 1])


@njit(parallel=True, cache=True)
def vorticityToTracer(f):
    return vorticityToTracer_stencil(f)


@stencil
def UtoT_stencil(u):
    """Stencil for interpolating u-velocity data onto tracer points"""
    return 0.5 * (u[0, 1] + u[0, 0])


@njit(parallel=True, cache=True)
def UtoT(u):
    return UtoT_stencil(u)


@stencil
def VtoT_stencil(v):
    """Stencil for interpolating v-velocity data onto tracer points"""
    return 0.5 * (v[1, 0] + v[0, 0])


@njit(parallel=True, cache=True)
def VtoT(v):
    return VtoT_stencil(v)


@stencil
def vr_stencil(psi, dxg):
    """Stencil for calculating v-velocity from the rotational component"""
    return (psi[0, 1] - psi[0, 0]) / dxg[0, 0]


@njit(parallel=True, cache=True)
def vr(psi, dxg):
    return vr_stencil(psi, dxg)


@stencil
def ur_stencil(psi, dyg):
    """Stencil for calculating u-velocity from the rotational component"""
    return (psi[0, 0] - psi[1, 0]) / dyg[0, 0]


@njit(parallel=True, cache=True)
def ur(psi, dyg):
    return ur_stencil(psi, dyg)


@stencil
def vorticity_stencil(u, v, dxc, dyc, raz):
    """Stencil for calculating vorticity on an Arakawa C-grid"""
    return (
        v[0, 0] * dyc[0, 0]
        - u[0, 0] * dxc[0, 0]
        - v[0, -1] * dyc[0, -1]
        + u[-1, 0] * dxc[-1, 0]
    ) / raz[0, 0]


@njit(parallel=True, cache=True)
def vorticity(u, v, dxc, dyc, raz):
    return vorticity_stencil(u, v, dxc, dyc, raz)


@stencil
def vd_stencil(phi, dyc):
    """Stencil for calculating v-velocity from the divergent component"""
    return (phi[1, 0] - phi[0, 0]) / dyc[0, 0]


@njit(parallel=True, cache=True)
def vd(phi, dyc):
    return vd_stencil(phi, dyc)


@stencil
def ud_stencil(phi, dxc):
    """Stencil for calculating u-velocity from the divergent component"""
    return (phi[0, 1] - phi[0, 0]) / dxc[0, 0]


@njit(parallel=True, cache=True)
def ud(phi, dxc):
    return ud_stencil(phi, dxc)


@stencil
def divergence_stencil(u, v, dxg, dyg, hfacw, hfacs, rac):
    """Stencil for calculating divergence on an Arakawa C-grid"""
    return (
        u[0, 1] * dyg[0, 1] * hfacw[0, 1]
        - u[0, 0] * dyg[0, 0] * hfacw[0, 0]
        + v[1, 0] * dxg[1, 0] * hfacs[1, 0]
        - v[0, 0] * dxg[0, 0] * hfacs[0, 0]
    ) / rac[0, 0]


@njit(parallel=True, cache=True)
def divergence(u, v, dxg, dyg, hfacw, hfacs, rac):
    return divergence_stencil(u, v, dxg, dyg, hfacw, hfacs, rac)


@stencil
def LapZ_stencil(s, dxc, dyc, dxg, dyg, raz, eShift):
    """Stencil for the laplacian on vorticity points"""

    a = dyc[0, 0] / dxg[0, 0]
    b = dyc[0, -1] / dxg[0, -1]
    c = dxc[0, 0] / dyg[0, 0]
    d = dxc[-1, 0] / dyg[-1, 0]

    return (
        a * s[0, 1]
        + b * s[0, -1]
        + c * s[1, 0]
        + d * s[-1, 0]
        - (a + b + c + d) * s[0, 0]
    ) / raz[0, 0]  - eShift * s[0, 0]


@njit(parallel=True, cache=True)
def LapZ(s, dxc, dyc, dxg, dyg, raz, eShift):
    return LapZ_stencil(s, dxc, dyc, dxg, dyg, raz, eShift)


@njit(parallel=True, cache=True)
def LapZ_Residual(s, b, mask, dxc, dyc, dxg, dyg, raz, eShift=0):
    """Calculates the residual for the laplacian"""

    return (b - LapZ(s, dxc, dyc, dxg, dyg, raz, eShift)) * mask


@stencil
def LapC_stencil(s, dxc, dyc, dxg, dyg, maskw, masks, rac, eShift=0):
    """Stencil for the laplacian on tracer points"""

    a = maskw[0, 1] * dyg[0, 1] / dxc[0, 1]
    b = maskw[0, 0] * dyg[0, 0] / dxc[0, 0]
    c = masks[1, 0] * dxg[1, 0] / dyc[1, 0]
    d = masks[0, 0] * dxg[0, 0] / dyc[0, 0]

    return (
        a * s[0, 1]
        + b * s[0, -1]
        + c * s[1, 0]
        + d * s[-1, 0]
        - (a + b + c + d) * s[0, 0]
    ) / rac[0, 0] - eShift * s[0, 0]


@njit(parallel=True, cache=True)
def LapC(s, dxc, dyc, dxg, dyg, maskw, masks, rac, eShift=0):
    return LapC_stencil(s, dxc, dyc, dxg, dyg, maskw, masks, rac, eShift)


@njit(parallel=True, cache=True)
def LapC_Residual(s, b, mask, dxc, dyc, dxg, dyg, maskw, masks, rac, eShift=0):
    """Calculates the residual for the laplacian on tracer points"""
    return (b - LapC(s, dxc, dyc, dxg, dyg, maskw, masks, rac, eShift)) * mask


# //////////////// Jacobi Method  //////////////// #


@stencil(
    neighborhood=(
        (-1, 1),
        (-1, 1),
    )
)
def LapZ_JacobiDinv_stencil(s, dxc, dyc, dxg, dyg, raz, shift=0.0):
    a = dyc[0, 0] / dxg[0, 0]
    b = dyc[0, -1] / dxg[0, -1]
    c = dxc[0, 0] / dyg[0, 0]
    d = dxc[-1, 0] / dyg[-1, 0]

    return -raz[0, 0] * s[0, 0] / ((a + b + c + d) + shift*raz[0,0])


@njit(parallel=True, cache=True)
def LapZ_JacobiDinv(s, dxc, dyc, dxg, dyg, raz, shift=0.0):
    return LapZ_JacobiDinv_stencil(s, dxc, dyc, dxg, dyg, raz, shift)


@stencil(
    neighborhood=(
        (-1, 1),
        (-1, 1),
    )
)
def LapC_JacobiDinv_stencil(s, dxc, dyc, dxg, dyg, maskw, masks, rac, eShift=0):
    a = maskw[0, 1] * dyg[0, 1] / dxc[0, 1]
    b = maskw[0, 0] * dyg[0, 0] / dxc[0, 0]
    c = masks[1, 0] * dxg[1, 0] / dyc[1, 0]
    d = masks[0, 0] * dxg[0, 0] / dyc[0, 0]

    return -rac[0, 0] / ((a + b + c + d) + eShift * rac[0, 0]) * s[0, 0]


@njit(parallel=True, cache=True)
def LapC_JacobiDinv(s, dxc, dyc, dxg, dyg, maskw, masks, rac, eShift=0):
    return LapC_JacobiDinv_stencil(s, dxc, dyc, dxg, dyg, maskw, masks, rac, eShift)


# //////////////// ///////////// ///////////////// #
