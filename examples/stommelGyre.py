#!/usr/bin/env python

from xnma import nma
from xnma import kernels

import inspect, os.path
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import cg

# import matplotlib
# matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
# matplotliba.rcParams.update({'font.family': 'serif', 'font.size': 18,
#    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
# matplotlib.rcParams['text.usetex'] = True


import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import sys
import time
import argparse

plt.style.use('dark_background')
plt.switch_backend('agg')

parser = argparse.ArgumentParser(
                    prog='constantVelocity',
                    description='Calculates spectra of constant vector field')

parser.add_argument('-x', '--nx',default=13, type=int)
parser.add_argument('-y', '--ny',default=13, type=int)
parser.add_argument('-m', '--nmodes',default=80, type=int)
parser.add_argument('-t', '--tolerance',default=1e-4, type=float)
parser.add_argument('-p', '--precision',default="float32", type=ascii)
parser.add_argument('-s', '--shift',default=0.0, type=float)

args = parser.parse_args()

nx = args.nx
ny = args.ny
shift = args.shift
n_numerical_modes = args.nmodes
tolerance = args.tolerance
if args.precision == "float32":
    prec = np.float32
elif args.precision == "float64":
    prec = np.float64
else:
    prec = np.float32
print(f"{args.nx}, {args.ny}, {args.nmodes}")


doplots = False
Lx = 1.0
Ly = 1.0

# Calculate the grid spacing
# so that the western most "wet"
# xg point is at 0 and the eastern
# most xg point is at 1
dx = Lx / (nx - 3)
dy = Ly / (ny - 3)

def DirichletModes( model ):
    """ Calculates the exact dirichlet modes for a rectangular domain"""

    nmx = int((nx-4))
    nmy = int((ny-4))
    nmodes = nmx*nmy
    eigenmodes = np.zeros((nmodes,ny,nx))
    eigenvalues = np.zeros(nmodes)
    # Get the wave-numbers
    kx = np.zeros(nmx)
    for m in range(0, nmx):
      kx[m] = (m+1)*np.pi/Lx
        
    ky = np.zeros(nmy)
    for m in range(0, nmy):
      ky[m] = (m+1)*np.pi/Ly

    k = 0
    tmp = np.zeros((nmodes,ny,nx))
    ev = np.zeros(nmodes)
    for m in range(0, nmy):
        for n in range(0, nmx):
            for j in range(0, ny):
                y = model.yg[j]*ky[m]
                for i in range(0, nx):
                    x = model.xg[i]*kx[n]
                    tmp[k,j,i] = np.sin( x )*np.sin( y )
            ev[k] = kx[n]**2 + ky[m]**2
            k+=1

    sort_index = np.argsort(ev)
    eigenvalues = ev[sort_index]
    for k in range(0,nmodes):
        sgrid = ma.masked_array( tmp[sort_index[k],:,:].squeeze(),
                                 mask=abs(model.maskZ - 1.0), 
                                 dtype=np.float32 )
        g = sgrid.data * model.maskZ
        # Normalize so that the norm of the eigenmode is 1
        mag = np.sqrt(np.sum(g * g * model.raz))
        eigenmodes[k,:,:] = g/mag

    return eigenvalues, eigenmodes

def NeumannModes( model ):
    """ Calculates the exact neumann modes for a rectangular domain"""

    nmx = int((nx-3))
    nmy = int((ny-3))
    nmodes = nmx*nmy
    eigenmodes = np.zeros((nmodes,ny,nx))
    eigenvalues = np.zeros(nmodes)
    # Get the wave-numbers
    kx = np.zeros(nmx)
    for m in range(1, nmx):
      kx[m] = m*np.pi/Lx
        
    ky = np.zeros(nmy)
    for m in range(1, nmy):
      ky[m] = m*np.pi/Ly

    k = 0
    tmp = np.zeros((nmodes,ny,nx))
    ev = np.zeros(nmodes)
    for m in range(0, nmy):
        for n in range(0, nmx):
            for j in range(0, ny):
                y = model.yc[j]*ky[m]
                for i in range(0, nx):
                    x = model.xc[i]*kx[n]
                    tmp[k,j,i] = np.cos( x )*np.cos( y )
            ev[k] = kx[n]**2 + ky[m]**2
            k+=1

    sort_index = np.argsort(ev)
    eigenvalues = ev[sort_index]

    for k in range(0,nmodes):

        sgrid = ma.masked_array( tmp[sort_index[k],:,:].squeeze(), 
                                mask=abs(model.maskC - 1.0), 
                                dtype=np.float32 )
        g = sgrid.data * model.maskC
        # Normalize so that the norm of the eigenmode is 1
        mag = np.sqrt(np.sum(g * g * model.rac))
        eigenmodes[k,:,:] = g/mag

    return eigenvalues, eigenmodes



def main():
    # Initialize the nma model
    model = nma.model()

    # Construct the model
    model.construct(dx, dy, nx, nx, prec)

    # Find the eigenmodes (all on tracer points)
    tic = time.perf_counter()
    model.findEigenmodes(nmodes=n_numerical_modes, tolerance=tolerance)
    toc = time.perf_counter()
    runtime = toc - tic

    # Get the exact modes
    n_eigenvalues, n_eigenmodes = NeumannModes(model)
    d_eigenvalues, d_eigenmodes = DirichletModes(model)

    u = np.zeros((ny, nx), dtype=prec)
    v = np.zeros((ny, nx), dtype=prec)
    psi = np.zeros((ny, nx), dtype=prec)

    # Fill in example u,v
    for j in range(0, model.yg.shape[0]):
        yg = model.yg[j]
        for i in range(0, model.xg.shape[0]):
            xg = model.xg[i]
            psi[j, i] = -pi * np.sin(pi * yg) * (1.0 - xg) * (np.exp(-xg / L) - 1.0)

    for j in range(0, model.yg.shape[0] - 1):
        for i in range(0, model.xg.shape[0]):
            u[j, i] = -(psi[j + 1, i] - psi[j, i]) / dy

    for j in range(0, model.yg.shape[0]):
        for i in range(0, model.xg.shape[0] - 1):
            v[j, i] = (psi[j, i + 1] - psi[j, i]) / dx

    # Calculate total energy
    uc = kernels.UtoT(u)
    vc = kernels.VtoT(v)
    Etot = np.sum(0.5 * (uc * uc + vc * vc) * model.rac * model.maskC)

    # Find the projection coefficients (using the model)
    (
        di_m,
        db_m,
        vi_m,
        vb_m
    ) = model.vectorProjection(u, v)
    
    # Calculate the energy associated with interior vorticity
    interior_divE = -0.5 * di_m * di_m / model.n_eigenvalues
    interior_divE[model.n_eigenvalues == 0.0] = 0.0

     # Calculate the energy associated with boundary vorticity
    boundary_divE = -0.5 * db_m * db_m / model.n_eigenvalues
    boundary_divE[model.n_eigenvalues == 0.0] = 0.0

    # Calculate the energy associated with interior vorticity
    interior_rotE = -0.5 * vi_m * vi_m / model.d_eigenvalues

     # Calculate the energy associated with boundary vorticity
    boundary_rotE = -0.5 * vb_m * vb_m / model.d_eigenvalues

    # Collapse degenerate modes
    uniq_n_evals = np.unique(model.n_eigenvalues)
    interior_divE_uniq = np.zeros_like(uniq_n_evals)
    boundary_divE_uniq = np.zeros_like(uniq_n_evals)
    k = 0
    for ev in uniq_n_evals:
        interior_divE_uniq[k] = np.sum(interior_divE[model.n_eigenvalues == ev])
        boundary_divE_uniq[k] = np.sum(boundary_divE[model.n_eigenvalues == ev])
        k+=1
        
    uniq_d_evals = np.unique(model.d_eigenvalues)
    interior_rotE_uniq = np.zeros_like(uniq_d_evals)
    boundary_rotE_uniq = np.zeros_like(uniq_d_evals)
    k = 0
    for ev in uniq_d_evals:
        interior_rotE_uniq[k] = np.sum(interior_rotE[model.d_eigenvalues == ev])
        boundary_rotE_uniq[k] = np.sum(boundary_rotE[model.d_eigenvalues == ev])
        k+=1

    # Find the projection (using the exact modes)
    # To do this, we swap the model's modes for the exact modes
    # Initialize the nma model
    exmodel = nma.model()

    # Construct the model
    exmodel.construct(dx, dy, nx, nx, prec)
    n = n_numerical_modes-1
    exmodel.d_eigenmodes = d_eigenmodes[0:n,:,:]
    exmodel.d_eigenvalues = -d_eigenvalues[0:n]
    exmodel.n_eigenmodes = n_eigenmodes[0:n,:,:]
    exmodel.n_eigenvalues = -n_eigenvalues[0:n]
    (
        ex_di_m,
        ex_db_m,
        ex_vi_m,
        ex_vb_m
    ) = exmodel.vectorProjection(u, v)

    # Calculate the energy associated with interior vorticity
    ex_interior_divE = -0.5 * ex_di_m * ex_di_m / exmodel.n_eigenvalues
    ex_interior_divE[exmodel.n_eigenvalues == 0.0] = 0.0
    
     # Calculate the energy associated with boundary vorticity
    ex_boundary_divE = -0.5 * ex_db_m * ex_db_m / exmodel.n_eigenvalues
    ex_boundary_divE[exmodel.n_eigenvalues == 0.0] = 0.0

    # Calculate the energy associated with interior vorticity
    ex_interior_rotE = -0.5 * ex_vi_m * ex_vi_m / exmodel.d_eigenvalues

     # Calculate the energy associated with boundary vorticity
    ex_boundary_rotE = -0.5 * ex_vb_m * ex_vb_m / exmodel.d_eigenvalues

    # Collapse degenerate eigenmodes
    uniq_nex_evals = np.unique(exmodel.n_eigenvalues)
    ex_interior_divE_uniq = np.zeros_like(uniq_nex_evals)
    ex_boundary_divE_uniq = np.zeros_like(uniq_nex_evals)
    k = 0
    for ev in uniq_nex_evals:
        ex_interior_divE_uniq[k] = np.sum(ex_interior_divE[exmodel.n_eigenvalues == ev])
        ex_boundary_divE_uniq[k] = np.sum(ex_boundary_divE[exmodel.n_eigenvalues == ev])
        k+=1
    
    uniq_dex_evals = np.unique(exmodel.d_eigenvalues)
    ex_interior_rotE_uniq = np.zeros_like(uniq_dex_evals)
    ex_boundary_rotE_uniq = np.zeros_like(uniq_dex_evals)
    k = 0
    for ev in uniq_dex_evals:
        ex_interior_rotE_uniq[k] = np.sum(ex_interior_rotE[exmodel.d_eigenvalues == ev])
        ex_boundary_rotE_uniq[k] = np.sum(ex_boundary_rotE[exmodel.d_eigenvalues == ev])
        k+=1


    plt.figure()
    plt.plot( np.abs(uniq_nex_evals[1:]), ex_boundary_divE_uniq[1:],'-o', label='Divergent (exact)', markersize=3, linewidth=1)
    plt.plot( np.abs(uniq_dex_evals), ex_boundary_rotE_uniq,'-o', label='Rotational (exact)', markersize=3, linewidth=1 )
    plt.plot( np.abs(uniq_n_evals[1:]), boundary_divE_uniq[1:],'-x', label='Divergent (numerical)', markersize=4, linewidth=1 )
    plt.plot( np.abs(uniq_d_evals), boundary_rotE_uniq,'-x', label='Rotational (numerical)', markersize=4, linewidth=1 )
    plt.title("spectra (boundary contribution)")
    plt.xlabel("$\lambda$")
    plt.ylabel("energy")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(f"stommelgyre-boundary-energy_{nx-3}.png")        


    plt.figure()
    plt.plot( np.abs(uniq_nex_evals[1:]), ex_interior_divE_uniq[1:],'-o', label='Divergent (exact)', markersize=3, linewidth=1)
    plt.plot( np.abs(uniq_dex_evals), ex_interior_rotE_uniq,'-o', label='Rotational (exact)', markersize=3, linewidth=1 )
    plt.plot( np.abs(uniq_n_evals[1:]), interior_divE_uniq[1:],'-x', label='Divergent (numerical)', markersize=4, linewidth=1 )
    plt.plot( np.abs(uniq_d_evals), interior_rotE_uniq,'-x', label='Rotational (numerical)', markersize=4, linewidth=1 )
    plt.title("spectra (interior)")
    plt.xlabel("$\lambda$")
    plt.ylabel("energy")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(f"stommelgyre-interior-energy_{nx-3}.png")

    csvfile = './stommel-gyre-energy.csv'
    with open(csvfile, 'a') as f:
        if( os.path.getsize(csvfile) == 0 ):
            f.write('nx, ny, nmodes, "interior divergent energy (exact)", "interior divergent energy (numerical)", "boundary divergent energy (exact)", "boundary divergent energy (numerical)", "interior rotational energy (exact)", "interior rotational energy (numerical)", "boundary rotational energy (exact)", "boundary rotational energy (numerical)", "eigenmode search runtime (s)" \n')
       
        ex_int_divE = np.sum(ex_interior_divE)
        ex_int_rotE = np.sum(ex_interior_rotE)
        int_divE = np.sum(interior_divE)
        int_rotE = np.sum(interior_rotE)
        ex_bound_divE = np.sum(ex_boundary_divE)
        ex_bound_rotE = np.sum(ex_boundary_rotE)
        bound_divE = np.sum(boundary_divE)
        bound_rotE = np.sum(boundary_rotE)
        f.write(f'{nx-3}, {ny-3}, {n_numerical_modes}, {ex_int_divE}, {int_divE}, {ex_bound_divE}, {bound_divE}, {ex_int_rotE}, {int_rotE}, {ex_bound_rotE}, {bound_rotE}, {runtime:0.4f}\n')


if __name__ == "__main__":
    main()
