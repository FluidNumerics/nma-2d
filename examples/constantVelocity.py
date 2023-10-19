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

plt.style.use('seaborn-v0_8-whitegrid')
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
      kx[m] = (m)*np.pi/Lx
        
    ky = np.zeros(nmy)
    for m in range(1, nmy):
      ky[m] = (m)*np.pi/Ly

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

    # Fill in example u,v
    for j in range(0, model.yg.shape[0]):
        for i in range(0, model.xg.shape[0]):
            u[j, i] = 1.0
            v[j, i] = 0.0

    # Calculate total energy
    uc = kernels.UtoT(u)
    vc = kernels.VtoT(v)
    Etot = np.sum(0.5 * (uc * uc + vc * vc) * model.rac * model.maskC)
    

    # Calculate total energy
    uc = kernels.UtoT(u)
    vc = kernels.VtoT(v)
    Etot = np.sum(0.5 * (uc * uc + vc * vc) * model.rac * model.maskC)
    ( 
        lambda_m,
        sigma_m,
        Edi_m,
        Eri_m,
        Edb_m,
        Erb_m 
    ) = model.spectra(u,v,decimals=2)

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
        lambda_m_exact,
        sigma_m_exact,
        Edi_m_exact,
        Eri_m_exact,
        Edb_m_exact,
        Erb_m_exact 
    ) = exmodel.spectra(u,v,decimals=2)

    # [0:-1] indexing for neumann modes is done to exlude the mode associated with eigenvalue of 0
    plt.figure(figsize=(8.4,4.8))
    plt.plot( np.abs(lambda_m_exact[0:-1]), Edb_m_exact[0:-1],'-o', label='Divergent (exact)', markersize=3, linewidth=1)
    plt.plot( np.abs(sigma_m_exact), Erb_m_exact,'-o', label='Rotational (exact)', markersize=3, linewidth=1 )
    plt.plot( np.abs(lambda_m[0:-1]), Edb_m[0:-1],'-x', label='Divergent (numerical)', markersize=4, linewidth=1 )
    plt.plot( np.abs(sigma_m), Erb_m,'-x', label='Rotational (numerical)', markersize=4, linewidth=1 )
    plt.title("spectra (boundary contribution)")
    plt.xlabel("$\lambda$")
    plt.ylabel("energy")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"constantvelocity-boundary-energy_{nx-3}_{n_numerical_modes}.png")        

    plt.figure(figsize=(8.4,4.8))
    plt.plot( np.abs(lambda_m_exact[0:-1]), Edi_m_exact[0:-1],'-o', label='Divergent (exact)', markersize=3, linewidth=1)
    plt.plot( np.abs(sigma_m_exact), Eri_m_exact,'-o', label='Rotational (exact)', markersize=3, linewidth=1 )
    plt.plot( np.abs(lambda_m[0:-1]), Edi_m[0:-1],'-x', label='Divergent (numerical)', markersize=4, linewidth=1 )
    plt.plot( np.abs(sigma_m), Eri_m,'-x', label='Rotational (numerical)', markersize=4, linewidth=1 )
    plt.title("spectra (interior)")
    plt.xlabel("$\lambda$")
    plt.ylabel("energy")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"constantvelocity-interior-energy_{nx-3}_{n_numerical_modes}.png")

    csvfile = './constantvelocity-energy.csv'
    with open(csvfile, 'a') as f:
        if( os.path.getsize(csvfile) == 0 ):
            f.write('nx,ny,nmodes,"interior divergent energy (exact)","interior divergent energy (numerical)","boundary divergent energy (exact)","boundary divergent energy (numerical)","interior rotational energy (exact)","interior rotational energy (numerical)","boundary rotational energy (exact)","boundary rotational energy (numerical)","integrated total energy","eigenmode search runtime (s)" \n')
       
        ex_int_divE = np.sum(Edi_m_exact)
        ex_int_rotE = np.sum(Eri_m_exact)
        int_divE = np.sum(Edi_m)
        int_rotE = np.sum(Eri_m)
        ex_bound_divE = np.sum(Edb_m_exact)
        ex_bound_rotE = np.sum(Erb_m_exact)
        bound_divE = np.sum(Edb_m)
        bound_rotE = np.sum(Erb_m)
        f.write(f'{nx-3},{ny-3},{n_numerical_modes},{ex_int_divE},{int_divE},{ex_bound_divE},{bound_divE},{ex_int_rotE},{int_rotE},{ex_bound_rotE},{bound_rotE},{Etot},{runtime:0.4f}\n')


if __name__ == "__main__":
    main()
