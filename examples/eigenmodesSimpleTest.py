#!/usr/bin/env python

from xnma import nma
from xnma import kernels

import inspect, os.path
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import sys
import time
import argparse

#plt.style.use('dark_background')
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
    if shift == 0.0 :
        deShift = 0.0
        neShift = 1e-2
    else:
        deShift = shift
        neShift = shift
        
    model.findEigenmodes(nmodes=n_numerical_modes, tolerance=tolerance, deShift=deShift, neShift=neShift)

    #print(model.n_eigenvalues)
    n_eigenvalues, n_eigenmodes = NeumannModes(model)
    d_eigenvalues, d_eigenmodes = DirichletModes(model)

    with open('eigenvalues.csv', 'w') as f:
        f.write('n, numerical dirichelt, exact dirichlet, numerical neumann, exact neumann \n')
        for k in range(0,n_numerical_modes-1):
            f.write(f'{k},{-model.d_eigenvalues[k]},{d_eigenvalues[k]},{-model.n_eigenvalues[k]},{n_eigenvalues[k]} \n')

    n=n_numerical_modes-1
    dirichlet_err = np.sqrt(np.sum(np.abs(d_eigenvalues[0:n]+model.d_eigenvalues[0:n])))
    neumann_err = np.sqrt(np.sum(np.abs(n_eigenvalues[0:n]+model.n_eigenvalues[0:n])))
  #  print( f"Dirichlet e-value error : {dirichlet_err}")
  #  print( f"Neumann e-value error : {neumann_err}")

    plt.figure()
    plt.title(f"Eigenvalues {nx-3} x {nx-3}")
    plt.plot(np.abs(d_eigenvalues[0:n_numerical_modes-1]),'-o',label = 'dirichlet (exact)', markersize=3, linewidth=1 )
    plt.plot(np.abs(n_eigenvalues[0:n_numerical_modes-1]),'-o',label = 'neumann (exact)', markersize=3, linewidth=1 )
    plt.plot(np.abs(model.d_eigenvalues[0:n_numerical_modes-1]),'-x',label = 'dirichlet (numerical)', markersize=4, linewidth=1 )
    plt.plot(np.abs(model.n_eigenvalues[0:n_numerical_modes-1]),'-x',label = 'neumann (numerical)', markersize=4, linewidth=1 )
    plt.legend(loc='upper left')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.savefig(f"eigenvalues-{nx-3}_{n_numerical_modes}.png")
            
    maxeval = np.max(np.abs(d_eigenvalues[0:n_numerical_modes-1]))

    plt.figure()
    plt.title(f"Eigenvalues {nx-3} x {nx-3}")
    plt.plot(np.abs(d_eigenvalues[0:n_numerical_modes-1]),np.abs(model.d_eigenvalues[0:n_numerical_modes-1]),'o',label = 'dirichlet', markersize=3, linewidth=1 )
    plt.plot(np.abs(n_eigenvalues[0:n_numerical_modes-1]),np.abs(model.n_eigenvalues[0:n_numerical_modes-1]),'x',label = 'neumann', markersize=3, linewidth=1 )
    plt.plot([0,maxeval],[0,maxeval],'--',label='match', linewidth=1 )
    plt.legend(loc='upper left')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("Exact")
    plt.ylabel("Numerical")
    plt.savefig(f"evalcomp-{nx-3}_{n_numerical_modes}.png")
    
    plt.figure(figsize=(10,12))
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    plt.suptitle("Eigenmodes", fontsize=18, y=0.95)

    for k in range(0,10):
        ei = k #nmodes-17+k
        sgrid = ma.masked_array( d_eigenmodes[ei,:,:].squeeze(), mask=abs(model.maskZ - 1.0), dtype=np.float32 )
        # add a new subplot iteratively
        ax = plt.subplot(4, 5, k+1)
    
        plt.pcolor(model.xg, model.yg, sgrid, vmin=-1.0, vmax=1.0)
        plt.set_cmap("cividis")
        # chart formatting
        ax.set_title(f"d_{ei}")
        ax.set_xlabel("x_g")
        ax.set_ylabel("y_g")
        plt.colorbar()

    for k in range(0,10):
        ei = k
        sgrid = ma.masked_array( n_eigenmodes[ei,:,:].squeeze(), mask=abs(model.maskC - 1.0), dtype=np.float32 )
        # add a new subplot iteratively
        ax = plt.subplot(4, 5, k+11)
    
        plt.pcolor(model.xc, model.yc, sgrid, vmin=-1.0, vmax=1.0)
        plt.set_cmap("cividis")
        # chart formatting
        ax.set_title(f"n_{ei}")
        ax.set_xlabel("x_c")
        ax.set_ylabel("y_c")
        plt.colorbar()
    plt.savefig(f"exact-eigenmodes-{nx-3}_{n_numerical_modes}.png")

    plt.close()

    plt.figure(figsize=(10,12))
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    plt.suptitle("Eigenmodes", fontsize=18, y=0.95)

    for k in range(0,10):
        ei = k #nmodes-17+k
        sgrid = ma.masked_array( model.d_eigenmodes[ei,:,:].squeeze(), mask=abs(model.maskZ - 1.0), dtype=np.float32 )
        # add a new subplot iteratively
        ax = plt.subplot(4, 5, k+1)
    
        plt.pcolor(model.xg, model.yg, sgrid, vmin=-1.0, vmax=1.0)
        plt.set_cmap("cividis")
        # chart formatting
        ax.set_title(f"d_{ei}")
        ax.set_xlabel("x_g")
        ax.set_ylabel("y_g")
        plt.colorbar()

    for k in range(0,10):
        ei = k
        sgrid = ma.masked_array( model.n_eigenmodes[ei,:,:].squeeze(), mask=abs(model.maskC - 1.0), dtype=np.float32 )
        # add a new subplot iteratively
        ax = plt.subplot(4, 5, k+11)
    
        plt.pcolor(model.xc, model.yc, sgrid, vmin=-1.0, vmax=1.0)
        plt.set_cmap("cividis")
        # chart formatting
        ax.set_title(f"n_{ei}")
        ax.set_xlabel("x_c")
        ax.set_ylabel("y_c")
        plt.colorbar()
    plt.savefig(f"numerical-eigenmodes-{nx-3}_{n_numerical_modes}.png")
    plt.close()
 

if __name__=="__main__":
    main()



