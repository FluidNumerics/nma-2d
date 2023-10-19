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

plt.style.use('dark_background')
plt.switch_backend('agg')

parser = argparse.ArgumentParser(
                    prog='constantVelocity',
                    description='Calculates spectra of constant vector field')

parser.add_argument('-x', '--nx',default=13, type=int)
parser.add_argument('-y', '--ny',default=13, type=int)
parser.add_argument('-m', '--nmodes',default=40, type=int)
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


def main():
    # Initialize the nma model
    model = nma.model()

    # Construct the model
    model.irregularHolesDemo(dx, dy, nx, nx, prec)

    # Find the eigenmodes (all on tracer points)
    if shift == 0.0 :
        deShift = 0.0
        neShift = -1e-4
    else:
        deShift = shift
        neShift = shift
        
    model.findEigenmodes(nmodes=n_numerical_modes, tolerance=tolerance, deShift=deShift, neShift=neShift)
    plt.figure()

    ei=12
    sgrid = ma.masked_array( model.d_eigenmodes[ei,:,:].squeeze(), mask=abs(model.maskZ - 1.0), dtype=np.float32 )
    plt.pcolor(model.xg, model.yg, sgrid, vmin=-1.0, vmax=1.0)
    plt.set_cmap("cividis")
    # chart formatting
    plt.savefig(f"d-{ei}.png")
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
    plt.savefig("numerical-eigenmodes.png")
    plt.close()


if __name__=="__main__":
    main()



