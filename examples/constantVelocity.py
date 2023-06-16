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

nmodes = 80
# Set model dimensions
Lx = 1.0
Ly = 1.0
nx = 28
ny = 28

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
    model.construct(dx, dy, nx, nx)

    # Find the eigenmodes (all on tracer points)
    model.findEigenmodes(nmodes=nmodes)

    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)

    # Fill in example u,v
    for j in range(0, model.yg.shape[0]):
        for i in range(0, model.xg.shape[0]):
            u[j, i] = 1.0
            v[j, i] = 0.0

 #   u = u * model.maskW
 #   v = v * model.maskS

    # Calculate total energy
    uc = kernels.UtoT(u)
    vc = kernels.VtoT(v)
    Etot = np.sum(0.5 * (uc * uc + vc * vc) * model.rac * model.maskC)

    # Find the projection
    (
        rotEnergy,
        divEnergy,
        proj_d,
        proj_v,
        divergence,
        vorticity,
        d_m,
        v_m,
    ) = model.vectorProjection(u, v)

    plt.figure(figsize=(10, 12))
    plt.subplots_adjust(hspace=1.0, wspace=0.5)
    plt.suptitle("simple test", fontsize=18, y=0.95)

    # add a new subplot iteratively
    ax = plt.subplot(1, 2, 1)
    plt.pcolor(model.xg, model.yc, u, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("u")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    # add a new subplot iteratively
    ax = plt.subplot(1, 2, 2)
    plt.pcolor(model.xc, model.yg, v, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("v")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    plt.figure(figsize=(10, 12))
    plt.subplots_adjust(hspace=1.0, wspace=0.5)
    plt.suptitle("simple test", fontsize=18, y=0.95)
    divU = ma.masked_array(divergence, mask=abs(model.maskC - 1.0), dtype=np.float32)
    curlU = ma.masked_array(vorticity, mask=abs(model.maskZ - 1.0), dtype=np.float32)
    projCurlUd = ma.masked_array(proj_v, mask=abs(model.maskZ - 1.0), dtype=np.float32)
    projDivUn = ma.masked_array(proj_d, mask=abs(model.maskC - 1.0), dtype=np.float32)

    # add a new subplot iteratively
    ax = plt.subplot(2, 2, 1)
    plt.pcolor(model.xc, model.yc, divU)  # , vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("divergence")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    # add a new subplot iteratively
    ax = plt.subplot(2, 2, 2)
    plt.pcolor(model.xg, model.yg, curlU)  # , vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("curl")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    # add a new subplot iteratively
    ax = plt.subplot(2, 2, 3)
    plt.pcolor(model.xc, model.yc, projDivUn)  # , vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("divergence (neumann projection)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    # add a new subplot iteratively
    ax = plt.subplot(2, 2, 4)
    plt.pcolor(model.xg, model.yg, projCurlUd)  # , vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("vorticity (dirichlet projection)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    plt.figure(figsize=(10, 12))
    plt.subplots_adjust(hspace=1.0, wspace=0.5)
    plt.suptitle("simple test - projection differences", fontsize=18, y=0.95)

    # add a new subplot iteratively
    ax = plt.subplot(2, 2, 1)
    plt.pcolor(model.xc, model.yc, divU, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("divergence")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    # add a new subplot iteratively
    ax = plt.subplot(2, 2, 2)
    plt.pcolor(model.xc, model.yc, curlU, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("curl")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    # add a new subplot iteratively
    ax = plt.subplot(2, 2, 3)
    plt.pcolor(model.xc, model.yc, divU - projDivUn)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("divergence (neumann projection)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    # add a new subplot iteratively
    ax = plt.subplot(2, 2, 4)
    plt.pcolor(model.xc, model.yc, curlU - projCurlUd)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("vorticity (dirichlet projection)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    plt.figure()
    plt.subplots_adjust(hspace=1.0, wspace=0.5)
    ax = plt.subplot(1, 1, 1)
    plt.plot(
        np.abs(model.n_eigenvalues), divEnergy, label="Divergence (Neumann)", marker="o"
    )
    plt.plot(
        np.abs(model.d_eigenvalues), rotEnergy, label="Vorticity(Dirichlet)", marker="o"
    )
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("spectra")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("energy")
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.grid()
    ax.legend(loc="upper right")

    # Plot the energy
    plt.figure()
    plt.subplots_adjust(hspace=1.0, wspace=0.5)
    ax = plt.subplot(1, 1, 1)
    plt.plot(
        np.abs(model.n_eigenvalues),
        np.cumsum(divEnergy),
        label="Divergence (Neumann)",
        marker="o",
    )
    plt.plot(
        np.abs(model.d_eigenvalues),
        np.cumsum(rotEnergy),
        label="Vorticity(Dirichlet)",
        marker="o",
    )
    plt.plot(
        [0, -model.d_eigenvalues[-1]],
        [Etot, Etot],
        "k--",
        label="Total Energy (Exact)",
    )
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("Integrated Spectra")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("energy")
    # ax.set_yscale('log')
    plt.grid()
    ax.legend(loc="right")
    plt.show()

    print(np.sum(divEnergy))
    print(np.sum(rotEnergy))


if __name__ == "__main__":
    main()
