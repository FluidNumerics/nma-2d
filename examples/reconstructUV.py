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


def main():
    # Get full path to examples/
    # From https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path     = os.path.dirname(os.path.abspath(filename))
    
    model = nma.model()

    model.load(f'{path}/data/',iters=0, depth=10, x=[8,12], y=[34,38])


    u = model.U.to_numpy().squeeze().astype(np.float32)
    v = model.V.to_numpy().squeeze().astype(np.float32)
    zeta = np.zeros( model.maskInZ.shape, dtype=np.float32 ) 
    zeta = kernels.vorticity( u, v, model.dxc, model.dyc, model.raz  )

    divu = np.zeros( model.maskInC.shape, dtype=np.float32 ) 
    divu = kernels.divergence( u, v, model.dxg, model.dyg, model.hFacS, model.hFacW, model.rac  )

    ## Calculate the rotational component
    #psi = np.ones(model.maskInZ.shape,dtype=np.float32)
    #psi = model.LapZInv_PCCG(zeta, s0=None, 
    #        pcitermax=20, pctolerance=1e-2, itermax=1500,
    #        tolerance=1e-12)

    #ur = kernels.ur(psi,model.dyg)
    #vr = kernels.vr(psi,model.dxg)


    ## Calculate the divergent component
    #phi = np.ones(model.maskInC.shape,dtype=np.float32)
    #phi = model.LapCInv_PCCG(zeta, s0=None, 
    #        pcitermax=20, pctolerance=1e-2, itermax=1500,
    #        tolerance=1e-12)

    #ud = kernels.ud(psi,model.dxc)
    #vd = kernels.vd(psi,model.dyc)


    plt.figure(figsize=(10,8))
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    plt.suptitle("Velocity", fontsize=18, y=0.95)

    ax = plt.subplot(3, 2, 1)
    plt.pcolor(model.xg, model.yc, u, vmin=-1.8, vmax=1.8)
    plt.set_cmap("cividis")
    ax.set_title(f"u")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    ax = plt.subplot(3, 2, 2)
    plt.pcolor(model.xg, model.yc, v, vmin=-1.2, vmax=1.2)
    plt.set_cmap("cividis")
    ax.set_title(f"v")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()

    ax = plt.subplot(3, 2, 3)
    plt.pcolor(model.xg, model.yg, zeta)
    plt.set_cmap("cividis")
    ax.set_title(f"vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    ax = plt.subplot(3, 2, 4)
    plt.pcolor(model.xg, model.yg, divu)
    plt.set_cmap("cividis")
    ax.set_title(f"divergence")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()


#    ax = plt.subplot(3, 2, 3)
#    plt.pcolor(model.xg, model.yg, zeta)
#    plt.set_cmap("cividis")
#    ax.set_title(f"vorticity")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    plt.colorbar()
#
#    ax = plt.subplot(3, 2, 4)
#    plt.pcolor(model.xc, model.yc, divu)
#    plt.set_cmap("cividis")
#    ax.set_title(f"div(u)")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    plt.colorbar()


#    ax = plt.subplot(3, 2, 3)
#    plt.pcolor(model.xg, model.yc, ur, vmin=-1.8, vmax=1.8)
#    plt.set_cmap("cividis")
#    ax.set_title(f"u_r")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    plt.colorbar()
#
#    ax = plt.subplot(3, 2, 4)
#    plt.pcolor(model.xc, model.yg, vr, vmin=-1.2, vmax=1.2)
#    plt.set_cmap("cividis")
#    ax.set_title(f"v_r")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    plt.colorbar()
#
#
#    ax = plt.subplot(3, 2, 5)
#    plt.pcolor(model.xg, model.yc, ud, vmin=-1.8, vmax=1.8)
#    plt.set_cmap("cividis")
#    ax.set_title(f"ud")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    plt.colorbar()
#
#    ax = plt.subplot(3, 2, 6)
#    plt.pcolor(model.xc, model.yg, vd, vmin=-1.2, vmax=1.2)
#    plt.set_cmap("cividis")
#    ax.set_title(f"vd")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    plt.colorbar()


#    ax = plt.subplot(3, 2, 5)
#    plt.pcolor(model.xg, model.yc, u-ur, vmin=-1.8, vmax=1.8)
#    plt.set_cmap("cividis")
#    ax.set_title(f"u-u_r")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    plt.colorbar()
#
#    ax = plt.subplot(3, 2, 6)
#    plt.pcolor(model.xc, model.yg, v-vr, vmin=-1.2, vmax=1.2)
#    plt.set_cmap("cividis")
#    ax.set_title(f"v-v_r")
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    plt.colorbar()


    plt.show()



if __name__=="__main__":
    main()



