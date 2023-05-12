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

nmodes = 100
# Set model dimensions
Lx = 1.0
Ly = 1.0
nx = 28
ny = 28    

# Calculate the grid spacing 
# so that the western most "wet"
# xg point is at 0 and the eastern
# most xg point is at 1
dx = Lx/(nx-3)
dy = Ly/(ny-3)

def main():
    
    # Initialize the nma model
    model = nma.model()

    # Construct the model
    model.construct(dx,dy,nx,nx)

    # Find the eigenmodes (all on tracer points)
    model.findEigenmodes( nmodes = nmodes ) 
    
    u = np.zeros( (ny,nx), dtype=np.float32 )
    v = np.zeros( (ny,nx), dtype=np.float32 )
    # Fill in example u,v
    for j in range(0,model.yg.shape[0]):
      yg = model.yg[j]
      yc = model.yc[j]
      for i in range(0,model.xg.shape[0]):
        xg = model.xg[i]
        xc = model.xc[i]
        
       # u[j,i] = xg**2 + yc**2
       # v[j,i] = xc**2 - yg**2
        u[j,i] = 1.0 #xg
        v[j,i] = 1.0 #yg
        
    
    # Find the projection
    d_m, divergence, v_m, vorticity, proj_d, proj_v = model.vectorProjection(u,v)

    plt.figure(figsize=(10,12))
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    plt.suptitle("simple test", fontsize=18, y=0.95)
    divU = ma.masked_array( divergence, mask=abs(model.maskC - 1.0), dtype=np.float32 )
    curlU = ma.masked_array( vorticity, mask=abs(model.maskC - 1.0), dtype=np.float32 )
    projDivU = ma.masked_array( proj_d, mask=abs(model.maskC - 1.0), dtype=np.float32 )
    projCurlU = ma.masked_array( proj_v, mask=abs(model.maskC - 1.0), dtype=np.float32 )
    
    print( np.min(divergence) )
    print( np.max(divergence) )
    print( np.min(vorticity) )
    print( np.max(vorticity) )
    # add a new subplot iteratively
    ax = plt.subplot(3,2,1) 
    plt.pcolor(model.xc, model.yc, divU)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("$\nabla \cdot \vec{u}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    # add a new subplot iteratively
    ax = plt.subplot(3,2,2) 
    plt.pcolor(model.xc, model.yc, projDivU)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("proj$(nabla \cdot \vec{u}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    # add a new subplot iteratively
    ax = plt.subplot(3,2,3) 
    plt.pcolor(model.xg, model.yg, curlU)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("$\nabla \times \vec{u}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    # add a new subplot iteratively
    ax = plt.subplot(3,2,4) 
    plt.pcolor(model.xc, model.yc, projCurlU)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("proj$(\nabla \times \vec{u})$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    ax = plt.subplot(3,2,5) 
    plt.plot(d_m**2+v_m**2)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("spectra")
    #ax.set_xlabel("$\lambda$")
    ax.set_xlabel("n")
    ax.set_ylabel("E_n")

    plt.show()

    plt.figure(figsize=(10,12))
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    plt.suptitle("spectra", fontsize=18, y=0.95)
    


if __name__=="__main__":
    main()



