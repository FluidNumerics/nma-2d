#!/usr/bin/env python

from xnma import nma
from xnma import kernels

import inspect, os.path
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import cg

#import matplotlib
#matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
#matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
#    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
#matplotlib.rcParams['text.usetex'] = True


import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import sys
import time

nmodes = 80
# Set model dimensions
Lx = 1.0
Ly = 1.0
nx = 13
ny = 13    

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
        
        u[j,i] = xg**2 + yc**2
        v[j,i] = xc**2 - yg**2
       # u[j,i] = xg
       # v[j,i] = yg
        
    
    # Find the projection
    d_m, v_m, proj_d, proj_v, divergence, vorticity, alpha, beta = model.vectorProjection(u,v)

    plt.figure(figsize=(10,12))
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    plt.suptitle("simple test", fontsize=18, y=0.95)
    divU = ma.masked_array( divergence, mask=abs(model.maskC - 1.0), dtype=np.float32 )
    curlU = ma.masked_array( vorticity, mask=abs(model.maskC - 1.0), dtype=np.float32 )
    projDivUd = ma.masked_array( np.squeeze(proj_d[:,:,0]), mask=abs(model.maskC - 1.0), dtype=np.float32 )
    projCurlUd = ma.masked_array( np.squeeze(proj_v[:,:,0]), mask=abs(model.maskC - 1.0), dtype=np.float32 )
    projDivUn = ma.masked_array( np.squeeze(proj_d[:,:,1]), mask=abs(model.maskC - 1.0), dtype=np.float32 )
    projCurlUn = ma.masked_array( np.squeeze(proj_v[:,:,1]), mask=abs(model.maskC - 1.0), dtype=np.float32 )
    
    
    # add a new subplot iteratively
    ax = plt.subplot(3,2,1) 
    plt.pcolor(model.xc, model.yc, divU, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("divergence")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    # add a new subplot iteratively
    ax = plt.subplot(3,2,2) 
    plt.pcolor(model.xc, model.yc, curlU, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("curl")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    # add a new subplot iteratively
    ax = plt.subplot(3,2,3) 
    plt.pcolor(model.xc, model.yc, projDivUd, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("dirichlet")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    # add a new subplot iteratively
    ax = plt.subplot(3,2,4) 
    plt.pcolor(model.xc, model.yc, projCurlUd, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("dirichlet")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    # add a new subplot iteratively
    ax = plt.subplot(3,2,5) 
    plt.pcolor(model.xc, model.yc, projDivUn, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("neumann")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    # add a new subplot iteratively
    ax = plt.subplot(3,2,6) 
    plt.pcolor(model.xc, model.yc, projCurlUn, vmin=-1, vmax=1)
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("neumann")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    
    
    plt.figure()
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    ax = plt.subplot(1,1,1) 
    plt.plot(np.abs(model.d_eigenvalues),-alpha[:,0]*d_m[:,0],label="Divergence (Dirichlet)",marker="o")
    plt.plot(np.abs(model.n_eigenvalues),-alpha[:,1]*d_m[:,1],label="Divergence (Neumann)",marker="o")
    plt.plot(np.abs(model.d_eigenvalues),-beta[:,0]*v_m[:,0],label="Vorticity(Dirichlet)",marker="o")
    plt.plot(np.abs(model.n_eigenvalues),-beta[:,1]*v_m[:,1],label="Vorticity (Neumann)",marker="o")
    plt.set_cmap("cividis")
    # chart formatting
    ax.set_title("spectra")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("energy")
    ax.set_yscale('log')
    plt.grid()
    ax.legend(loc='upper right')

    plt.show()

    


if __name__=="__main__":
    main()



