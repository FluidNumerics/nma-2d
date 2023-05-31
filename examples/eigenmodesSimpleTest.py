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

nmodes = 80

def main():
    # Get full path to examples/
    # From https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path     = os.path.dirname(os.path.abspath(filename))
    
    model = nma.model()

    Lx = 1.0
    Ly = 1.0
    nX = 103
    nY = 103
    
    dx = Lx/(nX-3)
    dy = Ly/(nY-3)
    
    model.construct(dx,dy,nX,nX)

    model.findEigenmodes( nmodes = nmodes ) 

    #print( model.eigenvalues )
    
    plt.figure(figsize=(10,12))
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    plt.suptitle("Eigenmodes", fontsize=18, y=0.95)

    for k in range(0,10):
        ei = k #nmodes-17+k
        sgrid = ma.masked_array( model.d_eigenmodes[ei,:,:].squeeze(), mask=abs(model.maskC - 1.0), dtype=np.float32 )
        # add a new subplot iteratively
        ax = plt.subplot(4, 5, k+1)
    
        plt.pcolor(model.xc, model.yc, sgrid, vmin=-1.0, vmax=1.0)
        plt.set_cmap("cividis")
        # chart formatting
        ax.set_title(f"d_{ei}")
        #ax.set_xlabel("x_g")
        #ax.set_ylabel("y_g")
        plt.colorbar()
        
    for k in range(0,10):
        ei = k #nmodes-17+k
        sgrid = ma.masked_array( model.n_eigenmodes[ei,:,:].squeeze(), mask=abs(model.maskC - 1.0), dtype=np.float32 )
        # add a new subplot iteratively
        ax = plt.subplot(4, 5, k+11)
    
        plt.pcolor(model.xc, model.yc, sgrid, vmin=-1.0, vmax=1.0)
        plt.set_cmap("cividis")
        # chart formatting
        ax.set_title(f"n_{ei}")
        #ax.set_xlabel("x_g")
        #ax.set_ylabel("y_g")
        plt.colorbar()


    plt.show()
    
    plt.figure(figsize=(10,12))
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    plt.suptitle("Eigenvalues", fontsize=18, y=0.95)
    ax = plt.subplot(1,2,1)
    plt.plot(np.abs(model.d_eigenvalues),label = 'dirichlet',marker='o')
    plt.plot(np.abs(model.n_eigenvalues),label = 'neumann',marker='o')
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()




if __name__=="__main__":
    main()



