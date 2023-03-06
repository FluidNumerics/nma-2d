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

nmodes = 20

def main():
    # Get full path to examples/
    # From https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path     = os.path.dirname(os.path.abspath(filename))

    
    model = nma.model()

    model.irregularHolesDemo(dx=0.5,dy=0.5,nx=101,ny=101)

    model.findEigenmodes( nmodes = nmodes )

    print( model.eigenvalues )
    
    plt.figure(figsize=(10,12))
    plt.subplots_adjust(hspace=1.0,wspace=0.5)
    plt.suptitle("Eigenmodes", fontsize=18, y=0.95)

    sgrid = ma.masked_array( np.zeros(model.mask.shape), mask=abs(model.mask - 1.0), dtype=np.float32 )
    for k in range(0,16):
        ei = nmodes-17+k
        sgrid[~sgrid.mask] = model.eigenmodes[:,ei]

        # add a new subplot iteratively
        ax = plt.subplot(4, 4, k+1)
    
        #plt.contourf(model.xg, model.yg, sgrid)
        plt.pcolor(model.xg, model.yg, sgrid,vmin=-0.05, vmax=0.05)
        plt.set_cmap("cividis")
        # chart formatting
        ax.set_title(f"e_{ei}")
        ax.set_xlabel("x_g")
        ax.set_ylabel("y_g")
        plt.colorbar()


    plt.show()

    #sgrid[~sgrid.mask] = model.eigenmodes[:,1]

    #plt.contourf(model.xg, model.yg, sgrid)
    #plt.colorbar()
    #plt.show()

    #sgrid[~sgrid.mask] = model.eigenmodes[:,-1]

    #plt.contourf(model.xg, model.yg, sgrid)
    #plt.colorbar()
    #plt.show()

if __name__=="__main__":
    main()



