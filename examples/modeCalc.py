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





def invLaplacian( b, model ):
    """ Calculates the inverse of the Laplacian using 
    the conjugate-gradient method """

    shape = (model.ndof,model.ndof)
    L = LinearOperator(shape, matvec=lambda x: laplacian(x, model))
    xsol, exit_code = cg(L,b)
    if exit_code != 0:
        print("Error")
        sys.exit(1)
    return xsol

def main():
    # Get full path to examples/
    # From https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path     = os.path.dirname(os.path.abspath(filename))
    
    model = nma.model()
    model.loadGrid(f'{path}/data/',
                   depth=-1000.0,y=[35.0,36.0],x=[7.0,8.0])
    
    s = np.ones(model.mask.shape,dtype=np.float32)

    print(s)
    tic = time.perf_counter()
    for i in range(0,10):
       Ls = kernels.LapZ( s, model.dxc, model.dyc, model.dxg, model.dyg, model.raz ) 
    toc = time.perf_counter()
    print(Ls)

    print(f"laplacian kernel runtime : {toc - tic:0.4f} s")


   # uwork = np.ones(model.mask.shape,dtype=np.float32)

    #shape = (model.ndof,model.ndof)
#    L = LinearOperator(shape, matvec=lambda x: laplacian(x, model))
    ##Linv = LinearOperator(shape, matvec=lambda b: invLaplacian(b, model))
    ##evals, evecs = eigsh(L, OPinv=Linv, k=2, sigma=0.0, which='SM', tol=1e-7 )
#    evals, evecs = eigsh(L, k=10, tol=1e-7 )
    
#    dx =  model.ds.dxC.to_numpy().astype(np.float32)
#    dy =  model.ds.dyG.to_numpy().astype(np.float32)
#    hfac = model.ds.hFacW.to_numpy().astype(np.float32)
#    nx, ny = model.mask.shape
#    print(nx)
#    print(ny)
#
#    print(dx.shape)
#    print(dy.shape)
#
#    tic = time.perf_counter()
#
#    for i in range(0,10):
#       u = calcU( s, uwork, dx, dy, hfac, nx, ny )
#       
#    toc = time.perf_counter()
#
#    print(f"Laplacian runtime : {toc - tic:0.4f} s")

#        As = laplacian( s, model )
   
    # Plot the last eigenvalue
#    sgrid = ma.array( np.zeros(model.mask.shape), mask = model.mask )

   # fig, axs = plt.subplots(3,3)
   # k = 0
   # for ax in axs.ravel():
   #    
   #    sgrid[~sgrid.mask] = evecs[:,k]

   #    plt.axes(ax)
   #    plt.contourf(model.ds.XC,model.ds.YC,sgrid)
   #    plt.colorbar() 
   #    ax.set_title(f"evec {k}")

   #    k+=1

#    s = ma.array( np.zeros(model.mask.shape), mask = model.mask )
#    #s[~s.mask] = np.squeeze(evecs[:,-1])
#    s[~s.mask] = As
if __name__=="__main__":
    main()



