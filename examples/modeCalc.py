#!/usr/bin/env python

import nmaspectra.nma as nma
import inspect, os.path
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import numpy.ma as ma
import sys
import cProfile

def laplacian( s, model ):
    """ Defines the Laplacian operator for the Neumann Modes """

    # s comes in DOF format
    # >> map to ij format
    sgrid = ma.array( np.zeros(model.mask.shape), mask = model.mask )
    sgrid[~sgrid.mask] = s

    # smodel is defined this way to inherit the properties of hFacC
    # it is a floating point dataset centered on tracer points
    # on the arakawa c-grid
    # We multiply hfacC by 0 to create a null field
    # and add sgrid.data to it to get the values we want
    # associated with tracer points.
    smodel = model.hFacC*0.0+sgrid.data;

    # Calculate grad s
    # Homogeneous neumann boundary conditions are applied by 
    #   > multiplying "u" by "hFacW"
    #   > multiplying "v" by "hFacS"
    #   > using the "exend" boundary conditions on domain boundaries
    # For the latter case, see 
    #    https://xgcm.readthedocs.io/en/latest/boundary_conditions.html 
    # for more details
    #u = model.grid.diff(smodel,'X',boundary="extend",to="left")
    u = model.grid.diff(smodel,'X',boundary="extend")
    u = u * model.ds.dyG * model.hFacW
    #v = model.grid.diff(smodel,'Y',boundary="extend",to="left")
    v = model.grid.diff(smodel,'Y',boundary="extend")
    v = v * model.ds.dxG * model.hFacS

    # See https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html#Divergence
    divUV = (model.grid.diff(u,'X',to="center") + model.grid.diff(v,'Y',to="center"))/model.ds.rA

    # Map to DOF format, store in As
    As = ma.array( divUV.to_numpy(), mask = model.mask ).compressed()
    return As

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
    #model.loadGrid(f'{path}/data/',depth=-1000.0,y=[33,37],x=[5,9])
    model.loadGrid(f'{path}/data/',depth=-1000.0,y=[35,36],x=[7,8])
    
    shape = (model.ndof,model.ndof)
    L = LinearOperator(shape, matvec=lambda x: laplacian(x, model))
    #Linv = LinearOperator(shape, matvec=lambda b: invLaplacian(b, model))
    #evals, evecs = eigsh(L, OPinv=Linv, k=2, sigma=0.0, which='SM', tol=1e-7 )
    evals, evecs = eigsh(L, k=10, tol=1e-7 )
    
    # Plot the last eigenvalue
   # sgrid = ma.array( np.zeros(model.mask.shape), mask = model.mask )

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

   # s = ma.array( np.zeros(model.mask.shape), mask = model.mask )
   # s[~s.mask] = np.squeeze(evecs[:,-1])
   # plt.contourf(model.ds.XC,model.ds.YC,s)
   # plt.colorbar()
   # plt.show()

if __name__=="__main__":
    main()



