#!/usr/bin/env python

from xnma import nma
from xnma import kernels

import inspect, os.path
import matplotlib.pyplot as plt
import numpy as np
import time


def main():
    # Get full path to examples/
    # From https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path     = os.path.dirname(os.path.abspath(filename))
    
    model = nma.model()

    model.construct(dx=0.5,dy=0.5,nx=101,ny=101)
    
    s = np.ones(model.maskInC.shape,dtype=np.float32)

    ny, nx = s.shape    

    s0 = np.zeros(model.maskInC.shape,dtype=np.float32)
    b = np.zeros(model.maskInC.shape,dtype=np.float32)

    wx = 2.0*np.pi/(model.xg[-1]-1.0)
    wy = 2.0*np.pi/(model.yg[-1]-1.0)

    # RHS of laplacian is calculated so that
    # s = sin(2*pi*x/Lx)*sin(2*pi*y/Ly)
    # is the solution to the laplacian equation
    #
    for j in range(0,model.yc.shape[0]):
      y = model.yc[j]-model.yc[1]
      for i in range(0,model.xc.shape[0]):
        x = model.xc[i]-model.xc[1]
        b[j,i] = - ( wx**2 + wy**2 )*np.cos(wx*x)*np.cos(wx*y)

    tic = time.perf_counter()
    s = model.LapCInv_PCCG(b, s0=None, itermax=100, tolerance=1e-3)
    toc = time.perf_counter()

    runtime = toc-tic
    print(f"Preconditioned Conjugate Gradient solve runtime : {runtime:0.4f} s")

    plt.contourf(model.xc,model.yc,s)
    plt.colorbar()
    plt.show()


if __name__=="__main__":
    main()



