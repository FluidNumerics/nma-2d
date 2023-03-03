#!/usr/bin/env python

from xnma import nma
from xnma import kernels

import inspect, os.path
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

def main():
    # Get full path to examples/
    # From https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path     = os.path.dirname(os.path.abspath(filename))
    
    model = nma.model()

    model.construct(dx=1.0,dy=1.0,nx=501,ny=501)
    
    s = np.ones(model.mask.shape,dtype=np.float32)

    ny, nx = s.shape    

    nrw = (nx)*(ny)*19
    nops = (nx)*(ny)*18

    # Pre-allocate output array to improve performance of kernel
    Ls = np.zeros_like(s)

    tic = time.perf_counter()
    for i in range(0,10):
       Ls = kernels.LapZ( s, model.dxc, model.dyc, model.dxg, model.dyg, model.raz ) 
    toc = time.perf_counter()

    runtime = toc-tic
    bandwidth = ((nrw*4)/runtime)*1e-6
    flops = ((nops)/runtime)*1e-6
    ai = flops/bandwidth

    print(f"laplacian kernel runtime : {runtime:0.4f} s")
    print(f"laplacian arithmetic intensity : {ai:0.4f} FLOPS/B")
    print(f"laplacian effective bandwidth: {bandwidth:0.4f} MB/s")
    print(f"laplacian effective FLOPS: {flops:0.4f} MFLOPs/s")


if __name__=="__main__":
    main()



