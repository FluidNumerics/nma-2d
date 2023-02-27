#!/usr/bin/env python
#

# A scipy linear operator ( https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html )
# Used in the linalg.eigs method (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html)
#from scipy.sparse.linalg import LinearOperator

class model:
    def __init__(self):

        self.ds = None
        self.grid = None

        self.ndof = 0
        self.mask = None

        self.eigenmodes = None #
        self.eigenvalues = None #

    def loadGrid(self, dataDir, depth=0,x=None,y=None):
        """Loads in grid from MITgcm metadata files in dataDir
        and configures masks at given depth"""
        import numpy as np
        import numpy.ma as ma
        import xmitgcm
        import xgcm

        self.ds = xmitgcm.open_mdsdataset(dataDir,
                iters=None,prefix=None,read_grid=True,
                geometry = "sphericalpolar")

        if x:
            self.ds = self.ds.sel(XC=slice(x[0],x[1]),XG=slice(x[0],x[1]))

        if y:
            self.ds = self.ds.sel(YC=slice(y[0],y[1]),YG=slice(y[0],y[1]))

        # Create a grid object
        self.grid = xgcm.Grid(self.ds)

        zd = -abs(depth) #ensure that depth is negative value
        self.hFacC = self.ds.hFacC.interp(Z=[zd],method="nearest").squeeze()
        self.hFacW = self.ds.hFacW.interp(Z=[zd],method="nearest").squeeze()
        self.hFacS = self.ds.hFacS.interp(Z=[zd],method="nearest").squeeze()
        self.mask = abs(np.ceil(self.hFacC['hFacC'][:,:])-1.0).to_numpy().astype(int)

        wetcells = abs(self.mask-1.0)
        self.ndof = wetcells.sum().astype(int)

    def setMask(self, mask):
        """Sets an additional mask by multiplying by the input mask"""

        self.hFacC = self.hFacC*mask
        self.hFacW = self.hFacW*mask
        self.hFacS = self.hFacS*mask
        self.mask = abs(np.ceil(self.hFacC['hFacC'][:,:])-1.0).to_numpy().astype(int)

