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

        self.eigenmodes = None # Defined on vorticicity points
        self.eigenvalues = None #

    def loadGrid(self, dataDir, chunks=None, depth=0, x=None, y=None, geometry="sphericalpolar"):
        """Loads in grid from MITgcm metadata files in dataDir
        and configures masks at given depth"""
        import numpy as np
        import numpy.ma as ma
        import xmitgcm
        import xgcm
        from dask import array as da

        localChunks = None

        if chunks:
            localChunks = {'XC':chunks[0],
                           'XG':chunks[0],
                           'YC':chunks[1],
                           'YG':chunks[1]}

        self.ds = xmitgcm.open_mdsdataset(dataDir,
                iters=None,prefix=None,read_grid=True,
                chunks=localChunks,geometry=geometry)

        if x:
            self.ds = self.ds.sel(XC=slice(x[0],x[1]),
                                  XG=slice(x[0],x[1]))

            if self.ds.XG.shape[0] > self.ds.XC.shape[0]:
                self.ds = self.ds.sel(XG=slice(x[0],self.ds.XC[-1]))

        if y:
            self.ds = self.ds.sel(YC=slice(y[0],y[1]),
                                  YG=slice(y[0],y[1]))

            # Some indexing by slicing returns YG values
            # surrounding YC values on south and north boundaries
            # This patch clips the last YG value
            if self.ds.YG.shape[0] > self.ds.YC.shape[0]:
                self.ds = self.ds.sel(YG=slice(y[0],self.ds.YC[-1]))

       
        # Create a grid object
        #self.grid = xgcm.Grid(self.ds)

        # Copy data to numpy arrays (for numba acceleration)
        self.dxc = self.ds.dxC.to_numpy().astype(np.float32)
        self.dyc = self.ds.dyC.to_numpy().astype(np.float32)
        self.dxg = self.ds.dxG.to_numpy().astype(np.float32)
        self.dyg = self.ds.dyG.to_numpy().astype(np.float32)
        self.raz = self.ds.rAz.to_numpy().astype(np.float32)

        zd = -abs(depth) #ensure that depth is negative value
        self.hFacC = self.ds.hFacC.interp(Z=[zd],method="nearest").squeeze().to_numpy().astype(np.float32)
        ny, nx = self.hFacC.shape
        self.mask = np.ones( self.hFacC.shape )
        # The mask applies to vorticity points on the
        # arakawa c-grid (z-points below).
        #
        # We have chosen to completely surround
        # all tracer points in the domain with vorticity points;
        # This means that valid tracer points are in the 
        # range (0:nx-2,0:ny-2)
        #
        # Vorticity points are in the range (0:nx-1,0:ny-1)
        # "Ghost points" for the vorticty points are imposed at
        # (0,:), (nx-1,:), (:,0), (:,ny-1)
        #
        #
        # Additionally, the grid metrics need to be of size (nx+1,ny+1)
        #
        # When tracer point t(i,j) is "dry", the four surrounding
        # vorticity points needs to be marked dry
        #
        #   z(i,j+1) ----- z(i+1,j+1)
        #     |                 |
        #     |                 |
        #     |                 |
        #     |      t(i,j)     |
        #     |                 |
        #     |                 |
        #     |                 |
        #   z(i,j) -------- z(i+1,j)
        #
        # A mask value of 0 corresponds to a wet cell (this cell is not masked)
        # A mask value of 1 corresponds to a dry cell (this cell is masked)
        # This helps with working with numpy's masked arrays
        #
        #
        
        self.mask[:,0] = 0.0
        self.mask[:,nx-1] = 0.0
        self.mask[0,:] = 0.0
        self.mask[ny-1,:] = 0.0
        for j in range(0,ny):
            for i in range(0,nx):
                if self.hFacC[j,i] == 0.0:

                    self.mask[j,i] = 0.0
                    self.mask[j,i+1] = 0.0
                    self.mask[j+1,i] = 0.0
                    self.mask[j+1,i+1] = 0.0

        wetcells = self.mask
        self.ndof = wetcells.sum().astype(int)

    def LapZInv_JacobiSolve(self, s0, b, itermax=1000, tolerance=1e-4):
        """Performs Jacobi iterations to iteratively solve L s = b,
        where `L s` is the Laplacian on vorticity points applied to s"""
        import numpy as np
        import xnma.kernels as kernels

        sk = s0
        r0 = kernels.LapZ_Residual(sk, b, 
                self.mask, self.dxc, self.dyc, 
                self.dxg, self.dyg, self.raz )

        for k in range(0,itermax):
            q = b - kernels.LapZ_JacobiLU( sk, self.dxc, 
                    self.dyc, self.dxg, self.dyg, self.raz )

            sk = kernels.LapZ_JacobiDinv( q, self.dxc, 
                    self.dyc, self.dxg, self.dyg, self.raz )

            r = kernels.LapZ_Residual(sk, b, 
                    self.mask, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.raz )

            if( r/r0 < tolerance ):
                break

        return sk

