#!/usr/bin/env python
#

# Notes
#
# Masking
# -------
#   The mask applies to vorticity points on the
#   arakawa c-grid (z-points below).
#
#   We have chosen to completely surround
#   all tracer points in the domain with vorticity points;
#   This means that valid tracer points are in the 
#   range (0:nx-2,0:ny-2)
#
#   Vorticity points are in the range (0:nx-1,0:ny-1)
#   "Ghost points" for the vorticty points are imposed at
#   (0,:), (nx-1,:), (:,0), (:,ny-1)
#
#
#   Additionally, the grid metrics need to be of size (nx+1,ny+1)
#
#   When tracer point t(i,j) is "dry", the four surrounding
#   vorticity points needs to be marked dry
#
#     z(i,j+1) ----- z(i+1,j+1)
#       |                 |
#       |                 |
#       |                 |
#       |      t(i,j)     |
#       |                 |
#       |                 |
#       |                 |
#     z(i,j) -------- z(i+1,j)
#
#   A mask value of 0 corresponds to a wet cell (this cell is not masked)
#   A mask value of 1 corresponds to a dry cell (this cell is masked)
#   This helps with working with numpy's masked arrays
#
#

class model:
    def __init__(self):

        self.ds = None
        self.grid = None

        self.ndof = 0
        self.mask = None

        self.eigenmodes = None # Defined on vorticicity points
        self.eigenvalues = None #

    def construct(self,dx=1.0,dy=1.0,nx=500,ny=500):
        """
        Constructs a quadrilateral domain with uniform grid
        spacing. 

        !!! warning
            This method currently does not populate ds and 
            grid. Instead, xc, xg, yc, yg are stored as 1d
            numpy arrays
        """
        import numpy as np
        from numpy import ma

        self.dxc = np.ones((ny,nx)).astype(np.float32)*dx
        self.dxg = np.ones((ny,nx)).astype(np.float32)*dx

        self.dyc = np.ones((ny,nx)).astype(np.float32)*dy
        self.dyg = np.ones((ny,nx)).astype(np.float32)*dy

        self.raz = np.ones((ny,nx)).astype(np.float32)*dx*dy

        dxl = np.ones((nx)).astype(np.float32)*dx
        self.xg = dxl.cumsum() - dx 
        self.xc = self.xg + dx*0.5

        dyl = np.ones((ny)).astype(np.float32)*dy
        self.yg = dyl.cumsum() - dy 
        self.yc = self.yg + dy*0.5

        self.mask = np.ones((ny,nx))
        self.mask[:,0] = 0.0
        self.mask[:,nx-1] = 0.0
        self.mask[0,:] = 0.0
        self.mask[ny-1,:] = 0.0
        wetcells = self.mask
        self.ndof = wetcells.sum().astype(int)

        # Create template masked arrays
        self.b = ma.masked_array( np.zeros((ny,nx)), mask=abs(self.mask - 1.0), dtype=np.float32 )
        self.s = ma.masked_array( np.zeros((ny,nx)), mask=abs(self.mask - 1.0), dtype=np.float32 )


    def circularDemo(self,dx=1.0,dy=1.0,nx=500,ny=500):
        """
        Constructs a quadrilateral domain with uniform grid
        spacing. An additional mask is applied to create
        a domain with circular geometry.

        !!! warning
            This method currently does not populate ds and 
            grid. Instead, xc, xg, yc, yg are stored as 1d
            numpy arrays
        """
        import numpy as np
        from numpy import ma

        self.dxc = np.ones((ny,nx)).astype(np.float32)*dx
        self.dxg = np.ones((ny,nx)).astype(np.float32)*dx

        self.dyc = np.ones((ny,nx)).astype(np.float32)*dy
        self.dyg = np.ones((ny,nx)).astype(np.float32)*dy

        self.raz = np.ones((ny,nx)).astype(np.float32)*dx*dy

        dxl = np.ones((nx)).astype(np.float32)*dx
        self.xg = dxl.cumsum() - dx 
        self.xc = self.xg + dx*0.5

        dyl = np.ones((ny)).astype(np.float32)*dy
        self.yg = dyl.cumsum() - dy 
        self.yc = self.yg + dy*0.5

        self.mask = np.ones((ny,nx))
        self.mask[:,0] = 0.0
        self.mask[:,nx-1] = 0.0
        self.mask[0,:] = 0.0
        self.mask[ny-1,:] = 0.0

        xc = self.xg[-1]*0.5
        yc = self.yg[-1]*0.5
        for j in range(0,ny):
            y = self.yg[j]
            for i in range(0,nx):
                x = self.xg[i]
                r = np.sqrt( (x-xc)**2 + (y-yc)**2 )

                if r >= 0.9*xc :
                    self.mask[j,i] = 0.0


        self.ndof = self.mask.sum().astype(int)


    def loadGrid(self, dataDir, chunks=None, depth=0, x=None, y=None, iters=None, geometry="sphericalpolar"):
        """Loads in grid from MITgcm metadata files in dataDir
        and configures masks at given depth"""
        import numpy as np
        import numpy.ma as ma
        import xmitgcm
        import xgcm
        from dask import array as da

        localChunks = None

        self.depth = depth

        if chunks:
            localChunks = {'XC':chunks[0],
                           'XG':chunks[0],
                           'YC':chunks[1],
                           'YG':chunks[1]}

        self.ds = xmitgcm.open_mdsdataset(dataDir,
                iters=iters,prefix=['U','V'],read_grid=True,
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


    def LapZInv_PCCG(self, b, s0=None, pcitermax=20, pctolerance=1e-2, itermax=1500, tolerance=1e-4):
        """Uses preconditioned conjugate gradient to solve L s = b,
        where `L s` is the Laplacian on vorticity points applied to s
        Stopping criteria is when the relative change in the solution is
        less than the tolerance.

        The preconditioner is the LapZInv_JacobianSolve method.

        Algorithm taken from pg.51 of 
        https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

        """
        import numpy as np
        import xnma.kernels as kernels

        if s0:
           sk = s0
        else:
           sk = np.zeros_like( b ) 

        sk = sk*self.mask 

        r = kernels.LapZ_Residual(sk, b, 
                self.mask, self.dxc, self.dyc, 
                self.dxg, self.dyg, self.raz )

        d = self.LapZInv_JacobiSolve( r, 
                itermax=pcitermax, tolerance=pctolerance )

        delta = np.sum(r*d)
        rmag = np.max(abs(r))
        r0 = rmag

        for k in range(0,itermax):

            q = kernels.LapZ(d, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.raz )*self.mask

            alpha = delta/(np.sum(d*q))

            sk += alpha*d
            if k % 50 == 0:
                r = kernels.LapZ_Residual(sk, b, 
                        self.mask, self.dxc, self.dyc, 
                        self.dxg, self.dyg, self.raz )
            else:
                r -= alpha*q

            x = self.LapZInv_JacobiSolve( r, 
                    itermax=pcitermax, tolerance=pctolerance )
        
            rmag = np.max(abs(r))
            deltaOld = delta 
            delta = np.sum(r*x)
            beta = delta/deltaOld
            d = x + beta*d
            if rmag/r0 <= tolerance:
                break

        if rmag/r0 > tolerance:
           print(f"Conjugate gradient method did not converge in {k+1} iterations : {delta}")

        return sk



    def LapZInv_JacobiSolve(self, b, s0=None, itermax=1000, tolerance=1e-4):
        """Performs Jacobi iterations to iteratively solve L s = b,
        where `L s` is the Laplacian on vorticity points applied to s
        Stopping criteria is when the relative change in the solution is
        less than the tolerance.

        !!! warning
            This tolerance is invalid when max(abs(b)) == max(abs(s)) = 0
        """
        import numpy as np
        import xnma.kernels as kernels

        if s0:
           sk = s0
        else:
           sk = np.zeros_like( b ) 

        sk = sk*self.mask 
        r = kernels.LapZ_Residual(sk, b, 
                self.mask, self.dxc, self.dyc, 
                self.dxg, self.dyg, self.raz )

        for k in range(0,itermax):

            ds = kernels.LapZ_JacobiDinv( r, self.dxc, 
                    self.dyc, self.dxg, self.dyg, self.raz )

            dsmag = np.max(abs(ds))
            smag = np.max(abs(sk))
            if smag < np.finfo(np.float32).eps:
                smag == np.max(abs(b))

            if( dsmag/smag <= tolerance ):
                #print(f"Jacobi method converged in {k} iterations : {dsmag/smag}")
                break

            # Update the solution
            sk += ds

            r = kernels.LapZ_Residual(sk, b, 
                    self.mask, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.raz )


        return sk

    def laplacian(self, x):
        """ Wrapper for the Laplacian, where x comes in as a flat 1-D array
            only at `wet` grid cell locations """

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels

        # x comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        xgrid = ma.masked_array( np.zeros(self.mask.shape), 
                mask = abs(self.mask - 1.0), dtype=np.float32 )
        xgrid[~xgrid.mask] = x # Set interior values to b

        # Invert the laplacian
        Lx = kernels.LapZ(x, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.raz )

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array( Lx, mask = abs(self.mask - 1.0), 
                dtype=np.float32 ).compressed()

    def laplacianInverse(self, b):
        """ Wrapper for the Laplacian Inverse (with preconditioned conjugate gradient),
            where b comes in as a flat 1-D array only at `wet` grid cell locations """

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels
        import time

        # b comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        # Use the model.b attribute to push the DOF formatted data

        bgrid = ma.masked_array( np.zeros(self.mask.shape),
                mask = abs(self.mask - 1.0), dtype=np.float32 )
        bgrid[~bgrid.mask] = b # Set interior values to b

        x = np.ones(self.mask.shape,dtype=np.float32)
        # Invert the laplacian
        tic = time.perf_counter()
        x = self.LapZInv_PCCG(bgrid.data, s0=None, 
                pcitermax=20, pctolerance=1e-2, itermax=1500,
                tolerance=1e-12)
        toc = time.perf_counter()
        runtime = toc-tic
        print(f"Laplacian inverse runtime : {runtime:0.4f} s")

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array( x, mask = abs(self.mask - 1.0), 
                dtype=np.float32 ).compressed()

    def findEigenmodes( self, nmodes = 10, tolerance=0):
        """ Finds the eigenmodes using sci-py `eigsh`.

        Parameters

        nmodes - the number of eigenmodes you wish to find
        sigma  - Eigenmodes with eigenvalues near sigma are returned
        which  - Identical to the scipy `which` argument

        See scipy/eigsh docs for more details

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html

        """
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import eigsh
        import time
        import numpy as np

        shape = (self.ndof,self.ndof)
    
#        L = LinearOperator(shape, matvec=lambda x: self.laplacian(x),dtype=np.float32)
        Linv = LinearOperator(shape, matvec=lambda b: self.laplacianInverse(b),dtype=np.float32)

        print("Starting eigenvalue search")
        tic = time.perf_counter()
        evals, evecs = eigsh(Linv, 
                             k=nmodes, 
                             tol=tolerance,
                             return_eigenvectors=True) 
        toc = time.perf_counter()
        runtime = toc-tic
        print(f"eigsh runtime : {runtime:0.4f} s")

        self.eigenvalues = evals
        self.eigenmodes = evecs

        #for k in range(0,nmodes):






