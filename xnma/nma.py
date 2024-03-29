#!/usr/bin/env python
#

# Notes
#
# Masking
# -------
#   The mask applies to vorticity points on the
#   arakawa c-grid (z-points below).
#
#
#   Vorticity points are in the range (0:nx-1,0:ny-1)
#   "Ghost points" for the vorticty points are imposed at
#   (0,:), (nx-1,:), (:,0), (:,ny-1)
#   
#   We have chosen to completely surround
#   all tracer points in the domain with vorticity points;
#   This means that valid tracer points are in the 
#   range (0:nx-2,0:ny-2)
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

zeroTol = 1e-6

class model:
    def __init__(self):

        self.ds = None
        self.grid = None

        self.ndofZ = 0
        self.ndofC = 0
        self.maskZ = None
        self.maskC = None

        self.eigenmodes = None # Defined on tracer points
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
        self.rac = np.ones((ny,nx)).astype(np.float32)*dx*dy

        dxl = np.ones((nx)).astype(np.float32)*dx
        self.xg = dxl.cumsum() - 2*dx 
        self.xc = self.xg + dx*0.5

        dyl = np.ones((ny)).astype(np.float32)*dy
        self.yg = dyl.cumsum() - 2*dy 
        self.yc = self.yg + dy*0.5

        # The stencil operations we use via numba
        # enforce masking on the bounding quad boundaries
        # Because of this, we set the maskC to zero on the
        # boundaries
        self.maskC = np.ones((ny,nx))
        self.maskC[:,0] = 0.0
        self.maskC[:,nx-1] = 0.0
        self.maskC[:,nx-2] = 0.0
        self.maskC[0,:] = 0.0
        self.maskC[ny-1,:] = 0.0
        self.maskC[ny-2,:] = 0.0
        self.ndofC = self.maskC.sum().astype(int)
        self.hFacC = self.maskC

        # A consistent z-mask is set, assuming
        # z[j,i] is at the southwest corner of
        # c[j,i]
        self.maskZ = np.ones((ny,nx))
        self.hFacW = np.ones((ny,nx))
        self.hFacS = np.ones((ny,nx))
        self.maskW = np.ones((ny,nx))
        self.maskS = np.ones((ny,nx))
        for j in range(0,ny-1):
            for i in range(0,nx-1):
                if self.maskC[j,i] == 0.0:

                    self.maskZ[j,i] = 0.0
                    self.maskZ[j,i+1] = 0.0
                    self.maskZ[j+1,i] = 0.0
                    self.maskZ[j+1,i+1] = 0.0
                    
                    self.maskW[j,i] = 0.0
                    self.maskW[j,i+1] = 0.0
                    self.maskS[j,i] = 0.0
                    self.maskS[j+1,i] = 0.0
                    
        self.ndofZ = self.maskZ.sum().astype(int)
        
        print("Construction report")
        print(f"nDOF (C) : {self.ndofC}")
        print(f"nDOF (Z) : {self.ndofZ}")


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
        self.rac = np.ones((ny,nx)).astype(np.float32)*dx*dy

        dxl = np.ones((nx)).astype(np.float32)*dx
        self.xg = dxl.cumsum() - dx 
        self.xc = self.xg + dx*0.5

        dyl = np.ones((ny)).astype(np.float32)*dy
        self.yg = dyl.cumsum() - dy 
        self.yc = self.yg + dy*0.5

        self.maskZ = np.ones((ny,nx))
        self.maskZ[:,0] = 0.0
        self.maskZ[:,nx-1] = 0.0
        self.maskZ[0,:] = 0.0
        self.maskZ[ny-1,:] = 0.0

        xc = self.xg[-1]*0.5
        yc = self.yg[-1]*0.5
        for j in range(0,ny):
            y = self.yg[j]
            for i in range(0,nx):
                x = self.xg[i]
                r = np.sqrt( (x-xc)**2 + (y-yc)**2 )

                if r >= 0.9*xc :
                    self.maskZ[j,i] = 0.0


        self.ndofZ = self.maskZ.sum().astype(int)

    def irregularHolesDemo(self,dx=1.0,dy=1.0,nx=500,ny=500):
        """
        Constructs a quadrilateral domain with uniform grid
        spacing. An additional mask is applied to create
        a domain with irregular boundaries and oddly placed
        holes.

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
        self.rac = np.ones((ny,nx)).astype(np.float32)*dx*dy

        dxl = np.ones((nx)).astype(np.float32)*dx
        self.xg = dxl.cumsum() - dx 
        self.xc = self.xg + dx*0.5

        dyl = np.ones((ny)).astype(np.float32)*dy
        self.yg = dyl.cumsum() - dy 
        self.yc = self.yg + dy*0.5

        self.maskZ = np.ones((ny,nx))
        self.maskZ[:,0] = 0.0
        self.maskZ[:,nx-1] = 0.0
        self.maskZ[0,:] = 0.0
        self.maskZ[ny-1,:] = 0.0

        xc = self.xg[-1]*0.5
        yc = self.yg[-1]*0.5
        for j in range(0,ny):
            y = self.yg[j]
            for i in range(0,nx):
                x = self.xg[i]

                # Place a hole to the north-east with 10% domain width radius 
                r = np.sqrt( (x-1.7*xc)**2 + (y-1.7*yc)**2 )
                if r <= 0.1*xc :
                    self.maskZ[j,i] = 0.0

                # Place a hole to the southwest with 15% domain width radius
                r = np.sqrt( (x-0.2*xc)**2 + (y-0.3*yc)**2 )
                if r <= 0.15*xc :
                    self.maskZ[j,i] = 0.0

                # Place a hole to the south-east with 40% domain width radius 
                r = np.sqrt( (x-1.3*xc)**2 + (y-0.4*yc)**2 )
                if r <= 0.4*xc :
                    self.maskZ[j,i] = 0.0

                r = np.sqrt( (x)**2 + (y-2.0*yc)**2 )
                f = np.exp( -r/(0.2*xc*xc) )
                if f > 0.85 :
                    self.maskZ[j,i] = 0.0


        self.ndofZ = self.maskZ.sum().astype(int)

    def load(self, dataDir, chunks=None, depth=0, x=None, y=None, iters=None, geometry="sphericalpolar"):
        """Loads in grid and velocity field from MITgcm metadata files in dataDir
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
        self.grid = xgcm.Grid(self.ds)

        # Copy data to numpy arrays (for numba acceleration)
        self.xc = self.ds.XC.to_numpy().astype(np.float32)
        self.yc = self.ds.YC.to_numpy().astype(np.float32)
        self.xg = self.ds.XG.to_numpy().astype(np.float32)
        self.yg = self.ds.YG.to_numpy().astype(np.float32)
        self.dxc = self.ds.dxC.to_numpy().astype(np.float32)
        self.dyc = self.ds.dyC.to_numpy().astype(np.float32)
        self.dxg = self.ds.dxG.to_numpy().astype(np.float32)
        self.dyg = self.ds.dyG.to_numpy().astype(np.float32)
        self.raz = self.ds.rAz.to_numpy().astype(np.float32)
        self.rac = self.ds.rA.to_numpy().astype(np.float32)


        zd = -abs(depth) #ensure that depth is negative value
        self.U = self.ds.U.interp(Z=[zd],method="nearest")
        self.V = self.ds.V.interp(Z=[zd],method="nearest")

        self.hFacC = self.ds.hFacC.interp(Z=[zd],method="nearest").squeeze().to_numpy().astype(np.float32)
        self.hFacW = self.ds.hFacW.interp(Z=[zd],method="nearest").squeeze().to_numpy().astype(np.float32)
        self.hFacS = self.ds.hFacS.interp(Z=[zd],method="nearest").squeeze().to_numpy().astype(np.float32)

        ny,nx = self.hFacC.shape

        self.maskC = np.ceil(self.hFacC)
        self.maskW = np.ceil(self.hFacW)
        self.maskS = np.ceil(self.hFacS)
        
        # The stencil operations we use via numba
        # enforce masking on the bounding quad boundaries
        # Because of this, we set the maskC to zero on the
        # boundaries
        self.maskC = np.ones((ny,nx))
        self.maskC[:,0] = 0.0
        self.maskC[:,nx-1] = 0.0
        self.maskC[:,nx-2] = 0.0
        self.maskC[0,:] = 0.0
        self.maskC[ny-1,:] = 0.0
        self.maskC[ny-2,:] = 0.0
        self.ndofC = self.maskC.sum().astype(int)
        self.hFacC = self.maskC

        # A consistent z-mask is set, assuming
        # z[j,i] is at the southwest corner of
        # c[j,i]
        self.maskZ = np.ones((ny,nx))
        self.hFacW = np.ones((ny,nx))
        self.hFacS = np.ones((ny,nx))
        self.maskW = np.ones((ny,nx))
        self.maskS = np.ones((ny,nx))
        for j in range(0,ny-1):
            for i in range(0,nx-1):
                if self.maskC[j,i] == 0.0:

                    self.maskZ[j,i] = 0.0
                    self.maskZ[j,i+1] = 0.0
                    self.maskZ[j+1,i] = 0.0
                    self.maskZ[j+1,i+1] = 0.0
                    self.maskW[j,i] = 0.0
                    self.maskW[j,i+1] = 0.0
                    self.maskS[j,i] = 0.0
                    self.maskS[j+1,i] = 0.0

        self.ndofZ = self.maskZ.sum().astype(int)
        self.ndofC = self.maskC.sum().astype(int)


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

        sk = sk*self.maskZ

        r = kernels.LapZ_Residual(sk, b, 
                self.maskZ, self.dxc, self.dyc, 
                self.dxg, self.dyg, self.raz )

        d = self.LapZInv_JacobiSolve( r, 
                itermax=pcitermax, tolerance=pctolerance )

        delta = np.sum(r*d)
        rmag = np.max(abs(r))
        r0 = rmag

        for k in range(0,itermax):

            q = kernels.LapZ(d, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.raz )*self.maskZ

            alpha = delta/(np.sum(d*q))

            sk += alpha*d
            if k % 50 == 0:
                r = kernels.LapZ_Residual(sk, b, 
                        self.maskZ, self.dxc, self.dyc, 
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
        where `L s` is the Laplacian on vorticity points with homogeneous Dirichlet
        boundary conditions applied to s
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

        sk = sk*self.maskZ
        r = kernels.LapZ_Residual(sk, b, 
                self.maskZ, self.dxc, self.dyc, 
                self.dxg, self.dyg, self.raz )

        for k in range(0,itermax):

            ds = kernels.LapZ_JacobiDinv( r, self.dxc, 
                    self.dyc, self.dxg, self.dyg, self.raz )

            dsmag = np.max(abs(ds))
            smag = np.max(abs(sk))
            if smag <= np.finfo(np.float32).eps:
                smag == np.max(abs(b))

            if smag > np.finfo(np.float32).eps:
                if( dsmag/smag <= tolerance ):
                    break

            # Update the solution
            sk += ds

            r = kernels.LapZ_Residual(sk, b, 
                    self.maskZ, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.raz )


        return sk

    def LapCInv_PCCG(self, b, s0=None, pcitermax=0, pctolerance=1e-2, itermax=1500, tolerance=1e-4, dShift = 1e-2):
        """Uses preconditioned conjugate gradient to solve L s = b,
        where `L s` is the Laplacian on tracer points applied to s
        Stopping criteria is when the relative change in the solution is
        less than the tolerance.

        The preconditioner is the LapCInv_JacobianSolve method.

        Algorithm taken from pg.51 of 
        https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

        """
        import numpy as np
        import xnma.kernels as kernels

       # if pcitermax > 0:
       #     print("Warning : Jacobi preconditioner unverified for homogeneous neumann modes")

        if s0:
           sk = s0
        else:
           sk = np.zeros_like( b ) 

        sk = sk*self.maskC 

        r = kernels.LapC_Residual(sk, b, 
                self.maskC, self.dxc, self.dyc, 
                self.dxg, self.dyg, self.maskW, 
                self.maskS,self.rac, dShift )

        if pcitermax == 0:
          d = r*self.maskC
        else:
          d = self.LapCInv_JacobiSolve( r, 
                  itermax=pcitermax, tolerance=pctolerance, dShift = dShift )

        delta = np.sum(r*d)
        rmag = np.max(abs(r))
        r0 = rmag

        for k in range(0,itermax):

            #print(f"(k,r) : ({k},{rmag})")
            q = kernels.LapC(d, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.maskW, 
                    self.maskS, self.rac, dShift )*self.maskC

            alpha = delta/(np.sum(d*q))

            sk += alpha*d
            if k % 50 == 0:
                r = kernels.LapC_Residual(sk, b, 
                        self.maskC, self.dxc, self.dyc, 
                        self.dxg, self.dyg, self.maskW, 
                        self.maskS,self.rac, dShift )
            else:
                r -= alpha*q

            if pcitermax == 0:
              x = r*self.maskC
            else:
              x = self.LapCInv_JacobiSolve( r, 
                      itermax=pcitermax, tolerance=pctolerance, dShift = dShift )
        
            rmag = np.max(abs(r))
            deltaOld = delta 
            delta = np.sum(r*x)
            beta = delta/deltaOld
            d = x + beta*d
            if rmag/r0 <= tolerance:
                break

        if rmag/r0 > tolerance:
           print(f"Conjugate gradient method did not converge in {k+1} iterations : {rmag}")

        return sk


    def LapCInv_JacobiSolve(self, b, s0=None, itermax=1000, tolerance=1e-4, dShift = 1e-2):
        """Performs Jacobi iterations to iteratively solve L s = b,
        where `L s` is the Laplacian on tracer points applied to s
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

        sk = sk*self.maskC
        r = kernels.LapC_Residual(sk, b, 
                self.maskC, self.dxc, self.dyc, 
                self.dxg, self.dyg, self.maskW,
                self.maskS, self.rac, dShift )

        for k in range(0,itermax):

            ds = kernels.LapC_JacobiDinv( r, self.dxc, 
                    self.dyc, self.dxg, self.dyg, self.maskW,
                    self.maskS, self.rac, dShift )

            dsmag = np.max(abs(ds))
            smag = np.max(abs(sk))
            if smag <= np.finfo(np.float32).eps:
                smag == np.max(abs(b))

            if smag > np.finfo(np.float32).eps:
                if( dsmag/smag <= tolerance ):
                    break

            # Update the solution
            sk += ds

            r = kernels.LapC_Residual(sk, b, 
                    self.maskC, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.maskW,
                    self.maskS, self.rac, dShift )

        return sk

    def laplacianZ(self, x, dShift):
        """ Wrapper for the Laplacian, where x comes in as a flat 1-D array
            only at `wet` grid cell locations """

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels

        # x comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        xgrid = ma.masked_array( np.zeros(self.maskZ.shape), 
                mask = abs(self.maskZ - 1.0), dtype=np.float32 )
        xgrid[~xgrid.mask] = x # Set interior values to b

        # Invert the laplacian
        Lx = kernels.LapZ(x, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.raz )

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array( Lx, mask = abs(self.maskZ - 1.0), 
                dtype=np.float32 ).compressed()

    def laplacianZInverse(self, b, dShift):
        """ Wrapper for the Laplacian Inverse (with preconditioned conjugate gradient),
            where b comes in as a flat 1-D array only at `wet` grid cell locations """

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels
        #import time

        # b comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        # Use the model.b attribute to push the DOF formatted data

        bgrid = ma.masked_array( np.zeros(self.maskZ.shape),
                mask = abs(self.maskZ - 1.0), dtype=np.float32 )
        bgrid[~bgrid.mask] = b # Set interior values to b

        x = np.ones(self.maskZ.shape,dtype=np.float32)
        # Invert the laplacian
        x = self.LapZInv_PCCG(bgrid.data, s0=None, 
                pcitermax=20, pctolerance=1e-2, itermax=1500,
                tolerance=1e-12)

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array( x, mask = abs(self.maskZ - 1.0), 
                dtype=np.float32 ).compressed()

    def laplacianC(self, x, dShift):
        """ Wrapper for the Laplacian, where x comes in as a flat 1-D array
            only at `wet` grid cell locations """

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels
        
        # x comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        xgrid = ma.masked_array( np.zeros(self.maskZ.shape), 
                mask = abs(self.maskC - 1.0), dtype=np.float32 )
        xgrid[~xgrid.mask] = x # Set interior values to b

        # Invert the laplacian
        Lx = kernels.LapC(x, self.dxc, self.dyc, 
                    self.dxg, self.dyg, self.maskW,
                    self.maskS, self.raC, dShift )

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array( Lx, mask = abs(self.maskC - 1.0), 
                dtype=np.float32 ).compressed()

    def laplacianCInverse(self, b, dShift):
        """ Wrapper for the Laplacian Inverse (with preconditioned conjugate gradient),
            where b comes in as a flat 1-D array only at `wet` grid cell locations """

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels
        import time
        
        # b comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        # Use the model.b attribute to push the DOF formatted data

        bgrid = ma.masked_array( np.zeros(self.maskC.shape),
                mask = abs(self.maskC - 1.0), dtype=np.float32 )
        bgrid[~bgrid.mask] = b # Set interior values to b

        x = np.ones(self.maskZ.shape,dtype=np.float32)
        # Invert the laplacian
        x = self.LapCInv_PCCG(bgrid.data, s0=None, 
                pcitermax=20, pctolerance=1e-2, itermax=1500,
                tolerance=1e-12, dShift = dShift)

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array( x, mask = abs(self.maskC - 1.0), 
                dtype=np.float32 ).compressed()

    def findEigenmodes( self, nmodes = 10, tolerance=0, deShift = 0, neShift = 1e-2):
        """ Finds the eigenmodes using sci-py `eigsh`.

        Parameters

        nmodes - the number of eigenmodes you wish to find
        sigma  - Eigenmodes with eigenvalues near sigma are returned
        which  - Identical to the scipy `which` argument
        deShift - factor to shift the diagonal by for the dirichlet mode operator
        neShift - factor to shift the diagonal by for the neumann mode operator

        See scipy/eigsh docs for more details

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html

        """
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import eigsh
        import time
        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels

        shape = (self.ndofZ,self.ndofZ)
        LZinv = LinearOperator(shape, matvec=lambda b: self.laplacianZInverse(b, deShift),dtype=np.float32)

        shape = (self.ndofC,self.ndofC)
        LCinv = LinearOperator(shape, matvec=lambda b: self.laplacianCInverse(b, neShift),dtype=np.float32)

        print("[Dirichlet modes] Starting eigenvalue search")
        tic = time.perf_counter()
        evals_d, evecs_d = eigsh(LZinv, 
                             k=nmodes, 
                             tol=tolerance,
                             return_eigenvectors=True) 
        toc = time.perf_counter()
        runtime = toc-tic
        print(f"[Dirichlet modes] eigsh runtime : {runtime:0.4f} s")
       
        sgridZ = ma.masked_array( np.zeros(self.maskZ.shape), mask=abs(self.maskZ - 1.0), dtype=np.float32 )
        sgrid = ma.masked_array( np.zeros(self.maskZ.shape), mask=abs(self.maskC - 1.0), dtype=np.float32 )
        ny, nx = self.maskZ.shape
        eigenvalues = np.zeros( (nmodes), dtype=np.float32 )
        eigenmodes = np.zeros( (nmodes,ny,nx), dtype=np.float32 )
        
        for k in range(0,nmodes):
            ev = 1.0/evals_d[k] + deShift
            if np.abs(ev-deShift) < np.abs(zeroTol):
               eigenvalues[k] = 0.0
            else:
               eigenvalues[k] = ev
               
            # Interpolate the dirichlet modes from the vorticity points
            # to the tracer points and store the result in sgrid
            sgridZ[~sgridZ.mask] = evecs_d[:,k]
            f = sgridZ.data*self.maskZ          
            g = kernels.vorticityToTracer(f)*self.maskC
            
            # Normalize so that the area integral is 1
            mag = np.sqrt(np.sum(g*g*self.rac))
            eigenmodes[k,:,:] = g/mag
        
        self.d_eigenvalues = eigenvalues
        self.d_eigenmodes = eigenmodes

        print("[Neumann modes] Starting eigenvalue search")
        tic = time.perf_counter()
        evals_n, evecs_n = eigsh(LCinv,
                             k=nmodes,
                             tol=tolerance,
                             return_eigenvectors=True) 
        toc = time.perf_counter()
        runtime = toc-tic
        print(f"[Neumann modes] eigsh runtime : {runtime:0.4f} s")
        
        eigenvalues = np.zeros( (nmodes), dtype=np.float32 )
        eigenmodes = np.zeros( (nmodes,ny,nx), dtype=np.float32 )
        for k in range(0,nmodes):
            ev = 1.0/evals_n[k] + neShift
            if np.abs(ev-neShift) < np.abs(zeroTol):
               eigenvalues[k] = 0.0
            else:
               eigenvalues[k] = ev
               
            sgrid[~sgrid.mask] = evecs_n[:,k]
            g = sgrid.data*self.maskC
            mag = np.sqrt(np.sum(g*g*self.rac))
            eigenmodes[k,:,:] = g/mag
        
        self.n_eigenvalues = eigenvalues
        self.n_eigenmodes = eigenmodes
        
        # Sort the eigenvalues
        #ind = np.argsort(eigenvalues)      
        #self.eigenvalues = eigenvalues[ind]
        #self.eigenmodes = eigenmodes[ind,:,:]
                
        
    def vectorProjection( self, u, v ):
        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels
        # if the eigenmodes have not been found
        # find them, using the default parameters
       # if (self.eigenmodes is None) :
       #     self.findEigenmodes()
            
        print("Calculating projection of u,v")
        
        # Calculate the divergence
        divergence = kernels.divergence( u, v, self.dxg, 
                                  self.dyg, self.hFacW,
                                  self.hFacS, self.rac  )*self.maskC
        
        # Calculate the vorticity
        curlU = kernels.vorticity( u, v, self.dxc,
                                  self.dyc, self.raz )
       
        vorticity = kernels.vorticityToTracer(curlU)*self.maskC
        
        nmodes = self.d_eigenvalues.shape[0]
        ny, nx = u.shape
        f_d = np.zeros( (nmodes,2), dtype=np.float32 )
        f_v = np.zeros( (nmodes,2), dtype=np.float32 )
        
        for k in range(0,nmodes):
           # Calculate the projection coefficients
           f_d[k,0] = np.sum(divergence*np.squeeze(self.d_eigenmodes[k,:,:])*self.rac) # dirichlet projection (\alpha_n * \tau_n)
           f_v[k,0] = np.sum(vorticity*np.squeeze(self.d_eigenmodes[k,:,:])*self.rac) # dirichlet projection (\beta_n * \tau_n)
           f_d[k,1] = np.sum(divergence*np.squeeze(self.n_eigenmodes[k,:,:])*self.rac) # neumann projection (\alpha_n * \tau_n)
           f_v[k,1] = np.sum(vorticity*np.squeeze(self.n_eigenmodes[k,:,:])*self.rac) # neumann projection (\beta_n * \tau_n)
       
        # Calculate the orthogonality matrix
        #q = np.zeros( (nmodes,nmodes), dtype=np.float32 )

        #for row in range(0,nmodes):
        #    g = np.squeeze(self.eigenmodes[row,:,:])
        #    for col in range(0,nmodes):
        #        f = np.squeeze(self.eigenmodes[col,:,:])
        #        proj = np.sum(f*g*self.rac)
        #        q[row,col] = proj
                
        # Invert the orthogonality matrix
        #qInv = np.linalg.inv(q)
        #d_m = np.matmul(qInv,f_d)
        #v_m = np.matmul(qInv,f_v)
          
        proj_d = np.zeros( (ny,nx,2), dtype=np.float32 )
        proj_v = np.zeros( (ny,nx,2), dtype=np.float32 )
        alpha = np.zeros( (nmodes,2), dtype=np.float32 )
        beta = np.zeros( (nmodes,2), dtype=np.float32 )
        for k in range(0,nmodes):
          # if self.d_eigenvalues[k] != 0.0:
             #d_m[k] = d_m[k]/self.eigenvalues[k]
             #v_m[k] = v_m[k]/self.eigenvalues[k]
             # Calculate the projection
           proj_d[:,:,0] += f_d[k,0]*np.squeeze(self.d_eigenmodes[k,:,:])
           proj_v[:,:,0] += f_v[k,0]*np.squeeze(self.d_eigenmodes[k,:,:])
           proj_d[:,:,1] += f_d[k,1]*np.squeeze(self.n_eigenmodes[k,:,:])
           proj_v[:,:,1] += f_v[k,1]*np.squeeze(self.n_eigenmodes[k,:,:])
           
           if np.abs(self.d_eigenvalues[k]) > zeroTol: 
              alpha[k,0] = f_d[k,0]/self.d_eigenvalues[k]
              beta[k,0] = f_v[k,0]/self.d_eigenvalues[k]
           else: # Zero eigenmode - this gets the mean value
              alpha[k,0] = f_d[k,0]/np.sum(self.rac)
              beta[k,0] = f_v[k,0]/np.sum(self.rac)
             
              
           if np.abs(self.n_eigenvalues[k]) > zeroTol:
              alpha[k,1] = f_d[k,1]/self.n_eigenvalues[k]
              beta[k,1] = f_v[k,1]/self.n_eigenvalues[k]
           else: # Zero eigenmode - this gets the mean value
              alpha[k,1] = f_d[k,1]/np.sum(self.rac)
              beta[k,1] = f_v[k,1]/np.sum(self.rac)
          
        
        return f_d, f_v, proj_d, proj_v, divergence, vorticity, alpha, beta
        
        
         
        
