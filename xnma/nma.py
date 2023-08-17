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
#   range (1:nx-2,1:ny-2)
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

import numpy as np
zeroTol = 1e-12


class model:
    def __init__(self, prec=np.float32):
        self.ds = None
        self.grid = None

        self.ndofZ = 0
        self.ndofC = 0
        self.maskZ = None
        self.maskC = None

        self.eigenmodes = None  # Defined on tracer points
        self.eigenvalues = None  #
        self.prec = prec

    def construct(self, dx=1.0, dy=1.0, nx=500, ny=500, prec=np.float32):
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

        self.prec = prec
        self.dxc = np.ones((ny, nx)).astype(self.prec) * dx
        self.dxg = np.ones((ny, nx)).astype(self.prec) * dx

        self.dyc = np.ones((ny, nx)).astype(self.prec) * dy
        self.dyg = np.ones((ny, nx)).astype(self.prec) * dy

        self.raz = np.ones((ny, nx)).astype(self.prec) * dx * dy
        self.rac = np.ones((ny, nx)).astype(self.prec) * dx * dy

        dxl = np.ones((nx)).astype(self.prec) * dx
        self.xg = dxl.cumsum() - 2 * dx
        self.xc = self.xg + dx * 0.5

        dyl = np.ones((ny)).astype(self.prec) * dy
        self.yg = dyl.cumsum() - 2 * dy
        self.yc = self.yg + dy * 0.5

        # The stencil operations we use via numba
        # enforce masking on the bounding quad boundaries
        # Because of this, we set the maskC to zero on the
        # boundaries
        self.maskC = np.ones((ny, nx))
        self.maskC[:, 0] = 0.0
        self.maskC[:, nx - 1] = 0.0
        self.maskC[:, nx - 2] = 0.0
        self.maskC[0, :] = 0.0
        self.maskC[ny - 1, :] = 0.0
        self.maskC[ny - 2, :] = 0.0
        self.ndofC = self.maskC.sum().astype(int)
        self.hFacC = self.maskC

        # A consistent z-mask is set, assuming
        # z[j,i] is at the southwest corner of
        # c[j,i]
        self.maskZ = np.ones((ny, nx))
        self.hFacW = np.ones((ny, nx))
        self.hFacS = np.ones((ny, nx))
        self.maskW = np.ones((ny, nx))
        self.maskS = np.ones((ny, nx))
        for j in range(0, ny - 1):
            for i in range(0, nx - 1):
                if self.maskC[j, i] == 0.0:
                    self.maskZ[j, i] = 0.0
                    self.maskZ[j, i + 1] = 0.0
                    self.maskZ[j + 1, i] = 0.0
                    self.maskZ[j + 1, i + 1] = 0.0

                    self.maskW[j, i] = 0.0
                    self.maskW[j, i + 1] = 0.0
                    self.maskS[j, i] = 0.0
                    self.maskS[j + 1, i] = 0.0

        self.ndofZ = self.maskZ.sum().astype(int)
        #self.hFacW = self.maskW
        #self.hFacS = self.maskS
        
        print("Construction report")
        print(f"nDOF (C) : {self.ndofC}")
        print(f"nDOF (Z) : {self.ndofZ}")

    def circularDemo(self, dx=1.0, dy=1.0, nx=500, ny=500):
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

        self.dxc = np.ones((ny, nx)).astype(self.prec) * dx
        self.dxg = np.ones((ny, nx)).astype(self.prec) * dx

        self.dyc = np.ones((ny, nx)).astype(self.prec) * dy
        self.dyg = np.ones((ny, nx)).astype(self.prec) * dy

        self.raz = np.ones((ny, nx)).astype(self.prec) * dx * dy
        self.rac = np.ones((ny, nx)).astype(self.prec) * dx * dy

        dxl = np.ones((nx)).astype(self.prec) * dx
        self.xg = dxl.cumsum() - dx
        self.xc = self.xg + dx * 0.5

        dyl = np.ones((ny)).astype(self.prec) * dy
        self.yg = dyl.cumsum() - dy
        self.yc = self.yg + dy * 0.5

        self.maskZ = np.ones((ny, nx))
        self.maskZ[:, 0] = 0.0
        self.maskZ[:, nx - 1] = 0.0
        self.maskZ[0, :] = 0.0
        self.maskZ[ny - 1, :] = 0.0

        xc = self.xg[-1] * 0.5
        yc = self.yg[-1] * 0.5
        for j in range(0, ny):
            y = self.yg[j]
            for i in range(0, nx):
                x = self.xg[i]
                r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

                if r >= 0.9 * xc:
                    self.maskZ[j, i] = 0.0

        self.ndofZ = self.maskZ.sum().astype(int)

    def irregularHolesDemo(self, dx=1.0, dy=1.0, nx=500, ny=500):
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

        self.dxc = np.ones((ny, nx)).astype(self.prec) * dx
        self.dxg = np.ones((ny, nx)).astype(self.prec) * dx

        self.dyc = np.ones((ny, nx)).astype(self.prec) * dy
        self.dyg = np.ones((ny, nx)).astype(self.prec) * dy

        self.raz = np.ones((ny, nx)).astype(self.prec) * dx * dy
        self.rac = np.ones((ny, nx)).astype(self.prec) * dx * dy

        dxl = np.ones((nx)).astype(self.prec) * dx
        self.xg = dxl.cumsum() - dx
        self.xc = self.xg + dx * 0.5

        dyl = np.ones((ny)).astype(self.prec) * dy
        self.yg = dyl.cumsum() - dy
        self.yc = self.yg + dy * 0.5

        self.maskZ = np.ones((ny, nx))
        self.maskZ[:, 0] = 0.0
        self.maskZ[:, nx - 1] = 0.0
        self.maskZ[0, :] = 0.0
        self.maskZ[ny - 1, :] = 0.0

        xc = self.xg[-1] * 0.5
        yc = self.yg[-1] * 0.5
        for j in range(0, ny):
            y = self.yg[j]
            for i in range(0, nx):
                x = self.xg[i]

                # Place a hole to the north-east with 10% domain width radius
                r = np.sqrt((x - 1.7 * xc) ** 2 + (y - 1.7 * yc) ** 2)
                if r <= 0.1 * xc:
                    self.maskZ[j, i] = 0.0

                # Place a hole to the southwest with 15% domain width radius
                r = np.sqrt((x - 0.2 * xc) ** 2 + (y - 0.3 * yc) ** 2)
                if r <= 0.15 * xc:
                    self.maskZ[j, i] = 0.0

                # Place a hole to the south-east with 40% domain width radius
                r = np.sqrt((x - 1.3 * xc) ** 2 + (y - 0.4 * yc) ** 2)
                if r <= 0.4 * xc:
                    self.maskZ[j, i] = 0.0

                r = np.sqrt((x) ** 2 + (y - 2.0 * yc) ** 2)
                f = np.exp(-r / (0.2 * xc * xc))
                if f > 0.85:
                    self.maskZ[j, i] = 0.0

        self.ndofZ = self.maskZ.sum().astype(int)

    def load(
        self,
        dataDir,
        chunks=None,
        depth=0,
        x=None,
        y=None,
        iters=None,
        geometry="sphericalpolar",
    ):
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
            localChunks = {
                "XC": chunks[0],
                "XG": chunks[0],
                "YC": chunks[1],
                "YG": chunks[1],
            }

        self.ds = xmitgcm.open_mdsdataset(
            dataDir,
            iters=iters,
            prefix=["U", "V"],
            read_grid=True,
            chunks=localChunks,
            geometry=geometry,
        )

        if x:
            self.ds = self.ds.sel(XC=slice(x[0], x[1]), XG=slice(x[0], x[1]))

            if self.ds.XG.shape[0] > self.ds.XC.shape[0]:
                self.ds = self.ds.sel(XG=slice(x[0], self.ds.XC[-1]))

        if y:
            self.ds = self.ds.sel(YC=slice(y[0], y[1]), YG=slice(y[0], y[1]))

            # Some indexing by slicing returns YG values
            # surrounding YC values on south and north boundaries
            # This patch clips the last YG value
            if self.ds.YG.shape[0] > self.ds.YC.shape[0]:
                self.ds = self.ds.sel(YG=slice(y[0], self.ds.YC[-1]))

        # Create a grid object
        self.grid = xgcm.Grid(self.ds)

        # Copy data to numpy arrays (for numba acceleration)
        self.xc = self.ds.XC.to_numpy().astype(self.prec)
        self.yc = self.ds.YC.to_numpy().astype(self.prec)
        self.xg = self.ds.XG.to_numpy().astype(self.prec)
        self.yg = self.ds.YG.to_numpy().astype(self.prec)
        self.dxc = self.ds.dxC.to_numpy().astype(self.prec)
        self.dyc = self.ds.dyC.to_numpy().astype(self.prec)
        self.dxg = self.ds.dxG.to_numpy().astype(self.prec)
        self.dyg = self.ds.dyG.to_numpy().astype(self.prec)
        self.raz = self.ds.rAz.to_numpy().astype(self.prec)
        self.rac = self.ds.rA.to_numpy().astype(self.prec)

        zd = -abs(depth)  # ensure that depth is negative value
        self.U = self.ds.U.interp(Z=[zd], method="nearest")
        self.V = self.ds.V.interp(Z=[zd], method="nearest")

        self.hFacC = (
            self.ds.hFacC.interp(Z=[zd], method="nearest")
            .squeeze()
            .to_numpy()
            .astype(self.prec)
        )
        self.hFacW = (
            self.ds.hFacW.interp(Z=[zd], method="nearest")
            .squeeze()
            .to_numpy()
            .astype(self.prec)
        )
        self.hFacS = (
            self.ds.hFacS.interp(Z=[zd], method="nearest")
            .squeeze()
            .to_numpy()
            .astype(self.prec)
        )

        ny, nx = self.hFacC.shape

        self.maskC = np.ceil(self.hFacC)
        self.maskW = np.ceil(self.hFacW)
        self.maskS = np.ceil(self.hFacS)

        # The stencil operations we use via numba
        # enforce masking on the bounding quad boundaries
        # Because of this, we set the maskC to zero on the
        # boundaries
        self.maskC = np.ones((ny, nx))
        self.maskC[:, 0] = 0.0
        self.maskC[:, nx - 1] = 0.0
        self.maskC[:, nx - 2] = 0.0
        self.maskC[0, :] = 0.0
        self.maskC[ny - 1, :] = 0.0
        self.maskC[ny - 2, :] = 0.0
        self.ndofC = self.maskC.sum().astype(int)
        self.hFacC = self.maskC

        # A consistent z-mask is set, assuming
        # z[j,i] is at the southwest corner of
        # c[j,i]
        self.maskZ = np.ones((ny, nx))
        self.hFacW = np.ones((ny, nx))
        self.hFacS = np.ones((ny, nx))
        self.maskW = np.ones((ny, nx))
        self.maskS = np.ones((ny, nx))
        for j in range(0, ny - 1):
            for i in range(0, nx - 1):
                if self.maskC[j, i] == 0.0:
                    self.maskZ[j, i] = 0.0
                    self.maskZ[j, i + 1] = 0.0
                    self.maskZ[j + 1, i] = 0.0
                    self.maskZ[j + 1, i + 1] = 0.0
                    self.maskW[j, i] = 0.0
                    self.maskW[j, i + 1] = 0.0
                    self.maskS[j, i] = 0.0
                    self.maskS[j + 1, i] = 0.0

        self.ndofZ = self.maskZ.sum().astype(int)
        self.ndofC = self.maskC.sum().astype(int)

    def LapZInv_PCCG(
        self, b, s0=None, pcitermax=20, pctolerance=1e-2, itermax=1500, tolerance=1e-4, shift=0.0
    ):
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
            sk = np.zeros_like(b)

        sk = sk * self.maskZ

        r = kernels.LapZ_Residual(
            sk, b, self.maskZ, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
        )

        d = self.LapZInv_JacobiSolve(r, itermax=pcitermax, tolerance=pctolerance, shift=shift)

        delta = np.sum(r * d)
        rmag = np.sqrt(np.sum(r*r))
        bmag = np.sqrt(np.sum(b*b))
        r0 = np.max([rmag, bmag])

        for k in range(0, itermax):
            q = (
                kernels.LapZ(d, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift)
                * self.maskZ
            )

            alpha = delta / (np.sum(d * q))

            sk += alpha * d
            if k % 50 == 0:
                r = kernels.LapZ_Residual(
                    sk, b, self.maskZ, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
                )
            else:
                r -= alpha * q

            x = self.LapZInv_JacobiSolve(r, itermax=pcitermax, tolerance=pctolerance, shift=shift)

            rmag = np.sqrt(np.sum(r*r))
            deltaOld = delta
            delta = np.sum(r * x)
            beta = delta / deltaOld
            d = x + beta * d
            if rmag <= tolerance*r0:
                break

        if rmag > tolerance*r0:
            print(
                f"Conjugate gradient method did not converge in {k+1} iterations : {delta}"
            )

        return sk

    def LapZInv_JacobiSolve(self, b, s0=None, itermax=1000, tolerance=1e-4, shift=0.0):
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
            sk = np.zeros_like(b)

        sk = sk * self.maskZ
        r = kernels.LapZ_Residual(
            sk, b, self.maskZ, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
        )

        for k in range(0, itermax):
            ds = kernels.LapZ_JacobiDinv(
                r, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
            )

            dsmag = np.max(abs(ds))
            smag = np.max(abs(sk))
            if smag <= np.finfo(self.prec).eps:
                smag == np.max(abs(b))

            if smag > np.finfo(self.prec).eps:
                if dsmag / smag <= tolerance:
                    break

            # Update the solution
            sk += ds

            r = kernels.LapZ_Residual(
                sk, b, self.maskZ, self.dxc, self.dyc, self.dxg, self.dyg, self.raz, shift
            )

        return sk

    def LapCInv_PCCG(
        self,
        b,
        s0=None,
        pcitermax=0,
        pctolerance=1e-2,
        itermax=1500,
        tolerance=1e-4,
        dShift=1e-2,
    ):
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
            sk = np.zeros_like(b)

        sk = sk * self.maskC

        r = kernels.LapC_Residual(
            sk,
            b,
            self.maskC,
            self.dxc,
            self.dyc,
            self.dxg,
            self.dyg,
            self.maskW,
            self.maskS,
            self.rac,
            dShift,
        )

        if pcitermax == 0:
            d = r * self.maskC
        else:
            d = self.LapCInv_JacobiSolve(
                r, itermax=pcitermax, tolerance=pctolerance, dShift=dShift
            )

        delta = np.sum(r * d)
        rmag = np.sqrt(np.sum(r*r))
        bmag = np.sqrt(np.sum(b*b))
        r0 = np.max([rmag, bmag])

        for k in range(0, itermax):
            # print(f"(k,r) : ({k},{rmag})")
            q = (
                kernels.LapC(
                    d,
                    self.dxc,
                    self.dyc,
                    self.dxg,
                    self.dyg,
                    self.maskW,
                    self.maskS,
                    self.rac,
                    dShift,
                )
                * self.maskC
            )

            alpha = delta / (np.sum(d * q))

            sk += alpha * d
            if k % 50 == 0:
                r = kernels.LapC_Residual(
                    sk,
                    b,
                    self.maskC,
                    self.dxc,
                    self.dyc,
                    self.dxg,
                    self.dyg,
                    self.maskW,
                    self.maskS,
                    self.rac,
                    dShift,
                )
            else:
                r -= alpha * q

            if pcitermax == 0:
                x = r * self.maskC
            else:
                x = self.LapCInv_JacobiSolve(
                    r, itermax=pcitermax, tolerance=pctolerance, dShift=dShift
                )

            rmag = np.sqrt(np.sum(r*r))
            deltaOld = delta
            delta = np.sum(r * x)
            beta = delta / deltaOld
            d = x + beta * d
            if rmag <= tolerance*r0 :
                break

        if rmag > tolerance*r0:
            print(
                f"Conjugate gradient method did not converge in {k+1} iterations : {rmag}"
            )

        return sk

    def LapCInv_JacobiSolve(
        self, b, s0=None, itermax=1000, tolerance=1e-4, dShift=1e-2
    ):
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
            sk = np.zeros_like(b)

        sk = sk * self.maskC
        r = kernels.LapC_Residual(
            sk,
            b,
            self.maskC,
            self.dxc,
            self.dyc,
            self.dxg,
            self.dyg,
            self.maskW,
            self.maskS,
            self.rac,
            dShift,
        )

        for k in range(0, itermax):
            ds = kernels.LapC_JacobiDinv(
                r,
                self.dxc,
                self.dyc,
                self.dxg,
                self.dyg,
                self.maskW,
                self.maskS,
                self.rac,
                dShift,
            )

            dsmag = np.max(abs(ds))
            smag = np.max(abs(sk))
            if smag <= np.finfo(self.prec).eps:
                smag == np.max(abs(b))

            if smag > np.finfo(self.prec).eps:
                if dsmag / smag <= tolerance:
                    break

            # Update the solution
            sk += ds

            r = kernels.LapC_Residual(
                sk,
                b,
                self.maskC,
                self.dxc,
                self.dyc,
                self.dxg,
                self.dyg,
                self.maskW,
                self.maskS,
                self.rac,
                dShift,
            )

        return sk

    def laplacianZ(self, x, dShift):
        """Wrapper for the Laplacian, where x comes in as a flat 1-D array
        only at `wet` grid cell locations"""

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels

        # x comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        xgrid = ma.masked_array(
            np.zeros(self.maskZ.shape), mask=abs(self.maskZ - 1.0), dtype=self.prec
        )
        xgrid[~xgrid.mask] = x  # Set interior values to b

        # Invert the laplacian
        Lx = kernels.LapZ(x, self.dxc, self.dyc, self.dxg, self.dyg, self.raz)

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array(
            Lx, mask=abs(self.maskZ - 1.0), dtype=self.prec
        ).compressed()

    def laplacianZInverse(self, b, dShift):
        """Wrapper for the Laplacian Inverse (with preconditioned conjugate gradient),
        where b comes in as a flat 1-D array only at `wet` grid cell locations"""

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels

        # import time

        # b comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        # Use the model.b attribute to push the DOF formatted data

        bgrid = ma.masked_array(
            np.zeros(self.maskZ.shape), mask=abs(self.maskZ - 1.0), dtype=self.prec
        )
        bgrid[~bgrid.mask] = b  # Set interior values to b

        x = np.ones(self.maskZ.shape, dtype=self.prec)
        # Invert the laplacian
        x = self.LapZInv_PCCG(
            bgrid.data,
            s0=None,
            pcitermax=20,
            pctolerance=1e-2,
            itermax=3000,
            tolerance=1e-14,
            shift=dShift
        )

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array(
            x, mask=abs(self.maskZ - 1.0), dtype=self.prec
        ).compressed()

    def laplacianC(self, x, dShift):
        """Wrapper for the Laplacian, where x comes in as a flat 1-D array
        only at `wet` grid cell locations"""

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels

        # x comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        xgrid = ma.masked_array(
            np.zeros(self.maskC.shape), mask=abs(self.maskC - 1.0), dtype=self.prec
        )
        xgrid[~xgrid.mask] = x  # Set interior values to b

        # Invert the laplacian
        Lx = kernels.LapC(
            x,
            self.dxc,
            self.dyc,
            self.dxg,
            self.dyg,
            self.maskW,
            self.maskS,
            self.raC,
            dShift,
        )

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array(
            Lx, mask=abs(self.maskC - 1.0), dtype=self.prec
        ).compressed()

    def laplacianCInverse(self, b, dShift):
        """Wrapper for the Laplacian Inverse (with preconditioned conjugate gradient),
        where b comes in as a flat 1-D array only at `wet` grid cell locations"""

        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels
        import time

        # b comes in as a 1-D array in "DOF" format
        # we need to convert it to a 2-D array consistent with the model grid
        # Use the model.b attribute to push the DOF formatted data

        bgrid = ma.masked_array(
            np.zeros(self.maskC.shape), mask=abs(self.maskC - 1.0), dtype=self.prec
        )
        bgrid[~bgrid.mask] = b  # Set interior values to b

        x = np.ones(self.maskC.shape, dtype=self.prec)
        # Invert the laplacian
        x = self.LapCInv_PCCG(
            bgrid.data,
            s0=None,
            pcitermax=20,
            pctolerance=1e-2,
            itermax=3000,
            tolerance=1e-14,
            dShift=dShift,
        )

        # Mask the data, so that we can return a 1-D array of unmasked values
        return ma.masked_array(
            x, mask=abs(self.maskC - 1.0), dtype=self.prec
        ).compressed()

    def findDirichletModes(self, nmodes=10, tolerance=0, shift=0):
        """Find the eigenpairs associated with the Laplacian operator on
        vorticity points with homogeneous Dirichlet boundary conditions.

        """
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import eigsh
        import time
        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels

        shape = (self.ndofZ, self.ndofZ)
        Linv = LinearOperator(
            shape, matvec=lambda b: self.laplacianZInverse(b, shift), dtype=self.prec
        )

        print("[Dirichlet modes] Starting eigenvalue search")
        tic = time.perf_counter()
        evals, evecs = eigsh(Linv, k=nmodes, tol=tolerance, return_eigenvectors=True)
        toc = time.perf_counter()
        runtime = toc - tic
        print(f"[Dirichlet modes] eigsh runtime : {runtime:0.4f} s")

        sgrid = ma.masked_array(
            np.zeros(self.maskZ.shape), mask=abs(self.maskZ - 1.0), dtype=self.prec
        )
        ny, nx = self.maskZ.shape
        eigenvalues = np.zeros((nmodes), dtype=self.prec)
        eigenmodes = np.zeros((nmodes, ny, nx), dtype=self.prec)

        for k in range(0, nmodes):
            ev = 1.0 / evals[k] + shift
            if np.abs(ev) < np.abs(zeroTol):
                eigenvalues[k] = 0.0
            else:
                eigenvalues[k] = ev

            # Interpolate the dirichlet modes from the vorticity points
            # to the tracer points and store the result in sgrid
            sgrid[~sgrid.mask] = evecs[:, k]
            g = sgrid.data * self.maskZ

            # Normalize so that the norm of the eigenmode is 1
            mag = np.sqrt(np.sum(g * g * self.raz))
            eigenmodes[k, :, :] = g / mag

        self.d_eigenvalues = eigenvalues
        self.d_eigenmodes = eigenmodes

    def findNeumannModes(self, nmodes=10, tolerance=0, shift=1e-2):
        """Find the eigenpairs associated with the Laplacian operator on
        tracer points with homogeneous Neumann boundary conditions.

        """
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import eigsh
        import time
        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels

        shape = (self.ndofC, self.ndofC)
        Linv = LinearOperator(
            shape, matvec=lambda b: self.laplacianCInverse(b, shift), dtype=self.prec
        )
        print("[Neumann modes] Starting eigenvalue search")
        tic = time.perf_counter()
        evals, evecs = eigsh(Linv, k=nmodes, tol=tolerance, return_eigenvectors=True)
        toc = time.perf_counter()
        runtime = toc - tic
        print(f"[Neumann modes] eigsh runtime : {runtime:0.4f} s")

        ny, nx = self.maskC.shape
        eigenvalues = np.zeros((nmodes), dtype=self.prec)
        eigenmodes = np.zeros((nmodes, ny, nx), dtype=self.prec)
        sgrid = ma.masked_array(
            np.zeros(self.maskC.shape), mask=abs(self.maskC - 1.0), dtype=self.prec
        )
        
        for k in range(0, nmodes):
            ev = 1.0 / evals[k] + shift
            if np.abs(ev) < np.abs(zeroTol):
                eigenvalues[k] = 0.0
            else:
                eigenvalues[k] = ev

            sgrid[~sgrid.mask] = evecs[:, k]
            g = sgrid.data * self.maskC

            # Normalize so that the norm of the eigenmode is 1
            mag = np.sqrt(np.sum(g * g * self.rac))
            eigenmodes[k, :, :] = g / mag

        self.n_eigenvalues = eigenvalues
        self.n_eigenmodes = eigenmodes

    def findEigenmodes(self, nmodes=10, tolerance=0, deShift=0, neShift=1e-2):
        """Finds the eigenmodes using sci-py `eigsh`.

        Parameters

        nmodes - the number of eigenmodes you wish to find
        sigma  - Eigenmodes with eigenvalues near sigma are returned
        which  - Identical to the scipy `which` argument
        deShift - factor to shift the diagonal by for the dirichlet mode operator
        neShift - factor to shift the diagonal by for the neumann mode operator

        See scipy/eigsh docs for more details

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html

        """
        
        self.findDirichletModes(nmodes, tolerance, deShift)
        self.findNeumannModes(nmodes, tolerance, neShift)

    def vectorProjection(self, u, v):
        import numpy as np
        from numpy import ma
        import xnma.kernels as kernels

        # if the eigenmodes have not been found
        # find them, using the default parameters
        # if (self.eigenmodes is None) :
        #     self.findEigenmodes()

        print("Calculating projection of u,v")

        # Calculate the divergence
        divergence = (
            kernels.divergence(
                u, v, self.dxg, self.dyg, self.hFacW, self.hFacS, self.rac
            )
            * self.maskC
        )

        # Calculate the vorticity
        vorticity = kernels.vorticity(u, v, self.dxc, self.dyc, self.raz) * self.maskZ

        nmodes = self.d_eigenvalues.shape[0]
        ny, nx = u.shape
        db_m = np.zeros(
            (nmodes), dtype=self.prec
        )  # Projection of divergence onto the neumann modes (boundary)
        di_m = np.zeros(
            (nmodes), dtype=self.prec
        )  # Projection of divergence onto the neumann modes (interior)
        vb_m = np.zeros(
            (nmodes), dtype=self.prec
        ) # Projection of vorticity onto the dirichlet modes (boundary)
        vi_m = np.zeros(
            (nmodes), dtype=self.prec
        )  # Projection of vorticity onto the dirichlet modes (interior)

        for k in range(0, nmodes):
            vi_m[k] = np.sum(
                vorticity * np.squeeze(self.d_eigenmodes[k, :, :]) * self.raz
            )  # Projection of vorticity onto the dirichlet modes
            di_m[k] = np.sum(
                divergence * np.squeeze(self.n_eigenmodes[k, :, :]) * self.rac
            )  # Projection of divergence onto the neumann modes

            # Calculate the n_eigenmodes on u-points
            etak = np.squeeze(self.n_eigenmodes[k, :, :])
            etak = kernels.prolongTracer(etak)
            uEtak = kernels.TtoU(etak) * u
            # Calculate the n_eigenmodes on v-points
            vEtak = kernels.TtoV(etak) * v

            # Calculate the divergence of \vec{u} \eta_k
            divUEta = (
                kernels.divergence(
                    uEtak, vEtak, self.dxg, self.dyg, self.hFacW, self.hFacS, self.rac
                )
                * self.maskC
            )
            # Subtract the boundary contribution from the divergence coefficients
            db_m[k] = -np.sum(divUEta * self.rac)

        # proj_d = np.zeros((ny, nx), dtype=self.prec)
        # proj_v = np.zeros((ny, nx), dtype=self.prec)
        # rotEnergy = np.zeros((nmodes), dtype=self.prec)
        # divEnergy = np.zeros((nmodes), dtype=self.prec)
        # for k in range(0, nmodes):
            
           
        #     # Rotational Energy (Dirichlet Modes)
        #     rotEnergy[k] = -0.5 * v_m[k] * v_m[k] / self.d_eigenvalues[k]
        #     # projection
        #     proj_v += v_m[k] * np.squeeze(self.d_eigenmodes[k, :, :])/ self.d_eigenvalues[k]

        #     # Divergent Energy (Neumann Modes)
        #     # Only calculate energy for eigenmodes with non-zero eigenvalue
        #     if np.abs(self.n_eigenvalues[k]) > zeroTol:
        #         divEnergy[k] = -0.5 * d_m[k] * d_m[k] / self.n_eigenvalues[k]
        #         # projection
        #         proj_d += d_m[k] * np.squeeze(self.n_eigenmodes[k, :, :]) / self.n_eigenvalues[k]

        return di_m, db_m, vi_m, vb_m
