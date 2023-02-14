#!/usr/bin/env python
#

from scipy.sparse.linalg import LinearOperator

class model:
    def __init__(self):

        self.ds = None
        self.grid = None

        # A scipy linear operator ( https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html )
        # Used in the linalg.eigs method (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html)
        # to obtain a subset of eigenmodes
        self.phiOperator = None # Velocity potential operator
        self.psiOperator = None # Pseudo-stream-function operator

        self.phiModes = None # Neumann Modes
        self.phiEvals = None # Neumann Mode eigenvalues
        self.psiModes = None # Dirichlet Modes
        self.psiEvals = None # Neumann Mode eigenvalues


    def loadGrid(self, dataDir):
        """Loads in grid from MITgcm metadata files in dataDir"""
        import xmitgcm
        import xgcm

        self.ds = xmitgcm.open_mdsdataset(dataDir,
                iters=None,prefix=None,read_grid=True,
                geometry = "sphericalpolar")

        # Create a grid object
        self.grid = xgcm.Grid(self.ds)

        return 0


#    def calculateModes(self, nmodes=10, klev=0):
#        """Calculates the Neumann and Dirichlet modes with their
#        associated eigenvalues. The first `nmodes` modes are calculated
#        using the grid variables at the vertical level `klev`
#        """
#
#        self.calculateNeumannModes( self, nmodes, klev )
#
#
#    def calculateNeumannModes(self, nmodes=10, klev=0):
#        from scipy.sparse.linalg import LinearOperator
#        from scipy.sparse.linalg import eigs
#        import numpy as np
#    
#        def dofToIJ( s, model, klev ):
#        def ijToDOF( s, model, klev ):
#
#        def boundaryConditions( s, model, klev ):
#        def gradS( s, model, klev ):
#        def divGradS( sx, sy, model, klev ):
#
#        def phiOperator( s, model, nmodes, klev ):
#        """ Defines the Laplacian operator for the Neumann Modes """
#    
#            # s comes in DOF format
#            # >> map to ij format
#            #
#
#            # See https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html#Divergence
#            # for quick formulas
#            # Apply boundary conditions ( homogeneous neumann )
#
#            # Calculate grad s
#
#            # Calculate div( grad s )
#
#            # Map to DOF format, store in As
#            As = s
#            return As
#    
#        shape = (10,10) # TO DO : get the number of wet points
#        A = LinearOperator(shape, matvec=lambda x: phiOperator(x, model, nmodes, klev))
