"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# IMPORT NECESSARY LIBRARIES
import numpy as np
from mpi4py import MPI
from tacs import TACS, elements, constitutive, functions
from pyfuntofem import TransferScheme
from paropt import ParOpt

################################################################################
#                                   TACS                                       #
################################################################################

# Load structural mesh from bdf file
tacs_comm = MPI.COMM_WORLD
struct_mesh = TACS.MeshLoader(tacs_comm)
struct_mesh.scanBDFFile("CRM_box_2nd.bdf")

# Set constitutive properties
rho = 2500.0  # density, kg/m^3
E = 70e9  # elastic modulus, Pa
nu = 0.3  # poisson's ratio
kcorr = 5.0 / 6.0  # shear correction factor
ys = 350e6  # yield stress, Pa
min_thickness = 0.002
max_thickness = 0.20
thickness = 0.04

# Loop over components, creating stiffness and element object for each
num_components = struct_mesh.getNumComponents()
for i in range(num_components):
    descriptor = struct_mesh.getElementDescript(i)
    stiff = constitutive.isoFSDT(
        rho, E, nu, kcorr, ys, thickness, i, min_thickness, max_thickness
    )
    element = None
    if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
        element = elements.MITCShell(2, stiff, component_num=i)
    struct_mesh.setElement(i, element)

# Create tacs assembler object
tacs = struct_mesh.createTACS(6)

# Create distributed node vector from TACS Assembler object and extract the
# node locations
struct_X_vec = tacs.createNodeVec()
tacs.getNodes(struct_X_vec)
struct_X = struct_X_vec.getArray().astype(TransferScheme.dtype)
struct_nnodes = len(struct_X) / 3

################################################################################
#                              TransferScheme                                  #
################################################################################

# Load points/forces from data file
XF = np.loadtxt("funtofemforces.dat")

# Get the points/forces
aero_X = XF[:, :3].flatten().astype(TransferScheme.dtype)
aero_loads = (251.8 * 251.8 / 2 * 0.3) * XF[:, 3:].flatten().astype(
    TransferScheme.dtype
)

aero_nnodes = aero_X.shape[0] / 3

# Create TransferScheme object
isymm = -1
num_nearest = 20
beta = 0.5
meld = TransferScheme.pyMELD(
    tacs_comm, tacs_comm, 0, tacs_comm, 0, isymm, num_nearest, beta
)

# Load structural and aerodynamic meshes into TransferScheme
meld.setStructNodes(struct_X)
meld.setAeroNodes(aero_X)

# Initialize TransferScheme
meld.initialize()

# Set and apply zero structural displacements (initial conditions)
struct_disps = np.zeros(len(struct_X), dtype=TransferScheme.dtype)
aero_disps = np.zeros(len(aero_X), dtype=TransferScheme.dtype)
meld.transferDisps(struct_disps, aero_disps)

# Transfer loads from fluid and get loads on structure
struct_loads = np.zeros(len(struct_X), dtype=TransferScheme.dtype)
meld.transferLoads(aero_loads, struct_loads)

# Set loads on structure
struct_loads_moments = np.zeros(2 * len(struct_loads))
struct_loads_moments[::6] = struct_loads[::3]
struct_loads_moments[1::6] = struct_loads[1::3]
struct_loads_moments[2::6] = struct_loads[2::3]

write_flag = (
    TACS.ToFH5.NODES
    | TACS.ToFH5.DISPLACEMENTS
    | TACS.ToFH5.STRAINS
    | TACS.ToFH5.STRESSES
    | TACS.ToFH5.EXTRAS
)
f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, write_flag)

################################################################################
#                                  ParOpt                                      #
################################################################################


# Create sizing as a ParOpt problem
class CRMSizing(ParOpt.pyParOptProblem):
    def __init__(self, tacs, f5):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD

        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.nvars = num_components / self.size
        if num_components % self.size != 0 and self.rank == self.size - 1:
            self.nvars += num_components % self.size
        self.ncon = 1

        # Initialize the base class
        super(CRMSizing, self).__init__(self.comm, self.nvars, self.ncon)

        self.tacs = tacs
        self.f5 = f5
        self.res = self.tacs.createVec()
        self.ans = self.tacs.createVec()
        self.mat = self.tacs.createFEMat()
        self.pc = TACS.Pc(self.mat)

        # Create list of required TACS functions (mass, ksfailure)
        self.mass = functions.StructuralMass(self.tacs)
        ksweight = 50.0
        alpha = 1.0
        self.ksfailure = functions.KSFailure(self.tacs, ksweight, alpha)
        self.ksfailure.setLoadFactor(1.5)
        self.funclist = [self.mass, self.ksfailure]

        self.svsens = self.tacs.createVec()
        self.adj = self.tacs.createVec()

        self.adjSensProdArray = np.zeros(num_components)
        self.tempdvsens = np.zeros(num_components)

        # Get initial mass for scaling
        self.initmass = self.tacs.evalFunctions([self.funclist[0]])
        self.xscale = 0.0025

        # Keep track of the number of gradient evaluations
        self.gevals = 0

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        self.tacs.getDesignVars(x)
        self.tacs.getDesignVarRange(lb, ub)

        # Scale the design variable values
        x[:] /= self.xscale
        lb[:] /= self.xscale
        ub[:] /= self.xscale

        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""

        xtacs = self.xscale * x

        # Set design variables into TACS
        self.tacs.setDesignVars(xtacs)

        # Assemble the Jacobian
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        self.tacs.assembleJacobian(alpha, beta, gamma, self.res, self.mat)
        self.pc.factor()

        # Set residual
        res_array = self.res.getArray()
        res_array[:] = struct_loads_moments[:]
        self.res.applyBCs()

        # Solve
        self.pc.applyFactor(self.res, self.ans)
        self.tacs.setVariables(self.ans)

        # Evaluate functions
        fvals = self.tacs.evalFunctions(self.funclist)
        massval = fvals[0] / self.initmass
        ksfailureval = fvals[1]

        fail = 0
        fobj = massval
        con = np.zeros(1)
        con[0] = 1 - ksfailureval
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # Write the output file
        if self.gevals % 10 == 0:
            self.f5.writeToFile("sizing%03d.f5" % (self.gevals))
        self.gevals += 1

        # The objective gradient
        # Evaluate the derivative of the mass w.r.t design variables
        self.tacs.evalDVSens(self.funclist[0], g)
        g[:] = g[:] / self.initmass

        # The constraint gradient
        # Evaluate the derivative of the constraint w.r.t design variables
        self.tacs.evalDVSens(self.funclist[1], self.tempdvsens)

        self.tacs.evalSVSens(self.funclist[1], self.svsens)
        self.svsens.scale(-1.0)
        self.pc.applyFactor(self.svsens, self.adj)
        self.tacs.evalAdjointResProduct(self.adj, self.adjSensProdArray)

        A[:] = -1.0 * (self.tempdvsens + self.adjSensProdArray)

        # Scale the gradient values
        g[:] *= self.xscale
        A[:] *= self.xscale

        return fail


# Create the problem object
sizing = CRMSizing(tacs, f5)
max_lbfgs = 30
opt = ParOpt.pyParOpt(sizing, max_lbfgs, ParOpt.BFGS)

# Set the output file
opt.setOutputFile("paropt_history.out")

# Set optimization parameters
opt.setGMRESSubspaceSize(30)
opt.setNKSwitchTolerance(1e3)
opt.setGMRESTolerances(0.1, 1e-30)
opt.setUseHvecProduct(0)
opt.setMajorIterStepCheck(45)
opt.setMaxMajorIterations(20000)
opt.setGradientCheckFrequency(100, 1e-6)
opt.setMaxLineSearchIters(5)
opt.setBarrierFraction(0.25)
opt.setBarrierPower(1.15)

opt.optimize()

x = opt.getOptimizedPoint()
np.savetxt("ans.dat", x)
