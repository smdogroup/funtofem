from __future__ import print_function
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys
from tacs import TACS, elements, constitutive

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
    
# initialize TACS for left plate
# Create the constitutvie propertes and model
props = constitutive.MaterialProperties()
con = constitutive.PlaneStressConstitutive(props)
heat = elements.HeatConduction2D(con)

# Create the basis class
quad_basis = elements.LinearQuadBasis()

# Create the element
element = elements.Element2D(heat, quad_basis)

# Load in the mesh
mesh = TACS.MeshLoader(comm)
mesh.scanBDFFile('rect_plate.bdf')

# Set the element
mesh.setElement(0, element)

# Create the assembler object
varsPerNode = heat.getVarsPerNode()
assembler = mesh.createTACS(varsPerNode)

res = assembler.createVec()
ans = assembler.createVec()
mat = assembler.createSchurMat()
pc = TACS.Pc(mat)

# Assemble the heat conduction matrix
assembler.assembleJacobian(1.0, 0.0, 0.0, res, mat)
pc.factor()
gmres = TACS.KSM(mat, pc, 20)

assembler.setBCs(res)
gmres.solve(res, ans)
assembler.setVariables(ans)

# Set the element flag
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS)
f5 = TACS.ToFH5(assembler, TACS.SCALAR_2D_ELEMENT, flag)
f5.writeToFile('verification.f5')
