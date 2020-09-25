from __future__ import print_function
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys
from tacs import TACS, elements, constitutive
from funtofem import TransferScheme

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
    
# initialize TACS for left plate
# Create the constitutvie propertes and model
left_kappa = 230.0
props = constitutive.MaterialProperties(kappa=left_kappa)
con = constitutive.PlaneStressConstitutive(props)
heat = elements.HeatConduction2D(con)

# Create the basis class
quad_basis = elements.LinearQuadBasis()

# Create the element
element = elements.Element2D(heat, quad_basis)

# Load in the mesh
mesh = TACS.MeshLoader(comm)
mesh.scanBDFFile('left_plate.bdf')

# Set the element
mesh.setElement(0, element)

# Create the assembler object
varsPerNode = heat.getVarsPerNode()
l_assembler = mesh.createTACS(varsPerNode)

# get structures nodes
left_pts = l_assembler.createNodeVec()
l_assembler.getNodes(left_pts)
left_pts_array = left_pts.getArray()

# get mapping of shared edge
# for left plate this is x=1.0 edge
# also get second column of node numbers (x=0.9 for this case)
# (used later to get the gradient in x direction)
left_surface = []
left_mapping = []
left_f_mapping = []
for i in range(len(left_pts_array) // 3):
    if left_pts_array[3*i]  > .99:
        left_surface.extend(left_pts_array[3*i:3*i+3])
        left_mapping.append(i)
    if left_pts_array[3*i]  > 0.89 and left_pts_array[3*i]  < 0.91:
        left_f_mapping.append(i)

# Create the vectors/matrices
l_res = l_assembler.createVec()
l_ans = l_assembler.createVec()
l_mat = l_assembler.createSchurMat()
l_pc = TACS.Pc(l_mat)

# Assemble the heat conduction matrix
l_assembler.assembleJacobian(1.0, 0.0, 0.0, l_res, l_mat)
l_pc.factor()
l_gmres = TACS.KSM(l_mat, l_pc, 20)

# Now initialize TACS for right plate in the same way
# Create the constitutvie propertes and model
#right_kappa = 460.0
right_kappa = 230.0
props = constitutive.MaterialProperties(kappa=right_kappa)
con = constitutive.PlaneStressConstitutive(props)
heat = elements.HeatConduction2D(con)

# Create the basis class
quad_basis = elements.LinearQuadBasis()

# Create the element
element = elements.Element2D(heat, quad_basis)

# Load in the mesh
mesh = TACS.MeshLoader(comm)
mesh.scanBDFFile('right_plate.bdf')

# Set the element
mesh.setElement(0, element)

# Create the assembler object
varsPerNode = heat.getVarsPerNode()
r_assembler = mesh.createTACS(varsPerNode)

# get structures nodes
right_pts = r_assembler.createNodeVec()
r_assembler.getNodes(right_pts)
right_pts_array = right_pts.getArray()

# get mapping of shared edge
# for right plate this is x=1.0 edge (since right plate
# goes from x=1.0 to x=2.0 and the BC is at 2.0)
right_surface = []
right_mapping = []
for i in range(len(right_pts_array) // 3):
    if right_pts_array[3*i]  < 1.01:
        right_surface.extend(right_pts_array[3*i:3*i+3])
        right_mapping.append(i)

# Create the vectors/matrices
r_res = r_assembler.createVec()
r_ans = r_assembler.createVec()
r_mat = r_assembler.createSchurMat()
r_pc = TACS.Pc(r_mat)

# Assemble the heat conduction matrix
r_assembler.assembleJacobian(1.0, 0.0, 0.0, r_res, r_mat)
r_pc.factor()
r_gmres = TACS.KSM(r_mat, r_pc, 20)

right_surface = np.array(right_surface)
left_surface = np.array(left_surface)
right_mapping = np.array(right_mapping)
left_mapping = np.array(left_mapping)

# initialize MELDThermal
# note that for this case with a very small number of nodes in each set,
# increasing the number of nearest neighbors drastically degrades the accuracy
# because it smears the flux over the interface.
# the main issue with this is that the flux at the top and bottom nodes in the mapping
# are half as much as the flux at he rest of the nodes
# also note that when setting the num_nearest to 1 (i.e. 1 to 1 transfer of quantites
# between selected nodes) we recover the exact solution at the boundary
meld = TransferScheme.pyMELDThermal(comm, comm, 0, comm, 0, -1, 3, 0.5) #axis of symmetry, num nearest, beta
# set left nodes as aero, right nodes as structure
# the nodes you want to send heat flux to become structure
# the nodes you want to send temperatures to become aero
meld.setStructNodes(right_surface)
meld.setAeroNodes(left_surface)
meld.initialize()

# assign an initial thermal BC to right edge of left plate
res_arr = l_res.getArray()
l_assembler.setBCs(l_res)
if len(left_mapping)>0:
    res_arr[left_mapping] = 380.0

r_assembler.setBCs(r_res)
r_gmres.solve(r_res, r_ans)
r_assembler.setVariables(r_ans)

# allocate storage vectors
flux_holder = np.zeros(len(left_mapping))
res_holder = np.zeros(len(right_mapping))
ans_holder = np.zeros(len(right_mapping))
theta_holder = np.zeros(len(left_mapping))

tempDiff = 1000.0
ct = 0
# iterate until sum of absolute values of temperature differences between subsequent solutions gets less than 0.01
while (tempDiff > .01):
    
    # get the heat flux from the left plate right edge
    # first, solve left:
    l_gmres.solve(l_res, l_ans)
    l_assembler.setVariables(l_ans)
    # temps at each node:
    ans_arr = l_ans.getArray()
    init_left_temps = []
    if len(left_mapping)>0:
        init_left_temps = ans_arr[left_mapping]

    # loop over points on the right edge
    for i in range(len(left_mapping)):
        # finite difference derivative
        # Division by 0.1 is for the distance between nodes in x direction
        grad = (ans_arr[left_mapping[i]] - ans_arr[left_f_mapping[i]])/0.1
        # put flux in array, multiplied by 0.1 for area weighting
        flux_holder[i] = -grad * left_kappa * 0.1
        if i == 0 or i == len(left_mapping) - 1:
            flux_holder[i] *= 0.5

    # set flux into TACS
    meld.transferFlux(flux_holder, res_holder)
    # transfer flux from res holder to res array based on mapping
    r_res.zeroEntries()
    res_array = r_res.getArray()
    for i in range(len(right_mapping)):
        res_array[right_mapping[i]] = res_holder[i]

    # set flux into assembler
    r_assembler.setBCs(r_res)
    
    # solve thermal problem
    r_gmres.solve(r_res, r_ans)
    r_assembler.setBCs(r_ans)
    r_assembler.setVariables(r_ans)
    
    ans_array = r_ans.getArray()
    
    # get the temps from the left edge of the right plate
    for i in range(len(right_mapping)):
        ans_holder[i] = ans_array[right_mapping[i]]

    # transfer surface temps to theta (size of nodes on left mapping)
    meld.transferTemp(ans_holder, theta_holder)

    res_array = l_res.getArray()    
    # now transfer those thetas to appropriate nodes in left plate residual
    # set the temperature to the initial temperature + 90% of the difference
    # between the new temp and the initial

    # note that the factor of 0.9 controls the speed of convergence
    # clearly at 0.5 it goes directly to the solution
    for i in range(len(left_mapping)):
        res_array[left_mapping[i]] = init_left_temps[i] + 0.75*(theta_holder[i] - init_left_temps[i])

    tempDiff = comm.allreduce(sum(abs(theta_holder - init_left_temps)))
    if comm.Get_rank()==0:
	print('Temperature Difference = ', tempDiff)
	ct += 1

if comm.Get_rank()==0:
    print('Iterations = ',ct)

# Set the element flag
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS)
f5 = TACS.ToFH5(r_assembler, TACS.SCALAR_2D_ELEMENT, flag)
f5.writeToFile('right_plate.f5')
f5 = TACS.ToFH5(l_assembler, TACS.SCALAR_2D_ELEMENT, flag)
f5.writeToFile('left_plate.f5')
