"""
Steps required to prepare the funtofem analysis:

1) Build the mesh of the structure using caps2tacs in setup_tacs.py
    The tacs or structure work directory is capsStruct/Scratch/tacs
    (mesh files are .bdf,.dat files under the structure work directory)
2) Build the mesh of the fluid and fun3d config files using caps2fun in setup_fun3d.py
    The fun3d work directory is capsFluid/Scratch/fun3d/
    (fluid mesh file is a .lb8.ugrid file and are also in pointwise folder too)
    (fun3d config files are the fun3d.nml, moving_body.input, and .mapbc files in )
3) Now you can run this file on the HPC using the run.pbs (qsub run.pbs with your allocation)

"""

# other imports
import os, time
import numpy as np
from mpi4py import MPI
from typing import TYPE_CHECKING

# Import pyfuntofem modules
from pyfuntofem.model  import *
from pyfuntofem.driver import *
from pyfuntofem.fun3d_interface import *
from pyfuntofem.tacs_interface import createTacsInterfaceFromBDF

start_time = time.time()

# run settings
n_tacs_procs:int = 30
n_f2f_steps:int = 10
f2f_analysis_type:str = "aeroelastic"
model_name:str = "F2FDiamondWing"

"""
units settings to convert from fun3d non-dimensional to funtofem dimensional forces,disps, temps, heat fluxes, etc.
    qinf = 0.5*rho*vinf**2
    thermal_scale = 0.5*rho*vinf**3
"""
qinf = 3000.0
thermal_scale = 1.0e5

# make the structural DVs
struct_DVs = []
nribs = 10
for rib_idx in range(1,nribs+1):
    thickness = 0.2-0.005*rib_idx
    struct_DVs.append(thickness)

nspars = 1
for spar_idx in range(1,nspars+1):
    thickness = 0.4-0.08*spar_idx
    struct_DVs.append(thickness)

nOML = nribs-1
for OML_idx in range(1,nOML+1):
    thickness = 0.05-0.002*OML_idx
    struct_DVs.append(thickness)

# work directories for fun3d and tacs
cwd = os.getcwd()
tacs_directory = os.path.join(cwd, "capsStruct", "Scratch", "tacs")

# TODO : fun3d_client should not operate this way, you should give it the fun3d parent dir for each scenario or something
fun3d_parent_dir = os.path.join(cwd, "capsFluid", "Scratch")
fun3d_folder_name = "fun3d"
scenario_name = fun3d_folder_name

# number of tacs processors and setup MPI
comm = MPI.COMM_WORLD

# get the tacs interface with pytacs
dat_file = os.path.join(tacs_directory, "nastran_CAPS.dat")

# Build the model
model = FUNtoFEMmodel(model_name)
wing_body = Body('diamond_wing', analysis_type=f2f_analysis_type, group=0,boundary=2)

# add struct DVs to the body
for idx,dv_value in enumerate(struct_DVs):
    wing_body.add_variable('struct',Variable(f"thickness{idx}",value=dv_value,lower = 0.0001, upper = 1.0))

# add the body to the 
model.add_body(wing_body)

# make a new steady scenario that evaluates ksfailure and mass in funtofem
my_scenario = Scenario(scenario_name, group=0, steady=True, steps=n_f2f_steps)

ks_failure_function = Function("ksfailure", analysis_type='structural', options={'ksweight': 1000.0})
mass_function = Function("mass", analysis_type="structural")
my_scenario.add_function(ks_failure_function)
my_scenario.add_function(mass_function)

# add the scenario to the model
model.add_scenario(my_scenario)

# initialize the funtofem solvers
solvers = {}

solvers['flow'] = Fun3dInterface(comm=comm,model=model,flow_dt=1.0, qinf=qinf, thermal_scale=thermal_scale, fun3d_dir=fun3d_parent_dir)
solvers['structural'] = createTacsInterfaceFromBDF(
    model=model, comm=comm, nprocs=n_tacs_procs, bdf_file=dat_file, prefix=tacs_directory, callback=None, struct_options={}
)

# setup the tacs comm again for the driver
world_rank = comm.Get_rank()
if world_rank < n_tacs_procs:
    color = 1
    key = world_rank
else:
    color = MPI.UNDEFINED
    key = world_rank
tacs_comm = comm.Split(color, key)

# L&D transfer options
transfer_options = {'analysis_type': f2f_analysis_type,
                    'scheme': 'meld'}

# instantiate the driver
driver = FUNtoFEMnlbgs(solvers,comm,tacs_comm,0,comm,0,transfer_options,model=model)

# solve the forward analysis
driver.solve_forward()
functions = model.get_functions()

# solve the adjoint
driver.solve_adjoint()
gradients = model.get_function_gradients()
variables = model.get_variables()

comm.Barrier()
if comm.Get_rank() == 0:
    print(f"\nFinished funtofem analysis of {model_name}...")
    print(f"Displaying funtofem functions, gradients")
    for ifunc,func in enumerate(functions):
        print(f"function {func.name} = {func.value}")
        for ivar, var in enumerate(variables):
            print(f"d{func.name}/d{var.name} = {gradients[ifunc][ivar]}")