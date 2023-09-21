import os
import importlib, numpy as np, os
from funtofem import *
from mpi4py import MPI

np.random.seed(1234567)
comm = MPI.COMM_WORLD

tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "input_files", "simple_naca_wing.csm")
dat_filepath = os.path.join(base_dir, "input_files", "simple_naca_wing.dat")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs

# make the funtofem and tacs model
f2f_model = FUNtoFEMmodel("wing")
tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=5,
    edge_pt_max=10,
    global_mesh_size=0.1,
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

# build a body which we will register variables to
wing = Body.aeroelastic("wing")

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))

for irib in range(1, nribs + 1):
    caps2tacs.ShellProperty(
        caps_group=f"rib{irib}", material=aluminum, membrane_thickness=0.05
    ).register_to(tacs_model)
for ispar in range(1, nspars + 1):
    caps2tacs.ShellProperty(
        caps_group=f"spar{ispar}", material=aluminum, membrane_thickness=0.05
    ).register_to(tacs_model)
caps2tacs.ShellProperty(
    caps_group="OML", material=aluminum, membrane_thickness=0.03
).register_to(tacs_model)

# register any shape variables to the wing which are auto-registered to tacs model
# Variable.shape(name="rib_a1").set_bounds(
#     lower=0.4, value=1.0, upper=1.6
# ).register_to(wing)
Variable.shape(name="spar_a1").set_bounds(lower=0.4, value=1.0, upper=1.6).register_to(
    wing
)

# register the wing body to the model
wing.register_to(f2f_model)

# add remaining information to tacs model
caps2tacs.PinConstraint("root").register_to(tacs_model)

# make a funtofem scenario
test_scenario = Scenario.steady("test", steps=10).include(Function.mass())
test_scenario.register_to(f2f_model)

solvers = SolverManager(comm)
solvers.flow = TestAerodynamicSolver(comm, f2f_model)
aero_driver = TestAeroOnewayDriver(solvers, f2f_model)
transfer_settings = TransferSettings(npts=200, beta=0.5)

# setup the tacs model
tacs_model.setup()

tacs_driver = TacsOnewayDriver.prime_loads(
    aero_driver, transfer_settings=transfer_settings, nprocs=2
)

tacs_driver.solve_forward()

tinerface = tacs_driver.tacs_interface

# read in the mesh sens file
msens_hdl = open("mesh_sens.txt", "r")
lines = msens_hdl.readlines()
msens_hdl.close()

ns = wing.get_num_struct_nodes()
mesh_sens = np.zeros((3, ns))
for iline, line in enumerate(lines):
    chunks = line.strip().split(" ")
    ind = int(chunks[0]) - 1
    dx = float(chunks[1])
    dy = float(chunks[2])
    dz = float(chunks[3])
    mesh_sens[0, ind] = dx
    mesh_sens[1, ind] = dy
    mesh_sens[2, ind] = dz

# write the mesh sens into the disps of tacs assembler
assembler = tinerface.assembler
ans_array = tinerface.ans.getArray()
ndof = assembler.getVarsPerNode()
for i in range(3):
    ans_array[i::ndof] = mesh_sens[i, :]

assembler.setVariables(tinerface.ans)
tinerface.gen_output()
