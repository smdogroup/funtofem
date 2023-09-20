import unittest, importlib, numpy as np, os, sys
from funtofem import *
from mpi4py import MPI

np.random.seed(1234567)
comm = MPI.COMM_WORLD

tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "input_files", "flat_plate.csm")
dat_filepath = os.path.join(base_dir, "input_files", "loaded_plate.dat")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs

f2f_model = FUNtoFEMmodel("wing")
tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
tacs_model.egads_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=5,
    edge_pt_max=10,
    global_mesh_size=0.25,
    max_surf_offset=0.05,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)
f2f_model.tacs_model = tacs_model

# build a body which we will register variables to
wing = Body.aeroelastic("wing")

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

for iface in range(1, 4):
    caps2tacs.ThicknessVariable(
        caps_group=f"face{iface}", material=aluminum, value=0.03
    ).register_to(tacs_model)

# register the wing body to the model
wing.register_to(f2f_model)

# add remaining information to tacs model
caps2tacs.PinConstraint("perimeter").register_to(tacs_model)
caps2tacs.GridForce("middle", direction=[0, 0, 1.0], magnitude=100).register_to(
    tacs_model
)

# make a funtofem scenario
test_scenario = Scenario.steady("test", steps=100).include(Function.mass())
test_scenario.register_to(f2f_model)

flow_solver = TestAerodynamicSolver(comm, f2f_model)
transfer_settings = TransferSettings(npts=5, beta=0.5)

caps2tacs.AnalysisFunction.mass().register_to(tacs_model)
# setup the tacs model
tacs_aim = tacs_model.tacs_aim
# tacs_aim.setup_aim()

tacs_model.setup()
tacs_model.pre_analysis()
tacs_model.run_analysis()

# shape_driver = TacsOnewayDriver.prime_loads_shape(
#     flow_solver,
#     tacs_aim,
#     transfer_settings,
#     nprocs=1,
#     bdf_file=dat_filepath,
# )

# shape_driver.solve_forward()
