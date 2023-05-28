import unittest, importlib, numpy as np, os, sys
from pyfuntofem import *
from mpi4py import MPI

np.random.seed(1234567)
comm = MPI.COMM_WORLD

tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "naca_wing2.csm")
# dat_filepath = os.path.join(base_dir, "input_files", "naca_wing2.dat")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs

# make the funtofem and tacs model
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
tacs_aim = tacs_model.tacs_aim

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)

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

# add remaining information to tacs model
caps2tacs.PinConstraint("root").register_to(tacs_model)

# setup the tacs model
tacs_aim.setup_aim()
# tacs_aim = tacs_model.tacs_aim
tacs_aim.pre_analysis()
