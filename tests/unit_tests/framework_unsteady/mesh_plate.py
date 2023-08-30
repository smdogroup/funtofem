import importlib, numpy as np, os, shutil
from funtofem import *
from mpi4py import MPI

comm = MPI.COMM_WORLD

tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "input_files", "stiffened_plate.csm")

if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs

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
# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nx = int(tacs_model.get_config_parameter("nx"))
ny = int(tacs_model.get_config_parameter("ny"))

for ix in range(1, nx + 1):
    for iy in range(1, ny + 1):
        caps2tacs.ShellProperty(
            caps_group=f"panel{ix}-{iy}", material=aluminum, membrane_thickness=0.05
        ).register_to(tacs_model)

# add remaining information to tacs model
caps2tacs.PinConstraint("fix").register_to(tacs_model)
caps2tacs.TemperatureConstraint("temp", temperature=0).register_to(tacs_model)

# setup the tacs model
tacs_model.setup()
tacs_model.pre_analysis()
