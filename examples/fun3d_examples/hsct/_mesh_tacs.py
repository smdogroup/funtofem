from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "meshes", "hsct.csm")

# make the funtofem and tacs model
tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=20,
    edge_pt_max=30,
    global_mesh_size=0.04,
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = int(tacs_model.get_output_parameter("nOML"))

for irib in range(1, nribs + 1):
    caps2tacs.ThicknessVariable(caps_group=f"rib{irib}", material=aluminum, value=0.04)

for ispar in range(1, nspars + 1):
    caps2tacs.ThicknessVariable(
        caps_group=f"spar{ispar}", material=aluminum, value=0.04
    )

for iOML in range(1, nOML + 1):
    caps2tacs.ThicknessVariable(caps_group=f"OML{iOML}", material=aluminum, value=0.04)

for prefix in ["LE", "TE"]:
    caps2tacs.ThicknessVariable(
        caps_group=f"{prefix}spar", material=aluminum, value=0.04
    )

# add remaining information to tacs model
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# setup the tacs model
tacs_aim = tacs_model.tacs_aim
tacs_aim.setup_aim()
tacs_aim.pre_analysis()
