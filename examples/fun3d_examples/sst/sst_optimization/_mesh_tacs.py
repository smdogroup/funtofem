"""
Sean P. Engelstad, Georgia Tech 2023
"""

from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "sst_v2.csm")

# make the funtofem and tacs model
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path, comm=comm, problem_name="struct_mesh"
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=2,
    edge_pt_max=20,
    global_mesh_size=0.3,
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)

egads_aim = tacs_model.mesh_aim
tacs_aim = tacs_model.tacs_aim

tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)

for proc in tacs_aim.active_procs:
    if comm.rank == proc:
        aim = tacs_model.mesh_aim.aim
        aim.input.Mesh_Sizing = {
            "horizX": {"numEdgePoints": 20},
            "horizY": {"numEdgePoints": 8},
            "vert": {"numEdgePoints": 4},
        }

view_mesh = False

if comm.rank == 0 and view_mesh:
    egads_aim.aim.runAnalysis()
    egads_aim.aim.geometry.view()
if view_mesh:
    exit()

# # setup the material and shell properties
titanium_alloy = caps2tacs.Isotropic.titanium_alloy().register_to(tacs_model)

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)
nribs = int(tacs_model.get_config_parameter("wing:nribs"))
nspars = int(tacs_model.get_config_parameter("wing:nspars"))
nOML = int(tacs_aim.get_output_parameter("wing:nOML"))

wing = Body.aeroelastic("wing", boundary=4)

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.05
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)
for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.05
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)
for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.03
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)
for name in ["LEspar", "TEspar"]:
    caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.03
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

# structural shape variables
for prefix in ["rib", "spar"]:
    Variable.shape(f"wing:{prefix}_a1", value=1.0).set_bounds(
        lower=0.6, upper=1.4
    ).register_to(wing)
    Variable.shape(f"wing:{prefix}_a2", value=0.0).set_bounds(
        lower=-0.3, upper=0.3
    ).register_to(wing)


# # add remaining information to tacs model
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.PinConstraint("wingFuse").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# # setup the tacs model
tacs_model.setup(include_aim=True)
tacs_model.pre_analysis()

# # print out the mesh empty soln (to view mesh)
tacs_model.createTACSProbs(addFunctions=True)
SPs = tacs_model.SPs
for caseID in SPs:
    SPs[caseID].writeSolution(
        baseName="tacs_output",
        outputDir=tacs_aim.analysis_dir,
        number=0,
    )
