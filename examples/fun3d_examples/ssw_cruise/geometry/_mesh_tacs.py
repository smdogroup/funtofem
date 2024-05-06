# from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))

# make the funtofem and tacs model
tacs_model = caps2tacs.TacsModel.build(
    csm_file="ssw.csm",
    comm=comm,
    problem_name="struct_mesh",
    verbosity=1,
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

tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)

if comm.rank == 0:
    aim = egads_aim.aim
    aim.input.Mesh_Sizing = {
        "chord": {"numEdgePoints": 20},
        "span": {"numEdgePoints": 8},
        "vert": {"numEdgePoints": 4},
        # "LEribFace": {"tessParams": [0.01, 0.1, 3]},
        # "LEribEdge": {"numEdgePoints": 20},
    }
    # "TEribFace" : {"tessParams" : [0.03, 0.1, 3]},
    # "TEribEdge": {"numEdgePoints": 20},

if comm.rank == 0:
    egads_aim.aim.runAnalysis()
    egads_aim.aim.geometry.view()
exit()

# # setup the material and shell properties
titanium_alloy = caps2tacs.Isotropic.titanium_alloy().register_to(tacs_model)

tacs_aim = tacs_model.tacs_aim
nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = nribs - 1

for irib in range(1, nribs + 1):
    caps2tacs.ThicknessVariable(
        caps_group=f"rib{irib}", material=titanium_alloy, value=0.04
    ).register_to(tacs_model)

for ispar in range(1, nspars + 1):
    caps2tacs.ThicknessVariable(
        caps_group=f"spar{ispar}", material=titanium_alloy, value=0.04
    ).register_to(tacs_model)
for iOML in range(1, nOML + 1):
    caps2tacs.ThicknessVariable(
        caps_group=f"OML{iOML}", material=titanium_alloy, value=0.04
    ).register_to(tacs_model)

for prefix in ["LE", "TE"]:
    caps2tacs.ThicknessVariable(
        caps_group=f"{prefix}spar", material=titanium_alloy, value=0.04
    ).register_to(tacs_model)

# # add remaining information to tacs model
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# # setup the tacs model
tacs_model.setup(include_aim=True)
tacs_model.pre_analysis()

# # print out the mesh empty soln (to view mesh)
# tacs_model.createTACSProbs(addFunctions=True)
# SPs = tacs_model.SPs
# for caseID in SPs:
#     SPs[caseID].writeSolution(
#         baseName="tacs_output",
#         outputDir=tacs_aim.analysis_dir,
#         number=0,
#     )
