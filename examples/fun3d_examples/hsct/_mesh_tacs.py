# from funtofem import *
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
    max_surf_offset=0.04,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)

egads_aim = tacs_model.mesh_aim
tacs_aim = tacs_model.tacs_aim

nribs = int(tacs_model.get_config_parameter("nribs"))

tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)

# mesh settings
interior_ct = 8
exterior_ct = 2 * interior_ct - 1  # +1 for small#, -1 for large #
if comm.rank == 0:
    egads_aim = tacs_model.mesh_aim
    egads_aim.aim.input.Mesh_Sizing = {
        "rib1interior": {"numEdgePoints": interior_ct},
        "rib1exterior": {"numEdgePoints": exterior_ct},
        f"rib{nribs}interior": {"numEdgePoints": interior_ct},
        f"rib{nribs}exterior": {"numEdgePoints": exterior_ct},
    }


# if comm.rank == 0:
#     aim = tacs_aim.aim
#     aim.input.Mesh_Sizing =
if comm.rank == 0:
    egads_aim.aim.runAnalysis()
    egads_aim.aim.geometry.view()
exit()

# # setup the material and shell properties
# aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

# tacs_aim = tacs_model.tacs_aim
# tacs_aim.set_config_parameter("mode:flow", 0)
# tacs_aim.set_config_parameter("mode:struct", 1)
# nribs = int(tacs_model.get_config_parameter("nribs"))
# nspars = int(tacs_model.get_config_parameter("nspars"))
# nOML = int(tacs_aim.get_output_parameter("nOML"))

# for irib in range(1, nribs + 1):
#     caps2tacs.ThicknessVariable(
#         caps_group=f"rib{irib}", material=aluminum, value=0.04
#     ).register_to(tacs_model)

# for ispar in range(1, nspars + 1):
#     caps2tacs.ThicknessVariable(
#         caps_group=f"spar{ispar}", material=aluminum, value=0.04
#     ).register_to(tacs_model)

# for iOML in range(1, nOML + 1):
#     caps2tacs.ThicknessVariable(
#         caps_group=f"OML{iOML}", material=aluminum, value=0.04
#     ).register_to(tacs_model)

# for prefix in ["LE", "TE"]:
#     caps2tacs.ThicknessVariable(
#         caps_group=f"{prefix}spar", material=aluminum, value=0.04
#     ).register_to(tacs_model)

# # add remaining information to tacs model
# caps2tacs.PinConstraint("root").register_to(tacs_model)
# caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# # setup the tacs model
# tacs_model.setup(include_aim=True)
# tacs_model.pre_analysis()

# # print out the mesh empty soln (to view mesh)
# tacs_model.createTACSProbs(addFunctions=True)
# SPs = tacs_model.SPs
# for caseID in SPs:
#     SPs[caseID].writeSolution(
#         baseName="tacs_output",
#         outputDir=tacs_aim.analysis_dir,
#         number=0,
#     )
