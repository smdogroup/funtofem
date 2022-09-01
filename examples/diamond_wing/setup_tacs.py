import caps2tacs
import os

panel_problem = caps2tacs.CapsStruct.default(csmFile="diamondWing.csm")
tacs_aim = panel_problem.tacsAim
egads_aim = panel_problem.egadsAim

aluminum = caps2tacs.Isotropic.aluminum()
tacs_aim.add_material(material=aluminum)

# add wing root constraint
constraint = caps2tacs.ZeroConstraint(name="fixRoot", caps_constraint="wingRoot")
tacs_aim.add_constraint(constraint=constraint)

thick_idx = 0
nribs = 10
for rib_idx in range(1,nribs+1):
    thick_DV = caps2tacs.ThicknessVariable(name=f"thick{thick_idx}", caps_group=f"rib{rib_idx}", value=0.2-0.005*rib_idx, material=aluminum)
    tacs_aim.add_variable(variable=thick_DV)
    thick_idx += 1

nspars = 1
for spar_idx in range(1,nspars+1):
    thick_DV = caps2tacs.ThicknessVariable(name=f"thick{thick_idx}", caps_group=f"spar{spar_idx}", value=0.4-0.08*spar_idx, material=aluminum)
    tacs_aim.add_variable(variable=thick_DV)
    thick_idx += 1

nOML = nribs-1
for OML_idx in range(1,nOML+1):
    thick_DV = caps2tacs.ThicknessVariable(name=f"thick{thick_idx}", caps_group=f"OML{OML_idx}", value=0.05-0.002*OML_idx, material=aluminum)
    tacs_aim.add_variable(variable=thick_DV)
    thick_idx += 1

tacs_aim.setup_aim()
egads_aim.set_mesh(edge_pt_min=20, edge_pt_max=30, global_mesh_size=0.10, max_surf_offset=0.01, max_dihedral_angle=5)

# make a pytacs function
pytacs_function = caps2tacs.MassStress()

caps_tacs = caps2tacs.CapsTacs(
    name="naca_wing_struct", tacs_aim=tacs_aim, 
    egads_aim=egads_aim, pytacs_function=pytacs_function
    )

# build the mesh
caps_tacs.tacs_aim.pre_analysis()