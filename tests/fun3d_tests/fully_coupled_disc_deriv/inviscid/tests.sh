python test_fun3d_grid_deformation_quads.py 2>&1 > grid_deformation_quads.txt
python test_fun3d_grid_deformation_tris.py 2>&1 > grid_deformation_tris.txt
python test_aero_loads_quads.py 2>&1 > aero_loads_quads.txt
python test_aero_loads_tris.py 2>&1 > aero_loads_tris.txt
mpiexec_mpt -n 4 python test_aero_loads_quads.py 2>&1 > aero_loads_mpi_quads.txt
python test_flow_states_quads.py 2>&1 > flow_states_quads.txt
pyhton test_flow_states_tris.py 2>&1 > flow_states_tris.txt
python test_fun3d_tacs_FD_quads.py 2>&1 > 1-derivs_quads.txt
python test_fun3d_tacs_FD_tris.py 2>&1 > 1-derivs_tris.txt
