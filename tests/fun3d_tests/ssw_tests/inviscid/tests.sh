export MY_MPI_EXEC=mpiexec_mpt
export MY_NPROCS=48
$MY_MPI_EXEC -n $MY_PROCS python test_aero_loads.py 2>&1 > aero_loads.txt
$MY_MPI_EXEC -n 1 python test_flow_states.py 2>&1 > flow_states.txt
$MY_MPI_EXEC -n 1 python test_grid_deformation.py 2>&1 > grid_deformation.txt
$MY_MPI_EXEC -n 1 python test_internal_adjoints.py 2>&1 > internal_adjoints.txt
$MY_MPI_EXEC -n $MY_NPROCS python 1_test_DV_derivs.py 2>&1 > DV_derivs.txt
