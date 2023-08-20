from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"comm size = {comm.size}")
print(f"processor {comm.rank}")
