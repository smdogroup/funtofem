import numpy as np, importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from pyfuntofem.model import (
    FUNtoFEMmodel,
    Scenario,
    Body,
    Function,
)
from pyfuntofem.interface import SolverManager, TacsSteadyInterface

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface
    from pyfuntofem.driver import FuntofemShapeDriver

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_file = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")

nprocs = comm.Get_size()

# build the funtofem model with one body and scenario
model = FUNtoFEMmodel("wing")
wing = Body.aeroelastic("wing", boundary=2)
wing.register_to(model)
test_scenario = Scenario.steady("turbulent", steps=5000).set_temperature(
    T_ref=300.0, T_inf=300.0
)
test_scenario.adjoint_steps = 2000
# aoa = test_scenario.get_variable("AOA")
test_scenario.include(Function.lift()).include(Function.drag())
test_scenario.register_to(model)

# build the solvers and coupled driver
solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(model, comm, nprocs, bdf_file)

# comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
driver = FuntofemShapeDriver.analysis(solvers, model)

# run the forward and adjoint analysis in one shot
driver.solve_forward()
driver.solve_adjoint()
