import numpy as np, importlib, os
from mpi4py import MPI
from tacs import caps2tacs

# from funtofem import TransferScheme
from pyfuntofem.model import (
    FUNtoFEMmodel,
    Scenario,
    Body,
    Function,
    Variable,
)
from pyfuntofem.interface import SolverManager

# check whether fun3d is available
tacs_loader = importlib.util.find_spec("tacs")
fun3d_loader = importlib.util.find_spec("fun3d")
has_tacs = tacs_loader is not None
has_fun3d = fun3d_loader is not None

if has_tacs:
    from pyfuntofem.interface import TacsSteadyInterface

if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface
    from pyfuntofem.driver import FuntofemShapeDriver, Fun3dRemote

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
fun3d_dir = os.path.join(base_dir, "meshes")
fun3d_remote = Fun3dRemote.paths(fun3d_dir)

# build the funtofem model with one body and scenario
model = FUNtoFEMmodel("wing")
wing = Body.aeroelastic("wing", boundary=2)
wing.register_to(model)
test_scenario = Scenario.steady("turbulent", steps=1000).set_temperature(
    T_ref=300.0, T_inf=300.0
)
aoa = test_scenario.get_variable("AOA")
test_scenario.include(Function.lift()).include(Function.drag())
test_scenario.include(Function.ksfailure()).include(Function.mass())
test_scenario.register_to(model)

# build the solvers and coupled driver
solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=model, comm=comm, nprocs=4, bdf_file=fun3d_remote.dat_file
)
driver = FuntofemShapeDriver.analysis(solvers, model)

# run the forward and adjoint analysis in one shot
driver.solve_forward()
driver.solve_adjoint()
