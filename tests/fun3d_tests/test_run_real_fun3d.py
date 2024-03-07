import importlib, os
from mpi4py import MPI

# os.environ["CMPLX_MODE"] = "1"

# from funtofem import TransferScheme
from tacs import TACS, elements, constitutive
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
    AitkenRelaxation,
)
from funtofem.interface import TacsSteadyInterface, SolverManager, make_test_directories
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
_, output_dir = make_test_directories(comm, base_dir)

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3d14Interface

    """
    Goal here is to start run a real flow for funtofem_coupling.f90 internal FD test
    Need to set massoud/ funtofem_internal_adjoint_test = .true. in the nml
    """

    # build the funtofem model with one body and scenario
    model = FUNtoFEMmodel("plate")
    plate = Body.aerothermoelastic("plate", boundary=6).relaxation(AitkenRelaxation())
    Variable.structural("thickness").set_bounds(
        lower=0.001, value=0.1, upper=2.0
    ).register_to(plate)
    plate.register_to(model)
    test_scenario = Scenario.steady("fortran_laminar", steps=200).set_temperature(
        T_ref=300.0, T_inf=300.0
    )
    test_scenario.include(Function.lift())
    test_scenario.set_flow_ref_vals(qinf=1.0e4)
    test_scenario.register_to(model)

    # build the solvers and coupled driver
    solvers = SolverManager(comm)
    solvers.flow = Fun3d14Interface(comm, model, fun3d_dir="meshes")

    bdf_file = os.path.join(
        base_dir, "meshes", "turbulent_miniMesh", "nastran_CAPS.dat"
    )
    solvers.structural = TacsSteadyInterface.create_from_bdf(
        model, comm, nprocs=1, bdf_file=bdf_file, prefix=output_dir
    )

    # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
    transfer_settings = TransferSettings(npts=10, elastic_scheme="meld")
    driver = FUNtoFEMnlbgs(
        solvers,
        transfer_settings=transfer_settings,
        model=model,
    )

    # change the flow to complex
    # driver.solvers.make_flow_complex()

    # run the complex flow
    driver.solve_forward()
