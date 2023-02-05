import importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from tacs import TACS, elements, constitutive
from pyfuntofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
    AitkenRelaxation,
)
from pyfuntofem.interface import (
    TacsSteadyInterface,
    SolverManager,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface

    """
    Goal here is to start run a complex flow for funtofem_coupling.f90 internal complex step test
    Need to have fun3d compiled with the complex step test 3 line call uncommented
    """

    # build the funtofem model with one body and scenario
    model = FUNtoFEMmodel("plate")
    plate = Body.aeroelastic("plate", boundary=6).relaxation(AitkenRelaxation())
    Variable.structural("thickness").set_bounds(
        lower=0.001, value=0.1, upper=2.0
    ).register_to(plate)
    plate.register_to(model)
    test_scenario = Scenario.steady("fortran_laminar", steps=200).set_temperature(
        T_ref=300.0, T_inf=300.0
    )
    test_scenario.include(Function.lift())
    test_scenario.register_to(model)

    # build the solvers and coupled driver
    comm = MPI.COMM_WORLD
    solvers = SolverManager(comm)
    solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(qinf=1.0)

    # build a tacs communicator on one proc
    n_tacs_procs = 1
    world_rank = comm.Get_rank()
    if world_rank < n_tacs_procs:
        color = 55
        key = world_rank
    else:
        color = MPI.UNDEFINED
        key = world_rank
    tacs_comm = comm.Split(color, key)

    # build the tacs assembler of the flat plate
    assembler = None
    if comm.rank < n_tacs_procs:
        # Create the constitutvie propertes and model
        props_plate = constitutive.MaterialProperties(
            rho=4540.0,
            specific_heat=463.0,
            kappa=6.89,
            E=118e9,
            nu=0.325,
            ys=1050e6,
        )
        con_plate = constitutive.SolidConstitutive(props_plate, t=1.0, tNum=0)
        model_plate = elements.LinearThermoelasticity3D(con_plate)

        # Create the basis class
        quad_basis = elements.LinearHexaBasis()

        # Create the element
        element_plate = elements.Element3D(model_plate, quad_basis)
        varsPerNode = model_plate.getVarsPerNode()

        # Load in the mesh
        mesh = TACS.MeshLoader(tacs_comm)
        bdf_path = os.path.join(os.getcwd(), "meshes", "tacs_aero.bdf")
        mesh.scanBDFFile(bdf_path)

        # Set the element
        mesh.setElement(0, element_plate)

        # Create the assembler object
        assembler = mesh.createTACS(varsPerNode)

    solvers.structural = TacsSteadyInterface(
        comm, model, assembler, gen_output=None, thermal_index=3
    )
    # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
    transfer_settings = TransferSettings()
    driver = FUNtoFEMnlbgs(
        solvers,
        transfer_settings=transfer_settings,
        model=model,
    )

    # change the flow to complex
    driver.solvers.make_flow_complex()

    # run the complex flow
    driver.solve_forward()
