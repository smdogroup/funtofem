import unittest, os, numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    PistonInterface,
    PistonTheoryFlow,
    PistonTheoryGrid,
    TacsSteadyInterface,
    SolverManager,
    CommManager,
    TestStructuralSolver,
    TestResult,
)
from bdf_test_utils import elasticity_callback
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings
from tacs import TACS, constitutive, elements

comm = MPI.COMM_WORLD
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex


class TestSteadyPistonTheory(unittest.TestCase):
    FILENAME = "steady-piston-tacs.txt"

    def test_model_derivatives(self):
        model = FUNtoFEMmodel("model")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness").set_bounds(value=0.025).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=10)
        test_scenario.set_variable("aerodynamic", name="AOA", value=5.0)
        test_scenario.include(Function.ksfailure()).register_to(model)

        # create solver and comm manager
        solvers = SolverManager(comm)
        piston_grid = PistonTheoryGrid(
            origin=np.array([0, 0, 0]),
            length_dir=PistonTheoryGrid.aoa_dir(aoa=5.0),
            width_dir=np.array([0, 1, 0]),
            length=1.2,
            width=1.2,
            n_length=10,
            n_width=20,
        )
        piston_flow = PistonTheoryFlow(qinf=101325.0, mach=1.5, U_inf=411)
        solvers.flow = PistonInterface(comm, model, piston_grid, piston_flow)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model,
            comm,
            nprocs=1,
            bdf_file=os.path.join(os.getcwd(), "input_files", "test_bdf_file.bdf"),
            callback=elasticity_callback,
        )

        # instantiate the driver to auto-construct some settings in the body
        FUNtoFEMnlbgs(
            solvers,
            transfer_settings=TransferSettings(npts=10, beta=10, isym=1),
            model=model,
        )

        # Check whether to use the complex-step method or not
        h = 1e-30
        rtol = 1e-9 if complex_mode else 1e-5

        fail = solvers.flow.test_adjoint(
            "flow",
            test_scenario,
            model.bodies,
            epsilon=h if complex_mode else 1e-8,
            complex_step=complex_mode,
            rtol=rtol,
        )
        assert fail == False

        fail = solvers.structural.test_adjoint(
            "structural",
            test_scenario,
            model.bodies,
            epsilon=h if complex_mode else 1e-6,
            complex_step=complex_mode,
            rtol=rtol,
        )
        assert fail == False

        return

    def test_with_onera(self):
        model = FUNtoFEMmodel("model")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness").set_bounds(value=0.025).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=10)
        test_scenario.set_variable("aerodynamic", name="AOA", value=5.0)
        test_scenario.include(Function.ksfailure()).register_to(model)

        # create solver and comm manager
        solvers = SolverManager(comm)
        piston_grid = PistonTheoryGrid(
            origin=np.array([0, 0, 0]),
            length_dir=PistonTheoryGrid.aoa_dir(aoa=5.0),
            width_dir=np.array([0, 1, 0]),
            length=1.2,
            width=1.2,
            n_length=10,
            n_width=20,
        )
        piston_flow = PistonTheoryFlow(qinf=101325.0, mach=1.5, U_inf=411)
        solvers.flow = PistonInterface(comm, model, piston_grid, piston_flow)

        n_tacs_procs = 1
        world_rank = comm.Get_rank()
        if world_rank < n_tacs_procs:
            color = 55
            key = world_rank
        else:
            color = MPI.UNDEFINED
            key = world_rank
        tacs_comm = comm.Split(color, key)
        assembler = OneraPlate(tacs_comm)
        solvers.structural = TacsSteadyInterface(comm, model, assembler)
        comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers,
            comm_manager,
            transfer_settings=TransferSettings(npts=10, beta=10, isym=1),
            model=model,
        )

        rtol = 1e-8 if complex_mode else 1e-5
        max_rel_error = TestResult.derivative_test(
            "piston+tacs-steady-onera",
            model,
            driver,
            TestSteadyPistonTheory.FILENAME,
            complex_mode,
        )
        self.assertTrue(max_rel_error < rtol)
        return


def OneraPlate(tacs_comm):
    # Set the creator object
    ndof = 6
    creator = TACS.Creator(tacs_comm, ndof)

    if tacs_comm.rank == 0:
        # Create the elements
        nx = 10
        ny = 10

        # Set the nodes
        nnodes = (nx + 1) * (ny + 1)
        nelems = nx * ny
        nodes = np.arange(nnodes).reshape((nx + 1, ny + 1))

        conn = []
        for j in range(ny):
            for i in range(nx):
                # Append the node locations
                conn.append(
                    [nodes[i, j], nodes[i + 1, j], nodes[i, j + 1], nodes[i + 1, j + 1]]
                )

        # Set the node pointers
        conn = np.array(conn, dtype=np.intc).flatten()
        ptr = np.arange(0, 4 * nelems + 1, 4, dtype=np.intc)
        elem_ids = np.zeros(nelems, dtype=np.intc)
        creator.setGlobalConnectivity(nnodes, ptr, conn, elem_ids)

        # Set the boundary conditions - fixed on the root
        bcnodes = np.array(nodes[:, 0], dtype=np.intc)
        creator.setBoundaryConditions(bcnodes)

        root_chord = 0.8
        semi_span = 1.2
        taper_ratio = 0.56
        sweep = 26.7  # degrees

        # Set the node locations
        Xpts = np.zeros(3 * nnodes, TACS.dtype)
        x = np.linspace(0, 1, nx + 1)
        y = np.linspace(0, 1, ny + 1)
        for j in range(ny + 1):
            for i in range(nx + 1):
                c = root_chord * (1.0 - y[j]) + root_chord * taper_ratio * y[j]
                xoff = 0.25 * root_chord + semi_span * y[j] * np.tan(
                    sweep * np.pi / 180.0
                )

                Xpts[3 * nodes[i, j]] = xoff + c * (x[i] - 0.25)
                Xpts[3 * nodes[i, j] + 1] = semi_span * y[j]

        # Set the node locations
        creator.setNodes(Xpts)

    # Set the material properties
    props = constitutive.MaterialProperties(rho=2570.0, E=70e9, nu=0.3, ys=350e6)

    # Set constitutive properties
    t = 0.025
    tnum = 0
    maxt = 0.015
    mint = 0.015
    con = constitutive.IsoShellConstitutive(props, t=t, tNum=tnum)

    # Create a transformation object
    transform = elements.ShellNaturalTransform()

    # Create the element object
    element = elements.Quad4Shell(transform, con)

    # Set the elements
    elems = [element]
    creator.setElements(elems)

    # Create TACS Assembler object from the mesh loader
    assembler = creator.createTACS()

    return assembler


if __name__ == "__main__":
    unittest.main()
