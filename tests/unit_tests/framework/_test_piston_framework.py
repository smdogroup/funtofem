import numpy as np, unittest
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    PistonInterface,
    TacsSteadyInterface,
    SolverManager,
    CommManager,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings
from tacs import TACS, elements, functions, constitutive


class CoupledFrameworkTest(unittest.TestCase):
    def _setup_model_and_driver(self):
        # Build the model
        model = FUNtoFEMmodel("model")
        wing = Body("plate", "aeroelastic", group=0, boundary=1)

        # Create structural variable
        thickness = 0.025
        svar = Variable("thickness", value=thickness, lower=0.001, upper=1.0)
        # wing.add_variable("structural", svar)

        model.add_body(wing)

        # Create a scenario to run
        steady = Scenario("steady", group=0, steps=100)
        steady.set_variable(
            "aerodynamic", name="AOA", value=5.0, lower=0.0, upper=15.0, active=True
        )

        # Add a function to the scenario
        # cl = Function("cl", analysis_type="aerodynamic")
        # steady.add_function(cl)

        ks = Function("ksfailure", analysis_type="structural")
        steady.add_function(ks)

        # Add the steady-state scenario
        model.add_scenario(steady)

        # Instantiate a test solver for the flow and structures
        comm = MPI.COMM_WORLD

        n_tacs_procs = 1
        world_rank = comm.Get_rank()
        if world_rank < n_tacs_procs:
            color = 55
            key = world_rank
        else:
            color = MPI.UNDEFINED
            key = world_rank
        tacs_comm = comm.Split(color, key)

        qinf = 101325.0
        M = 1.5
        U_inf = 411
        x0 = np.array([0, 0, 0])
        alpha = 5.0
        length_dir = np.array(
            [np.cos(alpha * np.pi / 180), 0, np.sin(alpha * np.pi / 180)]
        )
        width_dir = np.array([0, 1, 0])
        L = 1.2
        nL = 10
        w = 1.2
        nw = 20

        # create solver and comm manager
        solvers = SolverManager(comm)
        solvers.flow = PistonInterface(
            comm, model, qinf, M, U_inf, x0, length_dir, width_dir, L, w, nL, nw
        )

        # create a comm manager
        comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)

        assembler = None
        if world_rank < n_tacs_procs:
            assembler = OneraPlate(tacs_comm)
        solvers.structural = TacsSteadyInterface(comm, model, assembler=assembler)

        # L&D transfer options
        transfer_settings = TransferSettings(npts=10, beta=10, isym=1)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers,
            comm_manager=comm_manager,
            transfer_settings=transfer_settings,
            model=model,
        )
        # model.print_summary()

        return model, driver

    def test_model_derivatives(self):
        model, driver = self._setup_model_and_driver()

        # Check whether to use the complex-step method or not
        complex_step = False
        epsilon_flow = 1e-8
        epsilon_struct = 1e-6
        rtol = 1e-5
        if TransferScheme.dtype == complex:
            complex_step = True
            epsilon_flow = 1e-30
            epsilon_struct = 1e-30
            rtol = 1e-9

        # Manual test of the disciplinary solvers
        scenario = model.scenarios[0]
        bodies = model.bodies
        solvers = driver.solvers

        fail = solvers.flow.test_adjoint(
            "flow",
            scenario,
            bodies,
            epsilon=epsilon_flow,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        fail = solvers.structural.test_adjoint(
            "structural",
            scenario,
            bodies,
            epsilon=epsilon_struct,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        return

    def test_coupled_derivatives(self):
        model, driver = self._setup_model_and_driver()

        # Check whether to use the complex-step method or not
        complex_step = False
        epsilon = 1e-9
        rtol = 1e-4
        if TransferScheme.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        driver.solve_forward()

        # Get the functions
        functions = model.get_functions()
        variables = model.get_variables()

        # Store the function values
        fvals_init = []
        for func in functions:
            fvals_init.append(func.value)

        # Solve the adjoint and get the function gradients
        driver.solve_adjoint()
        grads = model.get_function_gradients()

        # Set the new variable values
        if complex_step:
            variables[0].value = variables[0].value + 1j * epsilon
            model.set_variables(variables)
        else:
            variables[0].value = variables[0].value + epsilon
            model.set_variables(variables)

        driver.solve_forward()

        # Store the function values
        fvals = []
        for func in functions:
            fvals.append(func.value)

        if complex_step:
            deriv = fvals[0].imag / epsilon

            rel_error = (deriv - grads[0][0]) / deriv
            pass_ = False
            if driver.comm.rank == 0:
                pass_ = abs(rel_error) < rtol
                print("Approximate gradient  = ", deriv.real)
                print("Adjoint gradient      = ", grads[0][0].real)
                print("Relative error        = ", rel_error.real)
                print("Pass flag             = ", pass_)

            pass_ = driver.comm.bcast(pass_, root=0)
            assert pass_
        else:
            deriv = (fvals[0] - fvals_init[0]) / epsilon

            rel_error = (deriv - grads[0][0]) / deriv
            pass_ = False
            if driver.comm.rank == 0:
                pass_ = abs(rel_error) < rtol
                print("Approximate gradient  = ", deriv)
                print("Adjoint gradient      = ", grads[0][0])
                print("Relative error        = ", rel_error)
                print("Pass flag             = ", pass_)

            pass_ = driver.comm.bcast(pass_, root=0)
            # assert pass_

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
