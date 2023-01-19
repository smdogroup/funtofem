import numpy as np, unittest
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
    Fun3dInterface,
    TacsSteadyInterface,
    SolverManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

np.random.seed(1234567)


class TestFun3dTacsLaminar(unittest.TestCase):
    def _complex_step_check(self, test_name, model, driver):

        # determine the number of functions and variables
        nfunctions = len(model.get_functions())
        nvariables = len(model.get_variables())
        func_names = [func.name for func in model.get_functions()]

        # generate random covariant tensor, an input space curve tangent dx/ds for design vars
        dxds = np.random.rand(nvariables)

        # solve the adjoint
        driver.solvers.make_flow_real()
        driver.solve_forward()
        driver.solve_adjoint()
        gradients = model.get_function_gradients()

        # compute the adjoint total derivative df/ds = df/dx * dx/ds
        adjoint_TD = np.zeros((nfunctions))
        for ifunc in range(nfunctions):
            for ivar in range(nvariables):
                adjoint_TD[ifunc] += gradients[ifunc][ivar].real * dxds[ivar]

        # perform complex step method
        driver.solvers.make_flow_complex()
        epsilon = 1e-30
        rtol = 1e-7
        variables = model.get_variables()

        # perturb the design vars by x_pert = x + 1j * h * dx/ds
        for ivar in range(nvariables):
            variables[ivar].value += 1j * epsilon * dxds[ivar]

        # run the complex step method
        driver.solve_forward()
        functions = model.get_functions()

        # compute the complex step total derivative df/ds = Im{f(x+ih * dx/ds)}/h for each func
        complex_TD = np.zeros((nfunctions))
        for ifunc in range(nfunctions):
            complex_TD[ifunc] += functions[ifunc].value.imag / epsilon

        # compute rel error between adjoint & complex step for each function
        rel_error = [
            (adjoint_TD[ifunc] - complex_TD[ifunc]) / complex_TD[ifunc]
            for ifunc in range(nfunctions)
        ]
        rel_error = [_.real for _ in rel_error]

        # make test results object and write it to file
        file_hdl = open("fun3d-tacs-laminar.txt", "a")
        TestResult(test_name, func_names, complex_TD, adjoint_TD, rel_error).write(
            file_hdl
        ).report()

        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))

        self.assertTrue(max_rel_error < rtol)

    def _build_assembler(self, comm):

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
            mesh.scanBDFFile("tacs_aero.bdf")

            # Set the element
            mesh.setElement(0, element_plate)

            # Create the assembler object
            assembler = mesh.createTACS(varsPerNode)

        return assembler

    def test_laminar_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar1", steps=200).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e4)

        assembler = self._build_assembler(comm)
        solvers.structural = TacsSteadyInterface(
            comm, model, assembler, gen_output=None, thermal_index=3
        )
        # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        self._complex_step_check("fun3d+tacs-laminar-aeroelastic", model, driver)

    def test_laminar_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar2", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.temperature()).include(Function.lift()).include(
            Function.drag()
        )
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model)

        assembler = self._build_assembler(comm)
        solvers.structural = TacsSteadyInterface(
            comm, model, assembler, gen_output=None, thermal_index=3
        )
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        self._complex_step_check("fun3d+tacs-laminar-aerothermal", model, driver)

    def test_laminar_aerothermoelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermoelastic("plate", boundary=6).relaxation(
            AitkenRelaxation()
        )
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar2", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.temperature()
        ).include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e4)

        assembler = self._build_assembler(comm)
        solvers.structural = TacsSteadyInterface(
            comm, model, assembler, gen_output=None, thermal_index=3
        )
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        self._complex_step_check("fun3d+tacs-laminar-aerothermoelastic", model, driver)

    def __del__(self):
        # close the file handle on deletion of the object
        try:
            self.file_hdl.close()
        except:
            pass


if __name__ == "__main__":
    # open and close the file to reset it
    open("fun3d-tacs-laminar.txt", "w").close()

    full_test = True
    if full_test:
        unittest.main()
    else:
        tester = TestFun3dTacsLaminar()
        tester.test_laminar_aeroelastic()
