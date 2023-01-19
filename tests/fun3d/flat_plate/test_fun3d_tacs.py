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
    CommManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

np.random.seed(1234567)


class TestFun3dTacs(unittest.TestCase):
    def _complex_step_check(self, test_name, model, driver):

        # make sure the flow is real
        driver.solvers.make_flow_real()

        # solve the adjoint
        driver.solve_forward()
        driver.solve_adjoint()
        gradients = model.get_function_gradients()
        adjoint_TD = gradients[0][0].real

        # switch to complex flow
        driver.solvers.make_flow_complex()

        # perform complex step method
        epsilon = 1e-30
        rtol = 1e-7
        variables = model.get_variables()
        variables[0].value = variables[0].value + 1j * epsilon
        driver.solve_forward()
        functions = model.get_functions()
        complex_TD = functions[0].value.imag / epsilon

        # compute rel error between adjoint & complex step
        rel_error = (adjoint_TD - complex_TD) / complex_TD
        rel_error = rel_error.real

        # make test results object and write it to file
        try:
            TestResult(test_name, complex_TD, adjoint_TD, rel_error).write(
                self.file_hdl
            )
        except:
            self.file_hdl = open("test_result.txt", "w")
            TestResult(test_name, complex_TD, adjoint_TD, rel_error).write(
                self.file_hdl
            )

        self.assertTrue(abs(rel_error) < rtol)

    def _build_assembler(self, comm, n_tacs_procs):

        world_rank = comm.Get_rank()
        if world_rank < n_tacs_procs:
            color = 55
            key = world_rank
        else:
            color = MPI.UNDEFINED
            key = world_rank
        tacs_comm = comm.Split(color, key)

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

        return assembler, tacs_comm

    def test_laminar_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=100).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure(ks_weight=50.0))
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e4)

        assembler, tacs_comm = self._build_assembler(comm, n_tacs_procs=1)
        solvers.structural = TacsSteadyInterface(
            comm, model, assembler, gen_output=None, thermal_index=3
        )
        comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers,
            comm_manager=comm_manager,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        self._complex_step_check("laminar-aeroelastic", model, driver)

    def test_laminar_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=100).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.temperature())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e4)

        assembler, tacs_comm = self._build_assembler(comm, n_tacs_procs=1)
        solvers.structural = TacsSteadyInterface(
            comm, model, assembler, gen_output=None, thermal_index=3
        )
        comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
        transfer_settings = TransferSettings(npts=20)
        driver = FUNtoFEMnlbgs(
            solvers,
            comm_manager=comm_manager,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        self._complex_step_check("laminar-aerothermal", model, driver)

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
        test_scenario = Scenario.steady("laminar", steps=100).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure(ks_weight=50.0))
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e4)

        assembler, tacs_comm = self._build_assembler(comm, n_tacs_procs=1)
        solvers.structural = TacsSteadyInterface(
            comm, model, assembler, gen_output=None, thermal_index=3
        )
        comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers,
            comm_manager=comm_manager,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        self._complex_step_check("laminar-aerothermoelastic", model, driver)


if __name__ == "__main__":
    full_test = False
    if full_test:
        unittest.main()
    else:
        tester = TestFun3dTacs()
        tester.test_laminar_aerothermal()
