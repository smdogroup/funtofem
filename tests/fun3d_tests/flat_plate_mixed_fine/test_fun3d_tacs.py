import numpy as np, unittest, importlib, os
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
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface

np.random.seed(1234567)
comm = MPI.COMM_WORLD

results_folder = os.path.join(os.getcwd(), "results")
if comm.rank == 0:
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


class TestFun3dTacs(unittest.TestCase):
    FILENAME = "fun3d-tacs-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def _build_assembler(self):
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
            bdf_file = os.path.join(os.getcwd(), "meshes", "tacs_aero.bdf")
            mesh.scanBDFFile(bdf_file)

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
        test_scenario = Scenario.steady("laminar", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.0e4
        )

        assembler, tacs_comm = self._build_assembler()
        solvers.structural = TacsSteadyInterface(
            comm,
            model,
            assembler,
            thermal_index=3,
            tacs_comm=tacs_comm,
        )
        # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aeroelastic",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_turbulent_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("turbulent", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.0e4
        )

        assembler, tacs_comm = self._build_assembler()
        solvers.structural = TacsSteadyInterface(
            comm,
            model,
            assembler,
            thermal_index=3,
            tacs_comm=tacs_comm,
        )

        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-turbulent-aeroelastic",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_laminar_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.temperature()).include(Function.lift()).include(
            Function.drag()
        )
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")

        assembler, tacs_comm = self._build_assembler()
        solvers.structural = TacsSteadyInterface(
            comm,
            model,
            assembler,
            thermal_index=3,
            tacs_comm=tacs_comm,
        )
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aerothermal",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_turbulent_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady(
            "turbulent",
            steps=500,
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.temperature()).include(Function.lift()).include(
            Function.drag()
        )
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")

        assembler, tacs_comm = self._build_assembler()
        solvers.structural = TacsSteadyInterface(
            comm,
            model,
            assembler,
            thermal_index=3,
            tacs_comm=tacs_comm,
        )
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-turbulent-aerothermal",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

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
        test_scenario = Scenario.steady(
            "laminar",
            steps=500,
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.temperature()
        ).include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.0e4
        )

        assembler, tacs_comm = self._build_assembler()
        solvers.structural = TacsSteadyInterface(
            comm,
            model,
            assembler,
            thermal_index=3,
            tacs_comm=tacs_comm,
        )
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aerothermolastic",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_turbulent_aerothermoelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermoelastic("plate", boundary=6).relaxation(
            AitkenRelaxation()
        )
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady(
            "turbulent",
            steps=500,
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.temperature()
        ).include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.0e4
        )

        assembler, tacs_comm = self._build_assembler()
        solvers.structural = TacsSteadyInterface(
            comm,
            model,
            assembler,
            thermal_index=3,
            tacs_comm=tacs_comm,
        )
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-turbulent-aerothermolastic",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_laminar_aeroelastic_noskinfric(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady(
            "laminar_noskinfric", steps=500
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.0e4
        )

        assembler, tacs_comm = self._build_assembler()
        solvers.structural = TacsSteadyInterface(
            comm,
            model,
            assembler,
            thermal_index=3,
            tacs_comm=tacs_comm,
        )

        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aeroelastic-noskinfric",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_euler_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("euler", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.0e4
        )

        assembler, tacs_comm = self._build_assembler()
        solvers.structural = TacsSteadyInterface(
            comm,
            model,
            assembler,
            thermal_index=3,
            tacs_comm=tacs_comm,
        )

        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-euler-aeroelastic",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def __del__(self):
        # close the file handle on deletion of the object
        try:
            if comm.rank == 0:
                self.file_hdl.close()
        except:
            pass


if __name__ == "__main__":
    # open and close the file to reset it
    if comm.rank == 0:
        open(TestFun3dTacs.FILEPATH, "w").close()

    unittest.main()
