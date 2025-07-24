import unittest, importlib, numpy as np, os, sys
from funtofem import *
from mpi4py import MPI

np.random.seed(1234567)
comm = MPI.COMM_WORLD

tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "input_files", "simple_naca_wing.csm")
dat_filepath = os.path.join(base_dir, "input_files", "simple_naca_wing.dat")
results_folder, _ = make_test_directories(comm, base_dir)
caps_work = os.path.join(base_dir, "tmp")

if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs

# check if we're in github to run only online vs offline tests
in_github_workflow = bool(os.getenv("GITHUB_ACTIONS"))
# in_github_workflow = True
optional = True  # whether to run optional tests

# if comm.rank == 0:
#     if not os.path.exists(caps_work):
#         os.mkdir(caps_work)

# TACS unsteady shape is correctly not working:
# likely needs to be updated to behave similarly to steady.


# @unittest.skipIf(
#     tacs_loader is None or caps_loader is None,
#     "skipping test using caps2tacs if caps or tacs are unavailable",
# )
# class TestTacsUnsteadyShapeDriver(unittest.TestCase):
#     N_PROCS = 2
#     FILENAME = "tacs_unsteady_shape_driver.txt"
#     FILEPATH = os.path.join(results_folder, FILENAME)

#     @unittest.skipIf(in_github_workflow, "this is an offline test")
#     def test_unsteady_shape_aeroelastic(self):
#         # make the funtofem and tacs model
#         caps_name = "shape-steady-AE"
#         caps_path = os.path.join(caps_work, caps_name)
#         f2f_model = FUNtoFEMmodel("wing")
#         tacs_model = caps2tacs.TacsModel.build(
#             csm_file=csm_path, comm=comm, problem_name=caps_path
#         )
#         tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
#             edge_pt_min=15,
#             edge_pt_max=20,
#             global_mesh_size=0.1,
#             max_surf_offset=0.01,
#             max_dihedral_angle=5,
#         ).register_to(
#             tacs_model
#         )
#         f2f_model.structural = tacs_model

#         # build a body which we will register variables to
#         wing = Body.aeroelastic("wing")

#         # setup the material and shell properties
#         aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

#         nribs = int(tacs_model.get_config_parameter("nribs"))
#         nspars = int(tacs_model.get_config_parameter("nspars"))

#         for irib in range(1, nribs + 1):
#             caps2tacs.ShellProperty(
#                 caps_group=f"rib{irib}", material=aluminum, membrane_thickness=0.05
#             ).register_to(tacs_model)
#         for ispar in range(1, nspars + 1):
#             caps2tacs.ShellProperty(
#                 caps_group=f"spar{ispar}", material=aluminum, membrane_thickness=0.05
#             ).register_to(tacs_model)
#         caps2tacs.ShellProperty(
#             caps_group="OML", material=aluminum, membrane_thickness=0.03
#         ).register_to(tacs_model)

#         # register any shape variables to the wing which are auto-registered to tacs model
#         Variable.shape(name="rib_a1").set_bounds(
#             lower=0.4, value=1.0, upper=1.6
#         ).register_to(wing)

#         # register the wing body to the model
#         wing.register_to(f2f_model)

#         # add remaining information to tacs model
#         caps2tacs.PinConstraint("root").register_to(tacs_model)

#         # make a funtofem scenario
#         test_scenario = Scenario.unsteady("test", steps=10).include(Function.mass())
#         integration_settings = TacsIntegrationSettings(
#             dt=0.001, num_steps=test_scenario.steps
#         )
#         test_scenario.include(integration_settings).register_to(f2f_model)

#         solvers = SolverManager(comm)
#         solvers.flow = TestAerodynamicSolver(comm, f2f_model)
#         aero_driver = TestAeroOnewayDriver(solvers, f2f_model)
#         transfer_settings = TransferSettings(npts=200, beta=0.5)

#         # setup the tacs model
#         tacs_model.setup()

#         tacs_driver = OnewayStructDriver.prime_loads(
#             aero_driver, transfer_settings=transfer_settings, nprocs=2
#         )

#         max_rel_error = TestResult.derivative_test(
#             "testaero=>tacs_shape_unsteady-aeroelastic",
#             f2f_model,
#             tacs_driver,
#             TestTacsUnsteadyShapeDriver.FILEPATH,
#             complex_mode=False,
#             epsilon=1e-4,
#         )
#         rtol = 1e-4
#         self.assertTrue(max_rel_error < rtol)
#         return

#     @unittest.skipIf(in_github_workflow, "this is an offline test")
#     def test_unsteady_shape_aerothermal(self):
#         # make the funtofem and tacs model
#         caps_name = "shape-steady-AT"
#         caps_path = os.path.join(caps_work, caps_name)
#         f2f_model = FUNtoFEMmodel("wing")
#         tacs_model = caps2tacs.TacsModel.build(
#             csm_file=csm_path, comm=comm, problem_name=caps_path
#         )
#         tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
#             edge_pt_min=15,
#             edge_pt_max=20,
#             global_mesh_size=0.1,
#             max_surf_offset=0.01,
#             max_dihedral_angle=5,
#         ).register_to(
#             tacs_model
#         )
#         f2f_model.structural = tacs_model

#         # build a body which we will register variables to
#         wing = Body.aerothermal("wing")

#         # setup the material and shell properties
#         aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

#         nribs = int(tacs_model.get_config_parameter("nribs"))
#         nspars = int(tacs_model.get_config_parameter("nspars"))

#         for irib in range(1, nribs + 1):
#             caps2tacs.ShellProperty(
#                 caps_group=f"rib{irib}", material=aluminum, membrane_thickness=0.05
#             ).register_to(tacs_model)
#         for ispar in range(1, nspars + 1):
#             caps2tacs.ShellProperty(
#                 caps_group=f"spar{ispar}", material=aluminum, membrane_thickness=0.05
#             ).register_to(tacs_model)
#         caps2tacs.ShellProperty(
#             caps_group="OML", material=aluminum, membrane_thickness=0.03
#         ).register_to(tacs_model)

#         # register any shape variables to the wing which are auto-registered to tacs model
#         Variable.shape(name="rib_a1").set_bounds(
#             lower=0.4, value=1.0, upper=1.6
#         ).register_to(wing)

#         # register the wing body to the model
#         wing.register_to(f2f_model)

#         # add remaining information to tacs model
#         caps2tacs.PinConstraint("root").register_to(tacs_model)
#         caps2tacs.TemperatureConstraint("midplane", temperature=0).register_to(
#             tacs_model
#         )

#         # make a funtofem scenario
#         test_scenario = Scenario.unsteady("test", steps=10).include(Function.mass())
#         integration_settings = TacsIntegrationSettings(
#             dt=0.001, num_steps=test_scenario.steps
#         )
#         test_scenario.include(integration_settings).register_to(f2f_model)

#         solvers = SolverManager(comm)
#         solvers.flow = TestAerodynamicSolver(comm, f2f_model)
#         aero_driver = TestAeroOnewayDriver(solvers, f2f_model)
#         transfer_settings = TransferSettings(npts=200, beta=0.5)

#         # setup the tacs model
#         tacs_model.setup()

#         tacs_driver = OnewayStructDriver.prime_loads(
#             aero_driver, transfer_settings=transfer_settings, nprocs=2
#         )

#         max_rel_error = TestResult.derivative_test(
#             "testaero=>tacs_shape_unsteady-aerothermal",
#             f2f_model,
#             tacs_driver,
#             TestTacsUnsteadyShapeDriver.FILEPATH,
#             complex_mode=False,
#             epsilon=1e-4,
#         )
#         rtol = 1e-4
#         self.assertTrue(max_rel_error < rtol)
#         return

#     @unittest.skipIf(in_github_workflow, "this is an offline test")
#     def test_unsteady_shape_aerothermoelastic(self):
#         # make the funtofem and tacs model
#         caps_name = "shape-steady-ATE"
#         caps_path = os.path.join(caps_work, caps_name)
#         f2f_model = FUNtoFEMmodel("wing")
#         tacs_model = caps2tacs.TacsModel.build(
#             csm_file=csm_path, comm=comm, problem_name=caps_path
#         )
#         tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
#             edge_pt_min=15,
#             edge_pt_max=20,
#             global_mesh_size=0.1,
#             max_surf_offset=0.01,
#             max_dihedral_angle=5,
#         ).register_to(
#             tacs_model
#         )
#         f2f_model.structural = tacs_model

#         # build a body which we will register variables to
#         wing = Body.aerothermoelastic("wing")

#         # setup the material and shell properties
#         aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

#         nribs = int(tacs_model.get_config_parameter("nribs"))
#         nspars = int(tacs_model.get_config_parameter("nspars"))

#         for irib in range(1, nribs + 1):
#             caps2tacs.ShellProperty(
#                 caps_group=f"rib{irib}", material=aluminum, membrane_thickness=0.05
#             ).register_to(tacs_model)
#         for ispar in range(1, nspars + 1):
#             caps2tacs.ShellProperty(
#                 caps_group=f"spar{ispar}", material=aluminum, membrane_thickness=0.05
#             ).register_to(tacs_model)
#         caps2tacs.ShellProperty(
#             caps_group="OML", material=aluminum, membrane_thickness=0.03
#         ).register_to(tacs_model)

#         # register any shape variables to the wing which are auto-registered to tacs model
#         Variable.shape(name="rib_a1").set_bounds(
#             lower=0.4, value=1.0, upper=1.6
#         ).register_to(wing)

#         # register the wing body to the model
#         wing.register_to(f2f_model)

#         # add remaining information to tacs model
#         caps2tacs.PinConstraint("root").register_to(tacs_model)
#         caps2tacs.TemperatureConstraint("midplane", temperature=0).register_to(
#             tacs_model
#         )

#         # make a funtofem scenario
#         test_scenario = Scenario.unsteady("test", steps=10).include(Function.mass())
#         integration_settings = TacsIntegrationSettings(
#             dt=0.001, num_steps=test_scenario.steps
#         )
#         test_scenario.include(integration_settings).register_to(f2f_model)

#         solvers = SolverManager(comm)
#         solvers.flow = TestAerodynamicSolver(comm, f2f_model)
#         aero_driver = TestAeroOnewayDriver(solvers, f2f_model)
#         transfer_settings = TransferSettings(npts=200, beta=0.5)

#         # setup the tacs model
#         tacs_model.setup()

#         tacs_driver = OnewayStructDriver.prime_loads(
#             aero_driver, transfer_settings=transfer_settings, nprocs=2
#         )

#         max_rel_error = TestResult.derivative_test(
#             "testaero=>tacs_shape_unsteady-aerothermoelastic",
#             f2f_model,
#             tacs_driver,
#             TestTacsUnsteadyShapeDriver.FILEPATH,
#             complex_mode=False,
#             epsilon=1e-4,
#         )
#         rtol = 1e-4
#         self.assertTrue(max_rel_error < rtol)
#         return


# if __name__ == "__main__":
#     if tacs_loader is not None and caps_loader is not None:
#         if comm.rank == 0:
#             open(TestTacsUnsteadyShapeDriver.FILEPATH, "w").close()

    # unittest.main()
    # tester = TestTacsUnsteadyShapeDriver()
    # tester.test_unsteady_shape_aeroelastic()
