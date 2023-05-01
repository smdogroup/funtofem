import os, importlib
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import SolverManager, TestResult

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface

# from bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
# bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")

# select the node and direction to use in the test
# repeat the whole test for different nodes and directions (changed here and in fun3d-folder/Flow/perturb.input file)
pert_ind = 1
pert_dir = 2  # 0 for x, 1 for y, 2 for z

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


@unittest.skipIf(
    not complex_mode, "only testing coordinate derivatives with complex step"
)
class TestFun3dAeroCoords(unittest.TestCase):
    FILENAME = "fun3daero-coords.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_steady_aero_aeroelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("turbulent", steps=500).include(Function.lift())
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")
        rtol = 1e-9

        """complex step test over FUN3D aero coordinate derivatives"""
        bodies = model.bodies
        nf = 1
        func_names = [func.full_name for func in model.get_functions()]

        # random contravariant
        daeroX_ds = np.random.rand(plate.aero_X.shape[0])

        # choose the id to perturb matching the perturb.input
        aero_id = plate.aero_id
        found_ind = False

        for i, ind in enumerate(aero_id):
            daeroX_ds[3 * i : 3 * i + 3] = 0.0
            if ind == pert_ind:
                daeroX_ds[3 * i + pert_dir] = 1.0
                found_ind = True
        assert found_ind

        daeroX_ds_row = np.reshape(daeroX_ds, newshape=(1, daeroX_ds.shape[0]))

        # first we do the adjoint analysis in real mode
        # forward analysis first
        for scenario in model.scenarios:
            # set functions and variables
            solvers.flow.set_variables(scenario, bodies)
            solvers.flow.set_functions(scenario, bodies)

            # run the forward analysis via iterate
            solvers.flow.initialize(scenario, bodies)
            for step in range(1, scenario.steps + 1):
                solvers.flow.iterate(scenario, bodies, step=0)
            solvers.flow.post(scenario, bodies)

            # get functions to store the function values into the model
            solvers.flow.get_functions(scenario, bodies)

        # adjoint analysis in real mode
        for scenario in model.scenarios:
            # set functions and variables
            solvers.flow.set_variables(scenario, bodies)
            solvers.flow.set_functions(scenario, bodies)

            # zero all coupled adjoint variables in the body
            for body in bodies:
                body.initialize_adjoint_variables(scenario)

            # initialize, run, and do post adjoint
            solvers.flow.initialize_adjoint(scenario, bodies)
            for step in range(1, scenario.steps + 1):
                solvers.flow.iterate_adjoint(scenario, bodies, step=step)
            solvers.flow.get_coordinate_derivatives(scenario, bodies, step=0)
            solvers.flow.post_adjoint(scenario, bodies)

            # call get function gradients to store the gradients w.r.t. aero DVs from FUN3D
            solvers.flow.get_function_gradients(scenario, bodies)

        # compute adjoint total derivative
        full_aero_shape_term = []
        for scenario in model.scenarios:
            aero_shape_term = body.get_aero_coordinate_derivatives(scenario)
            aero_id = body.aero_id
            for i, ind in enumerate(aero_id):
                if ind == pert_ind:
                    deriv = aero_shape_term[3 * i + pert_dir]
                    print(f"deriv = {deriv}", flush=True)
            full_aero_shape_term.append(aero_shape_term)
        full_aero_shape_term = np.concatenate(full_aero_shape_term, axis=1)
        # add in struct coordinate derivatives of this scenario
        local_dfds = np.zeros((1))
        local_dfds[0] = float((daeroX_ds_row @ full_aero_shape_term).real)
        adjoint_derivs = np.zeros((1))
        comm.Reduce(local_dfds, adjoint_derivs, root=0)
        adjoint_derivs = comm.bcast(adjoint_derivs, root=0)
        adjoint_derivs = [adjoint_derivs[i] for i in range(nf)]

        i_lift = model.get_functions()[0].value.real

        # finite diff analysis with perturbed aero coordinates
        # h = 1e-5
        h = 1e-30
        # plate.aero_X += 1j * h * daeroX_ds
        # convert to a complex flow solver now
        solvers.make_flow_complex()
        # it auto-multiplies by h = 1e-30 based on complex_epsilon
        # in the perturb.input file for imaginary perturbations
        # plate.aero_X += 1j * daeroX_ds # * h

        # plate.aero_X += daeroX_ds * h
        # forward analysis complex-mode
        for scenario in model.scenarios:
            # set functions and variables
            solvers.flow.set_variables(scenario, bodies)
            solvers.flow.set_functions(scenario, bodies)

            # run the forward analysis via iterate
            solvers.flow.initialize(scenario, bodies)
            # print(f"aeroX = {plate.aero_X}",flush=True)
            for step in range(1, scenario.steps + 1):
                solvers.flow.iterate(scenario, bodies, step=0)
            solvers.flow.post(scenario, bodies)

            # get functions to store the function values into the model
            solvers.flow.get_functions(scenario, bodies)

        # check deformed volume grid coords here
        f_lift = model.get_functions()[0].value
        print(f"complex lift = {f_lift}", flush=True)

        complex_step_derivs = np.array(
            [func.value.imag / h for func in model.get_functions()]
        )

        rel_error = [
            TestResult.relative_error(complex_step_derivs[i], adjoint_derivs[i])
            for i in range(nf)
        ]

        # make test results object and write it to file
        file_hdl = open(TestFun3dAeroCoords.FILEPATH, "a") if comm.rank == 0 else None
        dir_list = ["x", "y", "z"]
        dir_str = dir_list[pert_dir]
        TestResult(
            f"fun3d aero coords steady-aeroelastic, node {pert_ind}, {dir_str} dir",
            func_names,
            complex_step_derivs,
            adjoint_derivs,
            rel_error,
            comm=comm,
        ).write(file_hdl).report()

        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))

        assert abs(max_rel_error) < rtol
        return

    def test_steady_aero_aerothermoelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("turbulent", steps=500).include(Function.lift())
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")
        rtol = 1e-9

        """complex step test over FUN3D aero coordinate derivatives"""
        bodies = model.bodies
        nf = 1
        func_names = [func.full_name for func in model.get_functions()]

        # random contravariant
        daeroX_ds = np.random.rand(plate.aero_X.shape[0])

        # choose the id to perturb matching the perturb.input
        aero_id = plate.aero_id
        found_ind = False

        for i, ind in enumerate(aero_id):
            daeroX_ds[3 * i : 3 * i + 3] = 0.0
            if ind == pert_ind:
                daeroX_ds[3 * i + pert_dir] = 1.0
                found_ind = True
        assert found_ind

        daeroX_ds_row = np.reshape(daeroX_ds, newshape=(1, daeroX_ds.shape[0]))

        # first we do the adjoint analysis in real mode
        # forward analysis first
        for scenario in model.scenarios:
            # set functions and variables
            solvers.flow.set_variables(scenario, bodies)
            solvers.flow.set_functions(scenario, bodies)

            # run the forward analysis via iterate
            solvers.flow.initialize(scenario, bodies)
            for step in range(1, scenario.steps + 1):
                solvers.flow.iterate(scenario, bodies, step=0)
            solvers.flow.post(scenario, bodies)

            # get functions to store the function values into the model
            solvers.flow.get_functions(scenario, bodies)

        # adjoint analysis in real mode
        for scenario in model.scenarios:
            # set functions and variables
            solvers.flow.set_variables(scenario, bodies)
            solvers.flow.set_functions(scenario, bodies)

            # zero all coupled adjoint variables in the body
            for body in bodies:
                body.initialize_adjoint_variables(scenario)

            # initialize, run, and do post adjoint
            solvers.flow.initialize_adjoint(scenario, bodies)
            for step in range(1, scenario.steps + 1):
                solvers.flow.iterate_adjoint(scenario, bodies, step=step)
            solvers.flow.get_coordinate_derivatives(scenario, bodies, step=0)
            solvers.flow.post_adjoint(scenario, bodies)

            # call get function gradients to store the gradients w.r.t. aero DVs from FUN3D
            solvers.flow.get_function_gradients(scenario, bodies)

        # compute adjoint total derivative
        full_aero_shape_term = []
        for scenario in model.scenarios:
            aero_shape_term = body.get_aero_coordinate_derivatives(scenario)
            aero_id = body.aero_id
            for i, ind in enumerate(aero_id):
                if ind == pert_ind:
                    deriv = aero_shape_term[3 * i + pert_dir]
                    print(f"deriv = {deriv}", flush=True)
            full_aero_shape_term.append(aero_shape_term)
        full_aero_shape_term = np.concatenate(full_aero_shape_term, axis=1)
        # add in struct coordinate derivatives of this scenario
        local_dfds = np.zeros((1))
        local_dfds[0] = float((daeroX_ds_row @ full_aero_shape_term).real)
        adjoint_derivs = np.zeros((1))
        comm.Reduce(local_dfds, adjoint_derivs, root=0)
        adjoint_derivs = comm.bcast(adjoint_derivs, root=0)
        adjoint_derivs = [adjoint_derivs[i] for i in range(nf)]

        i_lift = model.get_functions()[0].value.real

        # finite diff analysis with perturbed aero coordinates
        # h = 1e-5
        h = 1e-30
        # plate.aero_X += 1j * h * daeroX_ds
        # convert to a complex flow solver now
        solvers.make_flow_complex()
        # it auto-multiplies by h = 1e-30 based on complex_epsilon
        # in the perturb.input file for imaginary perturbations
        # plate.aero_X += 1j * daeroX_ds # * h

        # plate.aero_X += daeroX_ds * h
        # forward analysis complex-mode
        for scenario in model.scenarios:
            # set functions and variables
            solvers.flow.set_variables(scenario, bodies)
            solvers.flow.set_functions(scenario, bodies)

            # run the forward analysis via iterate
            solvers.flow.initialize(scenario, bodies)
            # print(f"aeroX = {plate.aero_X}",flush=True)
            for step in range(1, scenario.steps + 1):
                solvers.flow.iterate(scenario, bodies, step=0)
            solvers.flow.post(scenario, bodies)

            # get functions to store the function values into the model
            solvers.flow.get_functions(scenario, bodies)

        # check deformed volume grid coords here
        f_lift = model.get_functions()[0].value
        print(f"complex lift = {f_lift}", flush=True)

        complex_step_derivs = np.array(
            [func.value.imag / h for func in model.get_functions()]
        )

        rel_error = [
            TestResult.relative_error(complex_step_derivs[i], adjoint_derivs[i])
            for i in range(nf)
        ]

        # make test results object and write it to file
        file_hdl = open(TestFun3dAeroCoords.FILEPATH, "a") if comm.rank == 0 else None
        dir_list = ["x", "y", "z"]
        dir_str = dir_list[pert_dir]
        TestResult(
            f"fun3d aero coords steady-aerothermoelastic, node {pert_ind}, {dir_str} dir",
            func_names,
            complex_step_derivs,
            adjoint_derivs,
            rel_error,
            comm=comm,
        ).write(file_hdl).report()

        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))

        assert abs(max_rel_error) < rtol
        return


if __name__ == "__main__":
    # open(TestFun3dAeroCoords.FILEPATH, "w").close()
    unittest.main()
