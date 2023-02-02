from contextlib import contextmanager
from os import devnull
import numpy as np, unittest, importlib
import os, sys
from mpi4py import MPI

from pyfuntofem.model import (
    FUNtoFEMmodel,
    Body,
    Scenario,
    Function,
    AitkenRelaxation,
    Variable,
)
from pyfuntofem.interface import (
    TestStructuralSolver,
    SolverManager,
    TestResult,
    usesFun3d,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface

results_folder = os.path.join(os.getcwd(), "results")
if not os.path.exists(results_folder): os.mkdir(results_folder)

# define a function to suppress stdout
@contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, "w")
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


class TestLaminarAeroelastic(unittest.TestCase):
    FILENAME = "full_plate.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    @usesFun3d
    def test_fun3d_interface(self):
        # build a funtofem model with one body and scenario
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
        test_scenario.include(Function.ksfailure())
        test_scenario.register_to(model)

        # suppress stdout during each FUN3D analysis
        with stdchannel_redirected(sys.stdout, os.devnull):

            # build the solvers and coupled driver
            comm = MPI.COMM_WORLD
            solvers = SolverManager(comm)
            solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e4)
            solvers.structural = TestStructuralSolver(comm, model, elastic_k=1000.0)
            # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
            transfer_settings = TransferSettings()

            # run one forward analysis of the driver to setup the fun3d interface as well and transfer schemes
            FUNtoFEMnlbgs(
                solvers,
                transfer_settings=transfer_settings,
                model=model,
            ).solve_forward()

            plate.initialize_adjoint_variables(test_scenario)

            """
            perform complex step method on the fun3d interface uA -> fA
            use the fact that adjoint evaluation psi_fA -> psi_uA exactly
            the same as backpropagation of output covariant dL/dfA -> dL/duA
            """
            # randomly generated contravariant duA/ds
            na = plate.get_num_aero_nodes()
            duAds = np.random.rand((3 * na))
            # randomly generated output covariant dL/dfA
            dLdfA = np.random.rand((3 * na))

            # perform the adjoint method on real flow mode
            real_interface = Fun3dInterface.copy_real_interface(solvers.flow)
            aero_loads_ajp = plate.get_aero_loads_ajp(test_scenario)
            aero_loads_ajp[:, 0] = dLdfA[:]

            real_interface.set_variables(test_scenario, model.bodies)
            real_interface.set_functions(test_scenario, model.bodies)

            real_interface.initialize_adjoint(test_scenario, model.bodies)
            for istep in range(test_scenario.steps):
                real_interface.iterate_adjoint(test_scenario, model.bodies, step=istep)
            print("finished iterate adjoint", flush=True)
            # real_interface.post_adjoint(test_scenario, model.bodies)
            os.chdir(real_interface.root_dir)
            aero_disps_ajp = plate.get_aero_disps_ajp(test_scenario)
            adjoint_dLds = np.sum(aero_disps_ajp[:, 0] * duAds).real

            # real_interface.post_adjoint(test_scenario, model.bodies)

            print("Adjoint test", flush=True)

            # perform the complex step method with complex fun3d flow
            h = 1e-30
            rtol = 1e-9
            aero_disps = plate.get_aero_disps(test_scenario)
            aero_disps[:] += 1j * h * duAds
            complex_interface = Fun3dInterface.copy_complex_interface(solvers.flow)

            # run the complex flow in FUN3D using Fun3dInterface
            complex_interface.set_variables(test_scenario, model.bodies)
            complex_interface.set_functions(test_scenario, model.bodies)
            complex_interface.initialize(test_scenario, model.bodies)
            print("starting complex flow iterations...")
            for istep in range(test_scenario.steps):
                complex_interface.iterate(test_scenario, model.bodies, step=istep)
            complex_interface.post(test_scenario, model.bodies)

            # compute the output contravariants dfA/ds under complex flow
            aero_loads = plate.get_aero_loads(test_scenario)
            dfAds = np.array([_.imag / h for _ in list(aero_loads)])

            # compute the complex step total derivative dL/ds = dL/dfA * dfA/ds
            complex_dLds = np.sum(dLdfA * dfAds)

        # compare the total derivatives of complex step vs adjoint & evaluate
        rel_error = (adjoint_dLds - complex_dLds) / complex_dLds
        rel_error = rel_error.real
        TestResult(
            "Fun3dInterface-Laminar-Aeroelastic",
            "None",
            complex_dLds,
            adjoint_dLds,
            rel_error,
        ).report()

        self.assertTrue(abs(rel_error) < rtol)
        return

    @usesFun3d
    def test_transfer_disps(self):
        # build a funtofem model with one body and scenario
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
        test_scenario.include(Function.lift())
        test_scenario.register_to(model)

        # suppress the standard output from fortran, fun3d
        with stdchannel_redirected(sys.stdout, os.devnull):

            # build the solvers and coupled driver
            comm = MPI.COMM_WORLD
            solvers = SolverManager(comm)
            solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e4)
            solvers.structural = TestStructuralSolver(comm, model, elastic_k=1000.0)
            # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
            transfer_settings = TransferSettings()

            # run one forward analysis of the driver to setup the fun3d interface as well and transfer schemes
            FUNtoFEMnlbgs(
                solvers,
                transfer_settings=transfer_settings,
                model=model,
            ).solve_forward()

        # start displacement transfer test
        plate.initialize_adjoint_variables(test_scenario)
        ns = plate.get_num_struct_nodes()
        na = plate.get_num_aero_nodes()
        duS_ds = np.random.rand((3 * ns))
        dL_duA = np.random.rand((3 * na))

        # adjoint computation
        plate.aero_disps_ajp[:, 0] = dL_duA[:]
        plate.transfer_disps_adjoint(test_scenario)
        dL_duS = plate.struct_disps_ajp_disps[:, 0].copy()
        adjoint_dLds = np.sum(dL_duS * duS_ds).real

        # complex step method
        h = 1e-30
        rtol = 1e-9
        struct_disps = plate.get_struct_disps(test_scenario)
        struct_disps[:] += 1j * h * duS_ds
        plate.transfer_disps(test_scenario)
        aero_disps = plate.get_aero_disps(test_scenario)
        duA_ds = np.array([_.imag / h for _ in list(aero_disps)])
        complex_dLds = np.sum(dL_duA * duA_ds).real

        # compare the total derivatives of complex step vs adjoint & evaluate
        rel_error = (adjoint_dLds - complex_dLds) / complex_dLds
        rel_error = rel_error.real
        TestResult(
            "TransferDisps-Laminar-Aeroelastic",
            "None",
            complex_dLds,
            adjoint_dLds,
            rel_error,
        ).report()

        self.assertTrue(abs(rel_error) < rtol)
        return

    @usesFun3d
    def test_transfer_loads(self):
        # build a funtofem model with one body and scenario
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
        test_scenario.include(Function.lift())
        test_scenario.register_to(model)

        # suppress the standard output from fortran, fun3d
        with stdchannel_redirected(sys.stdout, os.devnull):

            # build the solvers and coupled driver
            comm = MPI.COMM_WORLD
            solvers = SolverManager(comm)
            solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e4)
            solvers.structural = TestStructuralSolver(comm, model, elastic_k=1000.0)
            # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
            transfer_settings = TransferSettings()

            # run one forward analysis of the driver to setup the fun3d interface as well and transfer schemes
            FUNtoFEMnlbgs(
                solvers,
                transfer_settings=transfer_settings,
                model=model,
            ).solve_forward()

        # start displacement transfer test
        plate.initialize_adjoint_variables(test_scenario)
        ns = plate.get_num_struct_nodes()
        na = plate.get_num_aero_nodes()

        # random input contravariants for arbitrary space curves uS(s), fA(s) in input spaces
        duS_ds = np.random.rand((3 * ns))
        dfA_ds = np.random.rand((3 * na))

        # random output covariant to be written into struct_loads_ajp
        dL_dfS = np.random.rand((3 * ns))

        # adjoint method backprop fS adjoint covariant
        # to uS adjoint covariant and fA adjoint covariant
        plate.struct_loads_ajp[:, 0] = dL_dfS[:]
        plate.transfer_loads_adjoint(test_scenario)
        dL_duS = plate.struct_disps_ajp_loads[:, 0].copy()
        dL_dfA = plate.aero_loads_ajp[:, 0].copy()
        adjoint_dLds = np.sum(dL_duS * duS_ds).real + np.sum(dL_dfA * dfA_ds).real

        # complex step method with perturbations of input contravariants
        h = 1e-30
        rtol = 1e-9
        struct_disps = plate.get_struct_disps(test_scenario)
        aero_loads = plate.get_aero_loads(test_scenario)
        struct_disps[:] += 1j * h * duS_ds
        aero_loads[:] += 1j * h * dfA_ds

        # mainly just want to transfer_loads, but transfer_disps sets global data from uS used for loads
        plate.transfer_disps(test_scenario)
        plate.transfer_loads(test_scenario)
        struct_loads = plate.get_struct_loads(test_scenario)
        dfS_ds = np.array([_.imag / h for _ in list(struct_loads)])
        complex_dLds = np.sum(dL_dfS * dfS_ds).real

        # compare the total derivatives of complex step vs adjoint & evaluate
        rel_error = (adjoint_dLds - complex_dLds) / complex_dLds
        rel_error = rel_error.real
        TestResult(
            "TransferLoads-Laminar-Aeroelastic",
            "None",
            complex_dLds,
            adjoint_dLds,
            rel_error,
        ).report()


if __name__ == "__main__":
    unittest.main()
