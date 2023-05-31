import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
from tacs import TACS

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
    TestResult,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

import unittest

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
comm = MPI.COMM_WORLD


class CoupledUnsteadyFrameworkTest(unittest.TestCase):
    FILENAME = "fake-solvers-drivers.txt"

    @unittest.skipIf(not complex_mode, "aero solver test only runs in complex")
    def test_aero_solver(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body.aerothermoelastic("plate")
        for iS in range(5):
            Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(plate)
        plate.register_to(model)
        test = Scenario.unsteady("test", steps=10)
        test.include(Function.ksfailure()).register_to(model)

        # build a funtofem driver as it helps initialize body -> AeroSolver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        FUNtoFEMnlbgs(solvers, transfer_settings=TransferSettings(npts=10), model=model)
        flow_solver = solvers.flow
        plate.initialize_variables(test)
        plate.initialize_adjoint_variables(test)

        # random covariant and contravariant tensors
        na = plate.get_num_aero_nodes()
        dua_ds = np.random.rand(3 * na)  # contravariants of input states
        dta_ds = np.random.rand(na)
        dL_dfa = np.random.rand(3 * na)  # covariants of output states
        dL_dha = np.random.rand(na)

        # adjoint evaluation
        aero_loads_ajp = plate.get_aero_loads_ajp(test)
        aero_flux_ajp = plate.get_aero_heat_flux_ajp(test)
        aero_loads_ajp[:, 0] = dL_dfa[:]
        aero_flux_ajp[:, 0] = dL_dha[:]

        flow_solver.iterate_adjoint(test, model.bodies, step=0)

        aero_disps_ajp = plate.get_aero_disps_ajp(test)[:, 0]
        aero_temps_ajp = plate.get_aero_temps_ajp(test)[:, 0]
        adjoint_TD = aero_disps_ajp @ dua_ds + aero_temps_ajp @ dta_ds
        adjoint_TD = float(adjoint_TD.real)

        # forward contravariant pass with complex step
        h = 1e-30
        ua = plate.get_aero_disps(test, 0)
        ua[:] += dua_ds[:] * 1j * h
        ta = plate.get_aero_temps(test, 0)
        ta[:] += dta_ds[:] * 1j * h

        flow_solver.iterate(test, model.bodies, step=0)

        fa = list(plate.get_aero_loads(test, 0))
        dfa_ds = np.array([_.imag / h for _ in fa])
        ha = list(plate.get_aero_heat_flux(test, 0))
        dha_ds = np.array([_.imag / h for _ in ha])
        complex_step_TD = dL_dfa @ dfa_ds + dL_dha @ dha_ds

        rel_error = (adjoint_TD - complex_step_TD) / complex_step_TD
        TestResult(
            "aero_solver",
            ["ksfailure"],
            [complex_step_TD],
            [adjoint_TD],
            [rel_error],
            comm,
        ).report()
        self.assertTrue(abs(rel_error) < 1e-9)
        return

    @unittest.skipIf(not complex_mode, "struct solver test only runs in complex")
    def test_structural_solver(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body.aerothermoelastic("plate")
        for iS in range(5):
            Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(plate)
        plate.register_to(model)
        test = Scenario.unsteady("test", steps=10)
        test.include(Function.lift()).register_to(model)

        # build a funtofem driver as it helps initialize body -> AeroSolver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        FUNtoFEMnlbgs(solvers, transfer_settings=TransferSettings(npts=10), model=model)
        structural_solver = solvers.structural
        plate.initialize_variables(test)
        plate.initialize_adjoint_variables(test)

        # random covariant and contravariant tensors
        ns = plate.get_num_struct_nodes()
        dfs_ds = np.random.rand(3 * ns)  # contravariants of input states
        dhs_ds = np.random.rand(ns)
        dL_dus = np.random.rand(3 * ns)  # covariants of output states
        dL_dts = np.random.rand(ns)

        # adjoint evaluation
        struct_disps_ajp = plate.get_struct_disps_ajp(test)
        struct_temps_ajp = plate.get_struct_temps_ajp(test)
        struct_disps_ajp[:, 0] = dL_dus[:]
        struct_temps_ajp[:, 0] = dL_dts[:]

        structural_solver.iterate_adjoint(test, model.bodies, step=0)

        dL_dfs = plate.get_struct_loads_ajp(test)[:, 0]
        dL_dhs = plate.get_struct_heat_flux_ajp(test)[:, 0]
        adjoint_TD = dL_dfs @ dfs_ds + dL_dhs @ dhs_ds
        adjoint_TD = adjoint_TD.real

        # forward contravariant pass with complex step
        h = 1e-30
        fs = plate.get_struct_loads(test, 0)
        fs[:] += dfs_ds[:] * 1j * h
        hs = plate.get_struct_heat_flux(test, 0)
        hs[:] += dhs_ds[:] * 1j * h

        structural_solver.iterate(test, model.bodies, step=0)

        us = list(plate.get_struct_disps(test, 0))
        dus_ds = np.array([_.imag / h for _ in us])
        ts = list(plate.get_struct_temps(test, 0))
        dts_ds = np.array([_.imag / h for _ in ts])
        complex_step_TD = dL_dus @ dus_ds + dL_dts @ dts_ds

        rel_error = (adjoint_TD - complex_step_TD) / complex_step_TD
        TestResult(
            "struct_solver",
            ["ksfailure"],
            [complex_step_TD],
            [adjoint_TD],
            [rel_error],
            comm,
        ).report()
        self.assertTrue(abs(rel_error) < 1e-9)
        return

    def test_structDV_with_driver(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body.aerothermoelastic("plate")
        for iS in range(5):
            Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=10)
        test_scenario.include(Function.ksfailure()).include(Function.lift())
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )

        rtol = 1e-7 if complex_mode else 1e-4
        max_rel_error = TestResult.derivative_test(
            "fake_solvers-structDV",
            model,
            driver,
            CoupledUnsteadyFrameworkTest.FILENAME,
            complex_mode,
        )
        self.assertTrue(max_rel_error < rtol)
        return

    def test_aeroDV_with_driver(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body.aerothermoelastic("plate")
        for iA in range(5):
            Variable.aerodynamic(f"aero{iA}").set_bounds(value=0.1).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=10)
        test_scenario.include(Function.ksfailure()).include(Function.lift())
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )

        rtol = 1e-7 if complex_mode else 1e-4
        max_rel_error = TestResult.derivative_test(
            "fake_solvers-aeroDV",
            model,
            driver,
            CoupledUnsteadyFrameworkTest.FILENAME,
            complex_mode,
        )
        self.assertTrue(max_rel_error < rtol)
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(CoupledUnsteadyFrameworkTest.FILENAME, "w").close()  # clear file
    unittest.main()
