import numpy as np, os
from mpi4py import MPI
from funtofem import TransferScheme
from tacs import TACS

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

import unittest

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

np.random.seed(123456)

# steps = 10  # used for Nstep case
# couplings = ["aeroelastic", "aerothermal", "aerothermoelastic"]
# coupling = "aeroelastic"
# d(struct func)/d(struct var) passes to machine precision (TD and matrix)
# both structDV cases pass, but aeroDV cases don't do as well
# worst case is d(aero func)/d(aero var) 1e-2 right now
# DV_cases = ["structural", "aerodynamic"]

# settings for each test
DV_cases = ["structural"]
# functions = ["structural", "aerodynamic"]
function_type = "aerodynamic"


@unittest.skipIf(not complex_mode, "only designed for complex mode")
class TestFrameworkAdjointEqns(unittest.TestCase):
    FILENAME = "framework-adjoint-eqns.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_2step_adjoint_eqns_aeroelastic(self):
        # not all above features affect this test
        # these are fixed
        steps = 2
        coupling = "aeroelastic"

        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body("plate", analysis_type=coupling)
        if "structural" in DV_cases:
            for iS in range(5):
                Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(
                    plate
                )
        if "aerodynamic" in DV_cases:
            for iA in range(5):
                Variable.aerodynamic(f"aero{iA}").set_bounds(value=0.1).register_to(
                    plate
                )
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        # just do one function for now
        if function_type == "structural":
            test_scenario.include(Function.ksfailure())
        elif function_type == "aerodynamic":
            test_scenario.include(Function.lift())
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )
        test_aero = solvers.flow
        test_struct = solvers.structural
        aero_data = test_aero.scenario_data[test_scenario.id]
        struct_data = test_struct.scenario_data[test_scenario.id]
        bodies = model.bodies
        scenario = test_scenario

        driver.solve_forward()
        plate.initialize_adjoint_variables(test_scenario)
        # solve the adjoint variables and store states at each step
        # to compare with adjoint equations

        step = 2
        plate.transfer_disps(test_scenario, time_index=step - 1, jump=True)

        test_struct.iterate_adjoint(scenario, bodies, step)
        psi_S2 = -1 * plate.get_struct_disps_ajp(scenario).copy()
        plate.transfer_loads_adjoint(scenario)
        test_aero.iterate_adjoint(scenario, bodies, step)
        plate.transfer_disps_adjoint(scenario)

        psi_L2 = -1 * plate.get_struct_loads_ajp(scenario).copy()
        psi_A2 = -1 * plate.get_aero_loads_ajp(scenario).copy()
        psi_D2 = -1 * plate.get_aero_disps_ajp(scenario).copy()

        step = 1

        # need this to set the disps so the load transfer jacobian is
        # correct in intermediate steps
        plate.transfer_disps(test_scenario, time_index=step - 1, jump=True)

        test_struct.iterate_adjoint(scenario, bodies, step)
        psi_S1 = -1 * plate.get_struct_disps_ajp(scenario).copy()
        plate.transfer_loads_adjoint(scenario)
        test_aero.iterate_adjoint(scenario, bodies, step)
        plate.transfer_disps_adjoint(scenario)

        psi_L1 = -1 * plate.get_struct_loads_ajp(scenario).copy()
        psi_A1 = -1 * plate.get_aero_loads_ajp(scenario).copy()
        psi_D1 = -1 * plate.get_aero_disps_ajp(scenario).copy()

        # check each of the equations
        resids = []
        # df/duA1 = 0
        df_duA1 = psi_D1 - aero_data.Jac1.T @ psi_A1
        resids += [np.linalg.norm(df_duA1)]

        # df/dfA1 = 0
        aero_out = np.zeros(3 * plate.aero_nnodes, dtype=plate.dtype)
        plate.transfer.applydLdfATrans(psi_L1[:, 0].copy(), aero_out)
        df_dfA1 = psi_A1 + np.expand_dims(aero_out, axis=-1)
        resids += [np.linalg.norm(df_dfA1)]

        # df/dfS1 = 0
        df_dfS1 = psi_L1 - struct_data.Jac1.T @ psi_S1
        resids += [np.linalg.norm(df_dfS1)]

        # df/duS1 = 0
        # seed MELD with u_S^1 => u_A^2 transfer here so Jacobians are correct
        step = 2
        plate.transfer_disps(test_scenario, time_index=step - 1, jump=True)

        D2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
        plate.transfer.applydDduSTrans(psi_D2[:, 0].copy(), D2_ajp)
        L2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
        plate.transfer.applydLduSTrans(psi_L2[:, 0].copy(), L2_ajp)
        df_duS1 = psi_S1 + np.expand_dims(D2_ajp + L2_ajp, axis=-1)
        resids += [np.linalg.norm(df_duS1)]

        # df/duA2 = 0
        df_duA2 = psi_D2 - np.dot(aero_data.Jac1.T, psi_A2)
        if function_type == "aerodynamic":
            df_duA2 += np.expand_dims(aero_data.func_coefs1, axis=-1)
        resids += [np.linalg.norm(df_duA2)]

        # df/dfA2 = 0
        aero_out = np.zeros(3 * plate.aero_nnodes, dtype=plate.dtype)
        plate.transfer.applydLdfATrans(psi_L2[:, 0].copy(), aero_out)
        df_dfA2 = psi_A2 + np.expand_dims(aero_out, axis=-1)
        resids += [np.linalg.norm(df_dfA2)]

        # df/dfS2 = 0
        term2 = np.dot(struct_data.Jac1.T, psi_S2)
        df_dfS2 = psi_L2 - term2
        if function_type == "structural":
            df_dfS2 += np.expand_dims(struct_data.func_coefs1, axis=-1)
        resids += [np.linalg.norm(df_dfS2)]

        # df/duS2 = 0
        df_duS2 = psi_S2
        resids += [np.linalg.norm(df_duS2)]

        adjoints = [
            psi_D1,
            psi_A1,
            psi_L1,
            psi_S1,
            psi_D2,
            psi_A2,
            psi_L2,
            psi_S2,
        ]
        adjoint_norms = [np.linalg.norm(_) for _ in adjoints]
        print(f"2 step adjoint eqns-{coupling}")
        print(f"\tadjoint norms = {adjoint_norms}")

        resids = [abs(_) for _ in resids]
        passing_resid = [_ < 1e-9 for _ in resids]
        npassing = sum(passing_resid)
        nresid = len(resids)
        max_matrix_resid = max(resids)
        print(f"\t{npassing}/{nresid} passing residuals")
        print(f"\tmatrix resids = {resids}")
        print(f"\tmax matrix resid = {max_matrix_resid}")

        # compute the total directional derivative from the adjoint
        variables = model.get_variables()
        nvars = len(variables)
        dvar_ds = np.random.rand(nvars)
        adjoint_TD = 0.0
        for ivar, var in enumerate(variables):
            if var.analysis_type == "aerodynamic":
                adjoint_TD -= dvar_ds[ivar] * np.dot(
                    psi_A1[:, 0] + psi_A2[:, 0], aero_data.c1[:, ivar]
                )
            else:  # structural
                adjoint_TD -= dvar_ds[ivar] * np.dot(
                    psi_S1[:, 0] + psi_S2[:, 0], struct_data.c1[:, ivar]
                )
        adjoint_TD = adjoint_TD.real

        # complex step analysis
        h = 1e-30
        for ivar, var in enumerate(variables):
            var.value += dvar_ds[ivar] * 1j * h
        driver.solve_forward()
        functions = model.get_functions()
        complex_step_TD = functions[0].value.imag / h

        TD_rel_error = abs((adjoint_TD - complex_step_TD) / complex_step_TD)
        print(f"\tadjoint TD = {adjoint_TD}")
        print(f"\tcomplex step TD = {complex_step_TD}")
        print(f"\tTD rel error = {TD_rel_error}")

        assert max_matrix_resid < 1e-9
        assert TD_rel_error < 1e-9
        return

    def test_2step_adjoint_eqns_aerothermal(self):
        # not all above features affect this test
        # these are fixed
        steps = 2
        coupling = "aerothermal"

        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body("plate", analysis_type=coupling)
        if "structural" in DV_cases:
            for iS in range(5):
                Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(
                    plate
                )
        if "aerodynamic" in DV_cases:
            for iA in range(5):
                Variable.aerodynamic(f"aero{iA}").set_bounds(value=0.1).register_to(
                    plate
                )
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        # just do one function for now
        if function_type == "structural":
            test_scenario.include(Function.ksfailure())
        elif function_type == "aerodynamic":
            test_scenario.include(Function.lift())
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )
        test_aero = solvers.flow
        test_struct = solvers.structural
        aero_data = test_aero.scenario_data[test_scenario.id]
        struct_data = test_struct.scenario_data[test_scenario.id]
        bodies = model.bodies
        scenario = test_scenario

        driver.solve_forward()
        plate.initialize_adjoint_variables(test_scenario)
        # solve the adjoint variables and store states at each step
        # to compare with adjoint equations

        step = 2
        plate.transfer_temps(test_scenario, time_index=step - 1, jump=True)

        test_struct.iterate_adjoint(scenario, bodies, step)
        psi_P2 = -1 * plate.get_struct_temps_ajp(scenario).copy()
        plate.transfer_heat_flux_adjoint(scenario)
        test_aero.iterate_adjoint(scenario, bodies, step)
        plate.transfer_temps_adjoint(scenario)

        psi_Q2 = -1 * plate.get_struct_heat_flux_ajp(scenario).copy()
        psi_A2 = -1 * plate.get_aero_heat_flux_ajp(scenario).copy()
        psi_T2 = -1 * plate.get_aero_temps_ajp(scenario).copy()

        step = 1

        # sometimes I comment this extra transfer out
        plate.transfer_temps(test_scenario, time_index=step - 1, jump=True)

        test_struct.iterate_adjoint(scenario, bodies, step)
        psi_P1 = -1 * plate.get_struct_temps_ajp(scenario).copy()
        plate.transfer_heat_flux_adjoint(scenario)
        test_aero.iterate_adjoint(scenario, bodies, step)
        plate.transfer_temps_adjoint(scenario)

        psi_Q1 = -1 * plate.get_struct_heat_flux_ajp(scenario).copy()
        psi_A1 = -1 * plate.get_aero_heat_flux_ajp(scenario).copy()
        psi_T1 = -1 * plate.get_aero_temps_ajp(scenario).copy()

        # check each of the equations
        resids = []
        # df/dtA1 = 0
        df_dtA1 = psi_T1 - aero_data.Jac2.T @ psi_A1
        resids += [np.linalg.norm(df_dtA1)]

        # df/dhA1 = 0
        aero_out = np.zeros(plate.aero_nnodes, dtype=plate.dtype)
        plate.thermal_transfer.applydQdqATrans(psi_Q1[:, 0].copy(), aero_out)
        df_dhA1 = psi_A1 + np.expand_dims(aero_out, axis=-1)
        resids += [np.linalg.norm(df_dhA1)]

        # df/dhS1 = 0
        df_dhS1 = psi_Q1 - struct_data.Jac2.T @ psi_P1
        resids += [np.linalg.norm(df_dhS1)]

        # df/dtS1 = 0
        # seed MELD with t_S^1 => t_A^2 transfer here so Jacobians are correct
        step = 2
        plate.transfer_disps(test_scenario, time_index=step - 1, jump=True)

        T2_ajp = np.zeros(plate.struct_nnodes, dtype=plate.dtype)
        plate.thermal_transfer.applydTdtSTrans(psi_T2[:, 0].copy(), T2_ajp)
        df_dtS1 = psi_P1 + np.expand_dims(T2_ajp, axis=-1)
        resids += [np.linalg.norm(df_dtS1)]

        # df/dtA2 = 0
        df_dtA2 = psi_T2 - np.dot(aero_data.Jac2.T, psi_A2)
        if function_type == "aerodynamic":
            df_dtA2 += np.expand_dims(aero_data.func_coefs2, axis=-1)
        resids += [np.linalg.norm(df_dtA2)]

        # df/dhA2 = 0
        aero_out = np.zeros(plate.aero_nnodes, dtype=plate.dtype)
        plate.thermal_transfer.applydQdqATrans(psi_Q2[:, 0].copy(), aero_out)
        df_dhA2 = psi_A2 + np.expand_dims(aero_out, axis=-1)
        resids += [np.linalg.norm(df_dhA2)]

        # df/dhS2 = 0
        term2 = np.dot(struct_data.Jac2.T, psi_P2)
        df_dhS2 = psi_Q2 - term2
        if function_type == "structural":
            df_dhS2 += np.expand_dims(struct_data.func_coefs2, axis=-1)
        resids += [np.linalg.norm(df_dhS2)]

        # df/dtS2 = 0
        df_dtS2 = psi_P2
        resids += [np.linalg.norm(df_dtS2)]

        adjoints = [
            psi_T1,
            psi_A1,
            psi_Q1,
            psi_P1,
            psi_T2,
            psi_A2,
            psi_Q2,
            psi_P2,
        ]
        adjoint_norms = [np.linalg.norm(_) for _ in adjoints]
        print(f"2 step adjoint eqns-{coupling}")
        print(f"\tadjoint norms = {adjoint_norms}")

        resids = [abs(_) for _ in resids]
        passing_resid = [_ < 1e-9 for _ in resids]
        npassing = sum(passing_resid)
        nresid = len(resids)
        max_matrix_resid = max(resids)
        print(f"\t{npassing}/{nresid} passing residuals")
        print(f"\tmatrix resids = {resids}")
        print(f"\tmax matrix resid = {max_matrix_resid}")

        # compute the total directional derivative from the adjoint
        variables = model.get_variables()
        nvars = len(variables)
        dvar_ds = np.random.rand(nvars)
        adjoint_TD = 0.0
        for ivar, var in enumerate(variables):
            if var.analysis_type == "aerodynamic":
                adjoint_TD -= dvar_ds[ivar] * np.dot(
                    psi_A1[:, 0] + psi_A2[:, 0], aero_data.c2[:, ivar]
                )
            else:  # structural
                dfdx_step2 = -1 * np.dot(psi_P2[:, 0], struct_data.c2[:, ivar])
                dfdx_step1 = -1 * np.dot(psi_P1[:, 0], struct_data.c2[:, ivar])
                adjoint_TD -= dvar_ds[ivar] * np.dot(
                    psi_P1[:, 0] + psi_P2[:, 0], struct_data.c2[:, ivar]
                )
        adjoint_TD = adjoint_TD.real

        # complex step analysis
        h = 1e-30
        for ivar, var in enumerate(variables):
            var.value += dvar_ds[ivar] * 1j * h
        driver.solve_forward()
        functions = model.get_functions()
        complex_step_TD = functions[0].value.imag / h

        TD_rel_error = abs((adjoint_TD - complex_step_TD) / complex_step_TD)
        print(f"\tadjoint TD = {adjoint_TD}")
        print(f"\tcomplex step TD = {complex_step_TD}")
        print(f"\tTD rel error = {TD_rel_error}")

        assert max_matrix_resid < 1e-9
        assert TD_rel_error < 1e-9
        return

    # @unittest.skip("temp")
    # def test_Nstep_adjoint_eqns(self):
    #     # build the funtofem model with an unsteady scenario
    #     model = FUNtoFEMmodel("test")
    #     plate = Body("plate", analysis_type=coupling)
    #     if "structural" in DV_cases:
    #         for iS in range(5):
    #             Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(
    #                 plate
    #             )
    #     if "aerodynamic" in DV_cases:
    #         for iA in range(5):
    #             Variable.aerodynamic(f"aero{iA}").set_bounds(value=0.1).register_to(
    #                 plate
    #             )
    #     plate.register_to(model)
    #     test_scenario = Scenario.unsteady("test", steps=steps)
    #     # just do one function for now
    #     if function_type == "structural":
    #         test_scenario.include(Function.ksfailure())
    #     elif function_type == "aerodynamic":
    #         test_scenario.include(Function.lift())
    #     test_scenario.register_to(model)

    #     # build a funtofem driver
    #     solvers = SolverManager(comm)
    #     solvers.flow = TestAerodynamicSolver(comm, model)
    #     solvers.structural = TestStructuralSolver(comm, model)
    #     driver = FUNtoFEMnlbgs(
    #         solvers, transfer_settings=TransferSettings(npts=10), model=model
    #     )
    #     test_aero = solvers.flow
    #     test_struct = solvers.structural
    #     aero_data = test_aero.scenario_data[test_scenario.id]
    #     struct_data = test_struct.scenario_data[test_scenario.id]
    #     bodies = model.bodies
    #     scenario = test_scenario

    #     driver.solve_forward()
    #     plate.initialize_adjoint_variables(test_scenario)
    #     # solve the adjoint variables and store states at each step
    #     # to compare with adjoint equations

    #     plate.transfer_disps(test_scenario, time_index=steps - 1, jump=True)

    #     psi_D = {}
    #     psi_A = {}
    #     psi_L = {}
    #     psi_S = {}

    #     for istep in range(1, steps + 1):
    #         step = steps - istep + 1  # reverse step

    #         test_struct.iterate_adjoint(scenario, bodies, step)
    #         psi_S[step] = -1 * plate.get_struct_disps_ajp(scenario).copy()
    #         plate.transfer_loads_adjoint(scenario)
    #         test_aero.iterate_adjoint(scenario, bodies, step)
    #         plate.transfer_disps_adjoint(scenario)

    #         psi_L[step] = -1 * plate.get_struct_loads_ajp(scenario).copy()
    #         psi_A[step] = -1 * plate.get_aero_loads_ajp(scenario).copy()
    #         psi_D[step] = -1 * plate.get_aero_disps_ajp(scenario).copy()

    #     # post-analyze the adjoint equations
    #     resids = []
    #     for istep in range(1, steps + 1):
    #         step = steps - istep + 1
    #         # check each of the equations
    #         # df/duAi = 0
    #         df_duA = psi_D[step] - aero_data.Jac1.T @ psi_A[step]
    #         resids += [np.linalg.norm(df_duA)]

    #         # df/dfAi = 0
    #         aero_out = np.zeros(3 * plate.aero_nnodes, dtype=plate.dtype)
    #         plate.transfer.applydLdfATrans(psi_L[step][:, 0].copy(), aero_out)
    #         df_dfA = psi_A[step] + np.expand_dims(aero_out, axis=-1)
    #         resids += [np.linalg.norm(df_dfA)]

    #         # df/dfSi = 0
    #         df_dfS = psi_L[step] - struct_data.Jac1.T @ psi_S[step]
    #         if step == steps:
    #             df_dfS += np.expand_dims(struct_data.func_coefs1, axis=-1)
    #         resids += [np.linalg.norm(df_dfS)]

    #         # df/duSi = 0
    #         if step < steps:
    #             D2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
    #             plate.transfer.applydDduSTrans(psi_D[step + 1][:, 0].copy(), D2_ajp)
    #             L2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
    #             plate.transfer.applydLduSTrans(psi_L[step + 1][:, 0].copy(), L2_ajp)
    #             df_duS = psi_S[step] + np.expand_dims(D2_ajp + L2_ajp, axis=-1)
    #         else:
    #             df_duS = psi_S[step]
    #         resids += [np.linalg.norm(df_duS)]

    #     print(f"{steps} step adjoint eqns")

    #     resids = [abs(_) for _ in resids]
    #     passing_resid = [_ < 1e-9 for _ in resids]
    #     npassing = sum(passing_resid)
    #     nresid = len(resids)
    #     print(f"\t{npassing}/{nresid} passing residuals")
    #     print(f"\tresids = {resids}")
    #     print(f"\tmax resid = {max(resids)}")

    #     assert max(resids) < 1e-9
    #     return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFrameworkAdjointEqns.FILENAME, "w").close()  # clear file
    unittest.main()
