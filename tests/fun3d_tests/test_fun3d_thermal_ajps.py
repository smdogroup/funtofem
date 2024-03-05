from contextlib import contextmanager
from os import devnull
import numpy as np, unittest, importlib
import os, sys
from mpi4py import MPI

from funtofem.model import (
    FUNtoFEMmodel,
    Body,
    Scenario,
    Function,
    AitkenRelaxation,
    Variable,
)
from funtofem.interface import (
    make_test_directories,
)
from funtofem.driver import TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3d14Interface

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
results_folder, _ = make_test_directories(comm, base_dir)


@unittest.skipIf(not has_fun3d, "skipping fun3d test without fun3d")
class TestTurbulentAerothermal(unittest.TestCase):
    FILENAME = "full_plate.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_aerothermal_interface(self):
        # build a funtofem model with one body and scenario
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("turbulent2", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.temperature())
        test_scenario.set_flow_ref_vals(qinf=1.0e4)
        test_scenario.register_to(model)

        # suppress stdout during each FUN3D analysis
        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        _fun3d_interface = Fun3d14Interface(comm, model, fun3d_dir="meshes", complex_mode=False)
        # solvers.structural = TestStructuralSolver(comm, model, thermal_k=100.0)
        # # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
        # transfer_settings = TransferSettings()

        # initialize plate and transfer
        plate.initialize_transfer(
            comm,
            comm,
            0,
            comm,
            0,
            transfer_settings=TransferSettings(npts=20),
        )
        plate.initialize_variables(test_scenario)
        plate.initialize_adjoint_variables(test_scenario)

        _fun3d_interface.set_variables(test_scenario, [plate])
        _fun3d_interface.set_functions(test_scenario, [plate])

        # randomly generate the temperature variation, input adjoint
        na = plate.get_num_aero_nodes()
        aero_temps = plate.get_aero_temps(test_scenario)
        temp0 = np.ones(na) + 0.01 * np.random.rand(na)
        aero_temps[:] = temp0[:]
        dTds = np.random.rand(na)
        lamH = 0.1 * (np.ones(na) + 0.001 *np.random.rand(na))
        
        _lamH = np.reshape(lamH, newshape=(na,1))
        _lamH = np.asfortranarray(_lamH)

        adj_product = None
        fd_product = 0.0
        epsilon = 1e-4
        test_steps = test_scenario.steps

        # forward analysis h(T)
        _fun3d_interface.initialize(test_scenario, [plate])
        _fun3d_interface.fun3d_flow.input_wall_temperature(temp0, body=1)
        comm.Barrier()
        for i in range(test_steps):
            _fun3d_interface.fun3d_flow.iterate()
        cqa = _fun3d_interface.fun3d_flow.extract_cqa(na, body=1)
        _fun3d_interface.post(test_scenario, [plate])

        # adjoint analysis on h(T), input lam_H
        _fun3d_interface.set_variables(test_scenario, [plate])
        _fun3d_interface.set_functions(test_scenario, [plate])
        _fun3d_interface.initialize_adjoint(test_scenario, [plate])
        comm.Barrier()
        for i in range(test_steps):
            # input zero for force adjoint
            # lam_x = np.zeros((3*na,1), dtype=np.double)
            # lam_x = np.asfortranarray(lam_x)
            # _fun3d_interface.fun3d_adjoint.input_force_adjoint(lam_x, lam_x, lam_x, body=1)

            _fun3d_interface.fun3d_adjoint.input_cqa_adjoint(_lamH, body=1)
            _fun3d_interface.fun3d_adjoint.iterate(i+1)
        lamT = _fun3d_interface.fun3d_adjoint.extract_thermal_adjoint_product(na, 1, body=1)
        _fun3d_interface.post_adjoint(test_scenario, [plate])

        adj_product = np.dot(lamT[:,0], dTds)

        # forward analysis h(T+dT/ds*eps)
        _fun3d_interface.initialize(test_scenario, [plate])
        _fun3d_interface.fun3d_flow.input_wall_temperature(temp0+dTds*epsilon, body=1)
        comm.Barrier()
        for i in range(test_steps):
            _fun3d_interface.fun3d_flow.iterate()
        cqaR = _fun3d_interface.fun3d_flow.extract_cqa(na, body=1)
        _fun3d_interface.post(test_scenario, [plate])

        fd_product += np.dot(cqaR, lamH) / epsilon

        # forward analysis h(T-dT/ds*eps)
        _fun3d_interface.initialize(test_scenario, [plate])
        _fun3d_interface.fun3d_flow.input_wall_temperature(temp0-dTds*epsilon, body=1)
        comm.Barrier()
        for i in range(test_steps):
            _fun3d_interface.fun3d_flow.iterate()
        cqaL = _fun3d_interface.fun3d_flow.extract_cqa(na, body=1)
        _fun3d_interface.post(test_scenario, [plate])

        fd_product -= np.dot(cqaL, lamH) / epsilon

        rel_error = (adj_product - fd_product) / fd_product
        rtol = 1e-4

        print(f"Fun3d 14 Interface aerothermal ajp test")
        print(f"\tadj product = {adj_product}")
        print(f"\tfd product = {fd_product}")
        print(f"\trel error = {rel_error}")

        self.assertTrue(abs(rel_error) < rtol)
        return


if __name__ == "__main__":
    unittest.main()
