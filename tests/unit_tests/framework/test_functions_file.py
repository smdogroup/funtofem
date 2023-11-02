import os, numpy as np, unittest
from tacs import TACS
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
    CompositeFunction,
)
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    SolverManager,
    Remote,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings
from _bdf_test_utils import elasticity_callback

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")

comm = MPI.COMM_WORLD
ntacs_procs = 1
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex

_, output_dir = make_test_directories(comm, base_dir)

remote = Remote(comm, ".py", base_dir, nprocs=1)

class TestFunctionsFile(unittest.TestCase):

    def test_multiscenario_aeroelastic(self):
        """
        test composite functions over two scenarios with aerothermoelastic coupling and a funtofem driver
        """
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        svar = Variable.structural("thickness").set_bounds(value=0.1).register_to(plate)
        plate.register_to(model)

        # climb scenario with random test composite function 1
        cruise = Scenario.steady("cruise", steps=100)
        yaw = Variable.aerodynamic("yaw", value=10.0).register_to(cruise)
        ksfailure1 = Function.ksfailure(ks_weight=10.0).register_to(cruise)
        lift1 = Function.lift().register_to(cruise)
        drag1 = Function.drag().register_to(cruise)
        cruise.register_to(model)

        # cruise scenario with random test composite function 2
        climb = Scenario.steady("climb", steps=100)
        ksfailure2 = Function.ksfailure(ks_weight=10.0).register_to(climb)
        lift2 = Function.lift().register_to(climb)
        mach = Variable.aerodynamic("mach", value=1.45).register_to(climb)
        drag2 = Function.drag().register_to(climb)
        climb.register_to(model)

        # all composite functions must come after bodies and scenarios
        # otherwise they won't be linked to all potential design variables (which they may depend on)
        composite1 = (
            drag1
            * CompositeFunction.exp(ksfailure1 + 1.5 * lift1**2 / (1 + lift1))
            * yaw
        )
        composite1.optimize().set_name("composite1").register_to(model)
        composite2 = (
            drag2**3
            / (1.532 - ksfailure2)
            * CompositeFunction.log(1 + (lift2 / drag2) ** 2)
        ) - mach**2
        composite2.optimize().set_name("composite2").register_to(model)

        # composite function for minimum drag among two scenarios
        min_drag = CompositeFunction.boltz_min([drag1, drag2])
        min_drag.register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=elasticity_callback,
            output_dir=output_dir,
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # instantiate the driver to in initialize transfer scheme
        FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        # set analysis functions to random values
        for func in model.get_functions():
            func.value = np.random.rand()
            for var in model.get_variables():
                func.derivatives[var] = np.random.rand()

        # test out the functionality for the funtofem_shape_driver

        # write analysis functions file and then read it back in
        model.write_functions_file(comm, remote._functions_file, full_precision=True)

        # set all other functions to zero
        for func in model.get_functions(all=True):
            if func.value == 0.0:
                for var in model.get_variables():
                    func.derivatives[var] = 0.0

        model.read_functions_file(comm, remote._functions_file)

        # evaluate composite functions
        model.evaluate_composite_functions(compute_grad=True)

        # write out the functions file again
        model.write_functions_file(comm, remote._functions_file+"2", full_precision=True)

        # write the full functions file for optimization
        model.write_functions_file(comm, remote.functions_file, full_precision=False, optim=True)


if __name__ == "__main__":
    unittest.main()
