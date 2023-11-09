import unittest, os, numpy as np, sys
from _bdf_test_utils import elasticity_callback
from mpi4py import MPI
from tacs import TACS

np.set_printoptions(threshold=sys.maxsize)

from funtofem import TransferScheme
from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TacsInterface,
    SolverManager,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings, OnewayStructDriver

np.random.seed(1234567)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
comm = MPI.COMM_WORLD

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex

results_folder, output_dir = make_test_directories(comm, base_dir)
aero_loads_file = os.path.join(output_dir, "aero_loads.txt")


class TestAeroLoadsFileMPI(unittest.TestCase):
    N_PROCS = 2
    FILENAME = "test_aero_loads_mpi.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_matching_loads_mpi(self):
        # ---------------------------
        # Write the loads file
        # ---------------------------
        # build the model and driver
        f2f_model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(f2f_model)

        # build the scenario
        scenario = Scenario.steady("test", steps=150)
        Function.ksfailure().register_to(scenario)
        scenario.register_to(f2f_model)

        # generate random set of aero loads on each node
        # and distribute aero_id, etc among each of the processors
        nprocs = comm.Get_size()
        global_aero_id = [_ for _ in range(50)]
        global_aero_X = np.random.rand(150)

        plate.initialize_variables(scenario)
        for iproc in range(nprocs):
            if comm.rank == iproc:
                local_aero_id = [id for id in global_aero_id if id % nprocs == iproc]
                local_aero_X = list(
                    np.concatenate(
                        [global_aero_X[3 * id : 3 * id + 3] for id in local_aero_id]
                    )
                )
                plate.initialize_aero_nodes(local_aero_X, local_aero_id)

        # initialize random aero loads on each processor
        plate.aero_loads[scenario.id] = np.random.rand(3 * len(local_aero_id))

        # make the solvers for a CFD analysis to store and write the loads file
        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            f2f_model,
            comm,
            nprocs,
            bdf_filename,
            callback=elasticity_callback,
            output_dir=output_dir,
        )
        transfer_settings = TransferSettings(npts=5)
        FUNtoFEMnlbgs(solvers, transfer_settings=transfer_settings, model=f2f_model)

        # save initial loads in an array
        orig_aero_loads = plate.aero_loads[scenario.id] * 1.0
        f2f_model.write_aero_loads(comm, aero_loads_file, root=0)

        # zero the aero loads
        plate.aero_loads[scenario.id] *= 0.0

        # -----------------------------------------------
        # Read the loads file and test the oneway driver
        # -----------------------------------------------
        solvers.flow = None
        OnewayStructDriver.prime_loads_from_file(
            aero_loads_file, solvers, f2f_model, nprocs, transfer_settings
        )

        # verify the aero loads are the same on the local processor
        new_aero_loads = plate.aero_loads[scenario.id]
        diff_aero_loads = new_aero_loads - orig_aero_loads
        orig_norm = np.max(np.abs(orig_aero_loads))
        abs_err_norm = np.max(np.abs(diff_aero_loads))
        rel_err_norm = abs_err_norm / orig_norm

        # gather the results from each processor
        rtol = 1e-7
        matching_aero_loads = rel_err_norm < rtol
        print(f"matching aero loads = {matching_aero_loads} on rank {comm.rank}")
        matching_list = comm.gather(matching_aero_loads, root=0)
        if comm.rank == 0:
            all_matching = all(matching_list)
            print(f"Aero loads matching list for each proc = {matching_list}")
            if all_matching:
                print(f"Aero loads match for each individual processor!", flush=True)
            else:
                print(
                    f"Aero loads don't match on all processors: {matching_list}",
                    flush=True,
                )
        matching_list = comm.bcast(matching_list, root=0)
        assert all(matching_list)
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestAeroLoadsFileMPI.FILEPATH, "w").close()
    unittest.main()
