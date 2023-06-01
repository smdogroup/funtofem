import os, unittest, traceback
from mpi4py import MPI
from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
design_file = os.path.join(base_dir, "funtofem.in")
design_file2 = os.path.join(base_dir, "funtofem2.in")


class DesignVarFileTest(unittest.TestCase):
    def test_dv_file(self):
        model = FUNtoFEMmodel("test")
        fake_plate = Body.aeroelastic("plate")
        for irib in range(1, 11):
            Variable.structural(f"rib{irib}", value=0.1 * irib).register_to(fake_plate)
        for ispar in range(1, 3):
            Variable.structural(f"spar{ispar}", value=0.1 * ispar).register_to(
                fake_plate
            )
        Variable.shape("rib_a1", value=0.13).register_to(fake_plate)
        Variable.shape("rib_a2", value=0.14).register_to(fake_plate)
        Variable.shape("sspan", value=10.0).register_to(fake_plate)
        fake_plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=0)
        test_scenario.get_variable("AOA", set_active=True).set_bounds(value=0.21)
        test_scenario.get_variable("Mach", set_active=True).set_bounds(value=0.31)
        test_scenario.register_to(model)

        # write the design variables file
        model.write_design_variables_file(comm, design_file, root=0)

        file_exists = os.path.exists(design_file)
        self.assertTrue(file_exists)

        # set all variables to zero in between
        for var in model.get_variables():
            var.value = 0.0

        # read the design variables file back in
        model.read_design_variables_file(comm, design_file, root=0)

        model.write_design_variables_file(comm, design_file2, root=0)
        return


if __name__ == "__main__":
    unittest.main()
