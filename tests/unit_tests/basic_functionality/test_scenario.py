#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from funtofem.model import Scenario, Function, Variable
import unittest


class ScenarioTest(unittest.TestCase):
    # Note: most of the functionality for Scenario is tested by test_body.py
    def build_scenario(self):
        cruise = Scenario(name="cruise", group=0, steady=False, fun3d=True, steps=10)

        drag = Function("cd", analysis_type="aerodynamic")
        cruise.add_function(drag)

        mass = Function("mass", analysis_type="structural", adjoint=False)
        cruise.add_function(mass)

        cruise.set_variable("aerodynamic", "AOA", value=5.0)

        cruise.add_variable(
            "aerodynamic", Variable("qinf", 1e3, lower=1e2, upper=2e3, active=True)
        )

        return cruise

    def test_build_scenario(self):
        scenario = self.build_scenario()

        assert scenario.name == "cruise"

        # Check the attributes
        assert scenario.steps == 10
        assert scenario.steady == False

        # Check the variables
        assert scenario.variables["aerodynamic"][0].name == "Mach"
        aoa = scenario.variables["aerodynamic"][1]
        assert aoa.name == "AOA"
        assert aoa.active == True
        assert aoa.value == 5.0
        assert scenario.variables["aerodynamic"][5].name == "zrate"

        # now the functions
        assert len(scenario.functions) == 2
        assert scenario.functions[0].name == "cd"
        assert scenario.functions[0].analysis_type == "aerodynamic"
        assert scenario.functions[0].adjoint == True

        assert scenario.functions[1].name == "mass"
        assert scenario.functions[1].analysis_type == "structural"
        assert scenario.functions[1].adjoint == False


class ScenarioFunctionOrderTest(unittest.TestCase):
    """
    Functions may be registered in any order; Scenario reorders them internally so
    that all functions requiring an adjoint come first. See
    Scenario._canonicalize_functions for why that ordering is required.
    """

    def test_reorder_mass_first(self):
        """registering the non-adjoint function first must not be an error"""
        scenario = Scenario(name="cruise", steps=10)
        scenario.add_function(Function.mass())
        scenario.add_function(Function.ksfailure())
        scenario.add_function(Function.lift())

        names = [func.name for func in scenario.functions]
        assert names[-1] == "mass"
        assert all(func.adjoint for func in scenario.functions[:-1])

        # function.id must stay the 1-based index into scenario.functions, since it
        # is pushed into the FUN3D design interface
        assert [func.id for func in scenario.functions] == [1, 2, 3]

    def test_registration_order_preserved_within_group(self):
        """the sort is stable, so relative order within each group is untouched"""
        scenario = Scenario(name="cruise", steps=10)
        scenario.add_function(Function.mass())
        scenario.add_function(Function.ksfailure())
        scenario.add_function(Function.lift())
        scenario.add_function(Function.drag())

        adjoint_names = [func.name for func in scenario.adjoint_functions]
        assert adjoint_names == ["ksfailure", "cl", "cd"]

    def test_already_correct_order_unchanged(self):
        """a correctly ordered scenario must not be perturbed at all"""
        scenario = Scenario(name="cruise", steps=10)
        scenario.add_function(Function.ksfailure())
        scenario.add_function(Function.lift())
        scenario.add_function(Function.mass())

        assert [func.name for func in scenario.functions] == ["ksfailure", "cl", "mass"]
        assert [func.id for func in scenario.functions] == [1, 2, 3]
        assert scenario._reordered == False

    def test_early_stopping_puts_aero_first(self):
        """
        with early stopping on, an aerodynamic function must come first or the FUN3D
        adjoint early stopping criterion fails
        """
        scenario = Scenario(name="cruise", steps=10, early_stopping=True)
        scenario.add_function(Function.ksfailure())
        scenario.add_function(Function.lift())
        scenario.add_function(Function.mass())

        assert scenario.functions[0].name == "cl"
        assert [func.name for func in scenario.functions] == ["cl", "ksfailure", "mass"]

    def test_early_stopping_set_after_registration(self):
        """turning early stopping on later must still reorder"""
        scenario = Scenario(name="cruise", steps=10, early_stopping=False)
        scenario.add_function(Function.ksfailure())
        scenario.add_function(Function.lift())
        scenario.add_function(Function.mass())
        assert scenario.functions[0].name == "ksfailure"

        scenario.set_stop_criterion(early_stopping=True)
        assert scenario.functions[0].name == "cl"
        assert [func.id for func in scenario.functions] == [1, 2, 3]

    def test_adjoint_map_is_identity(self):
        """
        the canonical order is what makes adjoint_map the identity, which the driver
        and interface index arithmetic relies on
        """
        scenario = Scenario(name="cruise", steps=10)
        scenario.add_function(Function.mass())
        scenario.add_function(Function.ksfailure())
        scenario.add_function(Function.lift())

        nadjoint = scenario.count_adjoint_functions()
        assert scenario.adjoint_map == {i: i for i in range(nadjoint)}
        assert scenario.reverse_adjoint_map == {
            v: k for k, v in scenario.adjoint_map.items()
        }

    def test_strict_mode_raises_informative_error(self):
        """with auto reordering disabled, the failure must name the functions"""
        Scenario.AUTO_REORDER_FUNCTIONS = False
        try:
            scenario = Scenario(name="cruise", steps=10)
            scenario.add_function(Function.mass())
            with self.assertRaises(RuntimeError) as raised:
                scenario.add_function(Function.ksfailure())
        finally:
            Scenario.AUTO_REORDER_FUNCTIONS = True

        message = str(raised.exception)
        assert "ksfailure" in message
        assert "mass" in message
        assert "cruise" in message

        # the rejected function must not have been left half registered
        assert [func.name for func in scenario.functions] == ["mass"]

    def test_strict_mode_still_assigns_ids(self):
        Scenario.AUTO_REORDER_FUNCTIONS = False
        try:
            scenario = Scenario(name="cruise", steps=10)
            scenario.add_function(Function.ksfailure())
            scenario.add_function(Function.mass())
        finally:
            Scenario.AUTO_REORDER_FUNCTIONS = True

        assert [func.name for func in scenario.functions] == ["ksfailure", "mass"]
        assert [func.id for func in scenario.functions] == [1, 2]

    def test_composite_function_rejected(self):
        """composite functions register to the model, not to a scenario"""
        scenario = Scenario(name="cruise", steps=10)
        ksfailure = Function.ksfailure()
        scenario.add_function(ksfailure)

        composite = ksfailure * 2.0
        with self.assertRaises(TypeError):
            scenario.add_function(composite)


if __name__ == "__main__":
    unittest.main()
