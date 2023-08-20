import unittest
from funtofem.model import Function, CompositeFunction, Variable


class TestCompositeFunction(unittest.TestCase):
    def test_single_operations(self):
        lift = Function.lift()
        drag = Function.drag()

        2 + lift
        lift + 2
        2 - lift
        2 * lift
        lift / 4
        2 / lift
        lift**2
        myfunc = lift * drag
        myfunc2 = lift / drag
        lift + drag
        lift - drag
        assert myfunc.name == f"{lift.name}*{drag.name}"
        assert myfunc2.name == f"{lift.name}/{drag.name}"
        return

    def test_multi_operations(self):
        lift = Function.lift()
        drag = Function.drag()
        mass = Function.mass()

        def eval_2(funcs_dict):
            return 2

        func2 = CompositeFunction(name="2", eval_hdl=eval_2, functions=[])
        lift + drag - mass
        myfunc = (lift * drag) ** 2
        print(f"(lift*drag)**2 name = {myfunc.full_name}")
        full_func = func2 * lift / mass - drag
        pred_full_func_name = f"{func2.name}*{lift.name}/{mass.name}-{drag.name}"
        print(f"full func name = {full_func.name}")
        print(f"pred func name = {pred_full_func_name}")
        assert full_func.full_name == pred_full_func_name
        return

    def test_with_variables(self):
        lift = Function.lift()
        drag = Function.drag()
        mass = Function.mass()

        my_comp = lift / drag * mass

        # add rib variables into our composite function
        nribs = 10
        for irib in range(1, nribs + 1):
            rib_var = Variable.structural(f"rib{irib}", value=0.1 * irib)
            my_comp = my_comp + rib_var.composite_function

        # multiply aoa var by our composite function
        aoa = Variable.aerodynamic("AOA", value=2.0)
        my_comp = my_comp * aoa.composite_function
        return


if __name__ == "__main__":
    unittest.main()
