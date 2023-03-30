import unittest
from pyfuntofem.model import Function, CompositeFunction


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


if __name__ == "__main__":
    unittest.main()
