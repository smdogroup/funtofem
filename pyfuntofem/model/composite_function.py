__all__ = ["CompositeFunction"]

import numpy as np


class CompositeFunction:
    def __init__(self, name: str, eval_hdl, functions):
        """
        Define a function dependent on the analysis functions above

        Parameters
        ----------
        eval_hdl: lambda expression
            takes in dictionary of function names and returns the value
        functions: list[Function]
            list of the function objects Function/CompositeFunction used for the composite function class
        """

        self._name = name
        self.eval_hdl = eval_hdl
        self.functions = functions

        # Store the value of the function here
        self._eval_forward = False
        self._eval_deriv = False
        self.value = None

        # Store the values of the derivatives w.r.t. this function
        self.derivatives = {}

    def reset(self):
        """reset the function for a new analysis"""
        self._eval_forward = False
        self._eval_deriv = False
        return

    @property
    def funcs(self) -> dict:
        """dictionary of function values we compute from"""
        funcs = {}
        for function in self.functions:
            if isinstance(function, CompositeFunction):
                if not function._eval_forward:
                    function.evaluate()
            funcs[function.full_name] = function.value
        return funcs

    @property
    def function_names(self):
        return [func.full_name for func in self.functions]

    def evaluate(self, funcs=None):
        """compute the value of the composite function from the other functions"""
        save_value = False  # only save value when calling with default funcs
        if funcs is None:
            funcs = self.funcs
            save_value = True
            if self._eval_forward:  # exit early if already evaluated
                return
        value = self.eval_hdl(funcs)
        if save_value:
            self.value = value
        self._eval_forward = True
        return value

    def evaluate_gradient(self):
        """compute derivatives of the composite function for each variable"""
        if self._eval_deriv:  # exit early if already evaluated
            return

        self.complex_step_dict()

        # make sure dependent functions have their derivatives evaluated
        for function in self.functions:
            if isinstance(function, CompositeFunction):
                function.evaluate_derivatives()

        # get list of variables from one of the previously evaluated functions
        varlist = [var for var in self.functions[0].derivatives]

        # iteratively compute total derivatives df/dg_i * dg_i/dx_j
        for var in varlist:
            df_dx = 0.0
            for gi in self.functions:
                df_dgi = self.df_dgi[gi.full_name]
                dgi_dx = gi.derivatives[var]
                df_dx += df_dgi * dgi_dx
            self.derivatives[var] = df_dx
        return

    def get_gradient_component(self, var):
        """
        Get the gradient value stored - return 0 if not defined

        Parameter
        ---------
        var: Variable object
            Derivative of this function w.r.t. the given variable
        """

        if var in self.derivatives:
            return self.derivatives[var]

        return 0.0

    def complex_step_dict(self):
        """compute function dictionary df/dg_i for derivatives w.r.t. analysis functions"""
        # use complex step on the forward evaluation handle to get df/dg_i with
        # f this composite function and g_i the other functions f(g_i)
        h = 1e-30
        self.df_dgi = {key: None for key in self.funcs}
        for key in self.funcs:
            pert_funcs = self.funcs
            pert_funcs[key] += h * 1j
            # df / dg_i where f is this function and g_i are dependent functions
            self.df_dgi[key] = self.evaluate(pert_funcs).imag / h
        return

    def register_to(self, funtofem_model):
        """register the composite functions to the overall model"""
        funtofem_model.add_composite_function(self)
        return self  # for method cascading

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name
        return

    def set_name(self, new_name: str):
        self._name = new_name
        return self  # method cascading

    @property
    def full_name(self) -> str:
        return self.name

    @classmethod
    def takeoff_gross_weight(cls, lift, drag, mass, non_wing_mass):
        return (lift / drag + mass + non_wing_mass).set_name("togw")

    def __add__(self, func):
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError("Addition Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}+{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __radd__(self, func):
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError(
                "Reflected Addition Overload failed for unsupported type."
            )
        return CompositeFunction(
            name=f"{self.name}+{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __sub__(self, func):
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError("Subtraction Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}-{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __rsub__(self, func):
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError(
                "Reflected Subtraction Overload failed for unsupported type."
            )
        return CompositeFunction(
            name=f"{self.name}-{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __mul__(self, func):
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError("Multiple Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}*{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __rmul__(self, func):
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError(
                "Reflected Multiple Overload failed for unsupported type."
            )
        return CompositeFunction(
            name=f"{self.name}*{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __truediv__(self, func):
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError("Division Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}/{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __rtruediv__(self, func):
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError("Division Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}/{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __pow__(self, func):
        from .function import Function

        if isinstance(func, Function) or isinstance(func, CompositeFunction):
            raise AssertionError("Don't raise a function to a function power.")
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] ** func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError("Division Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}**{func_name}", eval_hdl=eval_hdl, functions=functions
        )
