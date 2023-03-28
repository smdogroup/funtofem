__all__ = ["CompositeFunction"]

import numpy as np


class CompositeFunction:
    def __init__(self, name: str, eval_hdl, functions, optim=False):
        """
        Define a function dependent on the analysis functions above

        Parameters
        ----------
        eval_hdl: lambda expression
            takes in dictionary of function names and returns the value
        functions: list[Function]
            list of the function objects Function/CompositeFunction used for the composite function class
        optim: bool
            whether to include this function in the optimization objective/constraints
            (can be active but not an objective/constraint if it is used to compute composite functions)
        """

        self._name = name
        self.eval_hdl = eval_hdl
        self.functions = functions
        self.optim = optim

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

    def optimize(self):
        """
        set optim to True indicating this function will be an objective or constraint
        """
        self.optim = True
        return self  # return function for method cascading

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
        # TODO : actually write the correct equation in here
        return (lift / drag + mass + non_wing_mass).set_name("togw")

    """
    this next block of code overloads the arithmetic operations +-*/ and **
    for operations on composite functions or regular analysis functions
    Also common mathematical functions are included as class methods such as
    exp(), log(), etc.
    """

    @classmethod
    def exp(cls, func):
        """compute composite function for exp(func)"""
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return np.exp(funcs_dict[func.full_name])

            func_name = func.full_name
            functions = [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return np.exp(func.eval_hdl(funcs_dict))

            func_name = func.name
            functions = func.functions
        else:
            raise AssertionError(
                "Can't take exp(func) for non func/composite func object."
            )
        return cls(name=func_name, eval_hdl=eval_hdl, functions=functions)

    @classmethod
    def log(cls, func):
        """compute composite function for ln(func) the natural log"""
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return np.log(funcs_dict[func.full_name])

            func_name = func.full_name
            functions = [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return np.log(func.eval_hdl(funcs_dict))

            func_name = func.name
            functions = func.functions
        else:
            raise AssertionError(
                "Can't take log(func) for non func/composite func object."
            )
        return cls(name=func_name, eval_hdl=eval_hdl, functions=functions)

    @classmethod
    def boltz_max(cls, functions, rho=1):
        """
        compute composite function for boltzmann_maximum(functions)
        boltz_max(vec x) = sum(x_i * exp(rho*x_i)) / sum(exp(rho*x_i))
        """
        from .function import Function

        def eval_hdl(funcs_dict):
            numerator = 0.0
            denominator = 0.0
            for func in functions:
                if isinstance(func, Function):
                    value = funcs_dict[func.full_name]
                    numerator += value * np.exp(rho * value)
                    denominator += np.exp(rho * value)
                elif isinstance(func, CompositeFunction):
                    value = func.eval_hdl(
                        funcs_dict
                    )  # chain the functions dict into the composite func
                    numerator += value * np.exp(rho * value)
                    denominator += np.exp(rho * value)
                # no need for else condition here since this is evaluated later and would be caught below
            return numerator / denominator

        func_name = ""
        for func in functions:
            assert isinstance(func, Function) or isinstance(func, CompositeFunction)
            func_name += func.full_name
        return cls(name=func_name, eval_hdl=eval_hdl, functions=functions)

    @classmethod
    def boltz_min(cls, functions, rho=1):
        """
        compute composite function for boltzman_min(functions)
        boltz_min(vec x) = sum(x_i * exp(-rho*x_i)) / sum(exp(-rho*x_i))
        """
        # negative sign on rho makes it a minimum
        return cls.boltz_max(functions, rho=-rho)

    def __add__(self, func):
        from .function import Function

        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) + funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) + func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) + func

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
                return self.eval_hdl(funcs_dict) + funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) + func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) + func

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
                return self.eval_hdl(funcs_dict) - funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) - func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) - func

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
                return self.eval_hdl(funcs_dict) - funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) - func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) - func

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
                return self.eval_hdl(funcs_dict) * funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) * func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) * func

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
                return self.eval_hdl(funcs_dict) * funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) * func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) * func

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
                return self.eval_hdl(funcs_dict) / funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) / func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) / func

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
                return self.eval_hdl(funcs_dict) / funcs_dict[func.full_name]

            func_name = func.name
            functions = self.functions + [func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) / func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = self.functions + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return self.eval_hdl(funcs_dict) / func

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
                return self.eval_hdl(funcs_dict) ** func

            func_name = "float"
            functions = self.functions
        else:
            raise AssertionError("Division Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}**{func_name}", eval_hdl=eval_hdl, functions=functions
        )
