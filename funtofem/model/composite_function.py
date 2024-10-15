__all__ = ["CompositeFunction"]

import numpy as np


def unique(vec):
    """make a unique version of list"""
    output = []
    for elem in vec:
        if not (elem in output):
            output += [elem]
    return output


class CompositeFunction:
    def __init__(
        self,
        name: str,
        eval_hdl,
        functions,
        variables=[],
        optim=False,
        plot_name: str = None,
        plot: bool = False,
    ):
        """
        Define a function dependent on the analysis functions above

        Parameters
        ----------
        eval_hdl: lambda expression
            takes in dictionary of function names and returns the value
        functions: list[Function]
            list of the function objects Function/CompositeFunction used for the composite function class
        variables: list[Variable]
            list of the variable objects used in evaluating the CompositeFunction object
        optim: bool
            whether to include this function in the optimization objective/constraints
            (can be active but not an objective/constraint if it is used to compute composite functions)
        plot: bool
            whether to include this function in optimization plots
        plot_name: str
            what name to give for optimization plots
        """

        self._name = name
        self.eval_hdl = eval_hdl
        self.functions = unique(functions)
        self.variables = unique(variables)
        self.optim = optim
        self.analysis_type = "composite"

        # optimization settings
        self.lower = None
        self.upper = None
        self.scale = None
        self._objective = False
        self._plot = False
        self._plot_name = plot_name

        # Store the value of the function here
        self._eval_forward = False
        self._eval_deriv = False
        self._done_complex_step = False
        self.value = None

        # Store the values of the derivatives w.r.t. this function
        self.derivatives = {}
        self.df_dgi = None

    @property
    def optim_derivatives(self):
        optim_derivs = {
            var: self.derivatives[var] for var in self.derivatives if not (var.state)
        }
        # print(f"optim derivatives = {optim_derivs}")
        return optim_derivs

    @classmethod
    def external(cls, name, optim=False, plot_name=None):
        return cls(
            name=name,
            eval_hdl=None,
            functions=[],
            variables=[],
            optim=optim,
            plot_name=plot_name,
        )

    def reset(self):
        """reset the function for a new analysis"""
        self._eval_forward = False
        self._eval_deriv = False
        self._done_complex_step = False
        return

    @property
    def vars_only(self) -> bool:
        """used for adjacency constraint functions"""
        return len(self.functions) == 0 and len(self.variables) >= 0

    @property
    def sparse_gradient(self):
        """used for adjacency constraints, vars only functions"""
        self.evaluate_gradient()
        np_array = np.array([self.derivatives[var] for var in self.optim_derivatives])
        # return csr_matrix(np_array, shape=(1,np_array.shape[0]))
        nvars = np_array.shape[0]
        cols = np.array(
            [
                ivar
                for ivar, var in enumerate(self.optim_derivatives)
                if self.derivatives[var] != 0.0
            ]
        )
        rows = np.array([0 for _ in range(cols.shape[0])])
        vals = np.array(
            [
                self.derivatives[var]
                for ivar, var in enumerate(self.optim_derivatives)
                if self.derivatives[var] != 0.0
            ]
        )
        return {
            "coo": [rows, cols, vals],
            "shape": (1, nvars),
        }

    def optimize(
        self,
        lower=None,
        upper=None,
        scale=None,
        objective=False,
        plot=False,
        plot_name: str = None,
    ):
        """
        automatically sets optim=True for optimization and sets optimization bounds for
        OpenMDAO or pyoptsparse
        """
        self.optim = True
        self.lower = lower
        self.upper = upper
        self.scale = scale
        self._objective = objective
        self._plot = plot
        if plot_name is not None:
            self._plot_name = plot_name
        return self

    @property
    def plot_name(self) -> str:
        if self._plot_name is not None:
            return self._plot_name
        else:
            return self.full_name

    def setup_sparse_gradient(self, f2f_model):
        """setup the sparse gradient for adjacency functions"""
        self.setup_derivative_dict(f2f_model.get_variables(optim=True))
        return self

    def setup_derivative_dict(self, variables):
        for var in variables:
            self.derivatives[var] = 0.0
        return

    def check_derivative_dict(self, variables):
        for var in variables:
            if not (var in self.derivatives):
                raise AssertionError(
                    f"Composite function {self.name} does not have variable {var.name} before it in the run script. All f2f variables need to be before composite functions are defined."
                )
        return

    @property
    def funcs(self) -> dict:
        """
        Dictionary of function values we compute from.
        """
        funcs = {}
        for function in self.functions:
            if isinstance(function, CompositeFunction):
                if not function._eval_forward:
                    function.evaluate()
            funcs[function.full_name] = function.value
        # also include variable names in the input dictionary funcs, acts like a function
        for variable in self.variables:
            funcs[variable.name] = variable.value
        return funcs

    @property
    def function_names(self):
        return [func.full_name for func in self.functions]

    def evaluate(self, funcs=None):
        """
        Compute the value of the composite function from the other functions.
        """
        if self.eval_hdl is None:
            return  # exit for external functions

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
        """
        Compute derivatives of the composite function for each variable.
        """
        if self.eval_hdl is None:
            return  # exit for external functions

        if self._eval_deriv:  # exit early if already evaluated
            return

        self.complex_step_dict()

        # iteratively compute total derivatives df/dg_i * dg_i/dx_j
        for var in self.derivatives:
            df_dx = 0.0
            for gi in self.functions:
                df_dgi = self.df_dgi[gi.full_name]
                dgi_dx = gi.derivatives[var]
                df_dx += df_dgi * dgi_dx
            for xi in self.variables:
                df_dxi = self.df_dgi[xi.name]
                if xi.name == var.name:
                    dxi_dvar = 1.0
                else:
                    dxi_dvar = 0.0
                df_dx += df_dxi * dxi_dvar
            self.derivatives[var] = df_dx
        return

    def get_gradient_component(self, var):
        """
        Get the gradient value stored - return 0 if not defined

        Parameters
        ----------
        var: Variable object
            Derivative of this function w.r.t. the given variable
        """

        if var in self.derivatives:
            return self.derivatives[var]

        return 0.0

    def directional_derivative(self, dvar_ds):
        """
        get the directional derivative df/ds in the direction of the test vector dvar/ds
        by the chain rule product df/dx * dx/ds

        Parameters
        ----------
        dvar_ds : dict
            dictionary of var_key : test vector entry
        """
        df_ds = 0.0
        for var in dvar_ds:
            # assumes that each variable is among the derivatives computed
            # by this function and other analysis functions
            df_ds += self.derivatives[var] * dvar_ds[var]
        return df_ds

    def complex_step_dict(self):
        """compute function dictionary df/dg_i for derivatives w.r.t. analysis functions"""
        # use complex step on the forward evaluation handle to get df/dg_i with
        # f this composite function and g_i the other functions f(g_i)
        # this now includes f(g_i,x_i) where x_i are variables
        if self._done_complex_step:
            return
        h = 1e-30
        self.df_dgi = {key: None for key in self.funcs}
        for key in self.funcs:
            pert_funcs = self.funcs
            pert_funcs[key] += h * 1j
            # df / dg_i where f is this function and g_i are dependent functions
            self.df_dgi[key] = self.evaluate(pert_funcs).imag / h
        self._done_complex_step = True
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
    def cast(cls, obj):
        """cast objects to a composite function"""
        from .function import Function
        from .variable import Variable

        if isinstance(obj, CompositeFunction):
            return obj
        elif isinstance(obj, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[obj.full_name]

            return cls(
                name=obj.name,
                eval_hdl=eval_hdl,
                functions=[obj],
                variables=[],
                optim=False,
            )
        elif isinstance(obj, Variable):

            def eval_hdl(funcs_dict):
                return funcs_dict[obj.name]

            return cls(
                name=obj.name,
                eval_hdl=eval_hdl,
                functions=[],
                variables=[obj],
                optim=False,
            )
        elif isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, complex):
            # cast a float,int,complex to a CompositeFunction
            def eval_hdl(funcs_dict):
                return obj

            return cls(name="float", eval_hdl=eval_hdl, functions=[], variables=[])
        else:
            raise AssertionError(
                "Object is not setup to be casted to a composite function."
            )

    @classmethod
    def combine(cls, name, eval_hdl, func1, func2):
        """combine two CompositeFunctions with a newly defined eval_hdl, mainly for under-the-hood arithmetic"""
        assert isinstance(func1, CompositeFunction)
        assert isinstance(func2, CompositeFunction)
        functions = func1.functions + func2.functions
        variables = func1.variables + func2.variables
        return cls(
            name=name, eval_hdl=eval_hdl, functions=functions, variables=variables
        )

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
    def abs(cls, func):
        """compute composite function for abs(func)"""

        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            value = func.eval_hdl(funcs_dict)
            if value.real >= 0:
                return value
            else:
                return -value

        return cls(
            name=func.name,
            eval_hdl=eval_hdl,
            functions=func.functions,
            variables=func.variables,
        )

    @classmethod
    def exp(cls, func):
        """compute composite function for exp(func)"""

        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return np.exp(func.eval_hdl(funcs_dict))

        return cls(
            name=func.name,
            eval_hdl=eval_hdl,
            functions=func.functions,
            variables=func.variables,
        )

    @classmethod
    def sin(cls, func):
        """compute composite function for sin(func)"""

        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return np.sin(func.eval_hdl(funcs_dict))

        return cls(
            name=func.name,
            eval_hdl=eval_hdl,
            functions=func.functions,
            variables=func.variables,
        )

    @classmethod
    def cos(cls, func):
        """compute composite function for sin(func)"""

        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return np.cos(func.eval_hdl(funcs_dict))

        return cls(
            name=func.name,
            eval_hdl=eval_hdl,
            functions=func.functions,
            variables=func.variables,
        )

    @classmethod
    def tan(cls, func):
        """compute composite function for sin(func)"""

        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return np.tan(func.eval_hdl(funcs_dict))

        return cls(
            name=func.name,
            eval_hdl=eval_hdl,
            functions=func.functions,
            variables=func.variables,
        )

    @classmethod
    def log(cls, func):
        """compute composite function for ln(func) the natural log"""
        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return np.log(func.eval_hdl(funcs_dict))

        return cls(
            name=func.name,
            eval_hdl=eval_hdl,
            functions=func.functions,
            variables=func.variables,
        )

    @classmethod
    def boltz_max(cls, functions, rho=1):
        """
        compute composite function for boltzmann_maximum(functions)
        boltz_max(vec x) = sum(x_i * exp(rho*x_i)) / sum(exp(rho*x_i))
        """

        def eval_hdl(funcs_dict):
            numerator = 0.0
            denominator = 0.0
            for func in functions:
                func = CompositeFunction.cast(func)
                value = func.eval_hdl(
                    funcs_dict
                )  # chain the functions dict into the composite func
                numerator += value * np.exp(rho * value)
                denominator += np.exp(rho * value)
                # no need for else condition here since this is evaluated later and would be caught below
            return numerator / denominator

        func_name = ""
        variables = []
        for func in functions:
            func = CompositeFunction.cast(func)
            func_name += func.full_name
            variables += func.variables
        return cls(
            name=func_name, eval_hdl=eval_hdl, functions=functions, variables=variables
        )

    @classmethod
    def boltz_min(cls, functions, rho=1):
        """
        compute composite function for boltzman_min(functions)
        boltz_min(vec x) = sum(x_i * exp(-rho*x_i)) / sum(exp(-rho*x_i))
        """
        # negative sign on rho makes it a minimum
        return cls.boltz_max(functions, rho=-rho)

    def __add__(self, func):
        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return self.eval_hdl(funcs_dict) + func.eval_hdl(funcs_dict)

        return CompositeFunction.combine(
            name=f"{self.name}+{func.name}", eval_hdl=eval_hdl, func1=self, func2=func
        )

    def __radd__(self, func):
        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return self.eval_hdl(funcs_dict) + func.eval_hdl(funcs_dict)

        return CompositeFunction.combine(
            name=f"{func.name}+{self.name}", eval_hdl=eval_hdl, func1=self, func2=func
        )

    def __sub__(self, func):
        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return self.eval_hdl(funcs_dict) - func.eval_hdl(funcs_dict)

        return CompositeFunction.combine(
            name=f"{self.name}-{func.name}", eval_hdl=eval_hdl, func1=self, func2=func
        )

    def __rsub__(self, func):
        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return func.eval_hdl(funcs_dict) - self.eval_hdl(funcs_dict)

        return CompositeFunction.combine(
            name=f"{func.name}-{self.name}", eval_hdl=eval_hdl, func1=self, func2=func
        )

    def __mul__(self, func):
        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return self.eval_hdl(funcs_dict) * func.eval_hdl(funcs_dict)

        return CompositeFunction.combine(
            name=f"{self.name}*{func.name}", eval_hdl=eval_hdl, func1=self, func2=func
        )

    def __rmul__(self, func):
        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return func.eval_hdl(funcs_dict) * self.eval_hdl(funcs_dict)

        return CompositeFunction.combine(
            name=f"{func.name}*{self.name}", eval_hdl=eval_hdl, func1=self, func2=func
        )

    def __truediv__(self, func):
        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return self.eval_hdl(funcs_dict) / func.eval_hdl(funcs_dict)

        return CompositeFunction.combine(
            name=f"{self.name}/{func.name}", eval_hdl=eval_hdl, func1=self, func2=func
        )

    def __rtruediv__(self, func):
        func = CompositeFunction.cast(func)

        def eval_hdl(funcs_dict):
            return func.eval_hdl(funcs_dict) / self.eval_hdl(funcs_dict)

        return CompositeFunction.combine(
            name=f"{func.name}/{self.name}", eval_hdl=eval_hdl, func1=self, func2=func
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
            variables = self.variables
        else:
            raise AssertionError("Division Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}**{func_name}",
            eval_hdl=eval_hdl,
            functions=functions,
            variables=variables,
        )
