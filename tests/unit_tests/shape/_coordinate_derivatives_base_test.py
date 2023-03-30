from pyfuntofem.interface import TestResult
import numpy as np


class CoordinateDerivativeTester:
    """
    Perform a complex step test over the coordinate derivatives of a driver
    """

    def __init__(self, driver, epsilon=1e-30):
        self.driver = driver
        self.epsilon = epsilon
        self.comm = self.driver.comm

    @property
    def flow_solver(self):
        return self.driver.solvers.flow

    @property
    def aero_X(self):
        """aero coordinate derivatives in FUN3D"""
        return self.flow_solver.aero_X

    @property
    def struct_solver(self):
        return self.driver.solvers.structural

    @property
    def struct_X(self):
        """structure coordinates in TACS"""
        return self.struct_solver.struct_X.getArray()

    @property
    def model(self):
        return self.driver.model

    def test_struct_coordinates(self, test_name, body=None, scenario=None):
        """test the structure coordinate derivatives struct_X with complex step"""
        if body is None:
            body = self.model.bodies[0]
        if scenario is None:
            scenario = self.model.scenarios[0]
        struct_X = body.struct_X

        # TODO : not sure if this will test multi-scenario coordinate derivatives correctly
        nfunc = scenario.count_adjoint_functions()
        dL_dfunc = np.random.rand(nfunc)
        dL_dfunc_col = np.reshape(dL_dfunc, newshape=(nfunc, 1))

        # random contravariant tensor d(struct_X)/ds for testing struct shape
        dstructX_ds = np.random.rand(struct_X.shape[0])
        dstructX_ds_row = np.reshape(dstructX_ds, newshape=(1, dstructX_ds.shape[0]))

        """Adjoint method to compute coordinate derivatives and TD"""
        self.driver.solve_forward()
        self.driver.solve_adjoint()

        struct_shape_term = body.get_struct_coordinate_derivatives(scenario)
        adjoint_TD = dstructX_ds_row @ struct_shape_term @ dL_dfunc_col
        adjoint_TD = float(adjoint_TD.real)

        """Complex step to compute coordinate total derivatives"""
        # perturb the coordinate derivatives
        struct_X += 1j * self.epsilon * dstructX_ds
        self.driver.solve_forward()
        functions = scenario.functions
        dfunc_ds = np.array([func.value.imag / self.epsilon for func in functions])
        complex_step_TD = np.sum(dL_dfunc * dfunc_ds)

        rel_error = TestResult.relative_error(adjoint_TD, complex_step_TD)
        print(f"\n{test_name}")
        print(f"\tadjoint TD = {adjoint_TD}")
        print(f"\tcomplex step TD = {complex_step_TD}")
        print(f"\trel error = {rel_error}\n")
        return rel_error
