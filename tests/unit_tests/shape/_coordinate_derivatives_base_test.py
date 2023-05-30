from funtofem.interface import TestResult
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

    def test_struct_coordinates(self, test_name, body=None):
        """test the structure coordinate derivatives struct_X with complex step"""
        # assumes only one body
        if body is None:
            body = self.model.bodies[0]
        struct_X = body.struct_X

        # random covariant tensor to aggregate derivative error among one or more functions
        # compile full struct shape term
        nf = len(self.model.get_functions())
        dL_dfunc = np.random.rand(nf)
        dL_dfunc_col = np.reshape(dL_dfunc, newshape=(nf, 1))

        # random contravariant tensor d(struct_X)/ds for testing struct shape
        dstructX_ds = np.random.rand(struct_X.shape[0])
        dstructX_ds_row = np.reshape(dstructX_ds, newshape=(1, dstructX_ds.shape[0]))

        """Adjoint method to compute coordinate derivatives and TD"""
        self.driver.solve_forward()
        self.driver.solve_adjoint()

        # add coordinate derivatives among scenarios
        adjoint_TD = 0.0

        full_struct_shape_term = []
        for scenario in self.model.scenarios:
            struct_shape_term = body.get_struct_coordinate_derivatives(scenario)
            full_struct_shape_term.append(struct_shape_term)
        full_struct_shape_term = np.concatenate(full_struct_shape_term, axis=1)
        # add in struct coordinate derivatives of this scenario
        adjoint_TD += (dstructX_ds_row @ full_struct_shape_term @ dL_dfunc_col).real
        adjoint_TD = float(adjoint_TD)

        """Complex step to compute coordinate total derivatives"""
        # perturb the coordinate derivatives
        struct_X += 1j * self.epsilon * dstructX_ds
        self.driver.solve_forward()

        dfunc_ds = np.array(
            [func.value.imag / self.epsilon for func in self.model.get_functions()]
        )
        complex_step_TD = np.sum(dL_dfunc * dfunc_ds)

        rel_error = TestResult.relative_error(adjoint_TD, complex_step_TD)
        print(f"\n{test_name}")
        print(f"\tadjoint TD = {adjoint_TD}")
        print(f"\tcomplex step TD = {complex_step_TD}")
        print(f"\trel error = {rel_error}\n")
        return rel_error
