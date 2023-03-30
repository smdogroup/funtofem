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

    def test_struct_coordinates(self, test_name, body=None):
        """test the structure coordinate derivatives struct_X with complex step"""
        # assumes only one body
        if body is None:
            body = self.model.bodies[0]
        struct_X = body.struct_X

        # random covariant tensor to aggregate derivative error among one or more functions
        first_scenario = self.model.scenarios[0]
        nfunc = first_scenario.count_adjoint_functions()
        dL_dfunc_single = np.random.rand(nfunc)
        dL_dfunc_single_col = np.reshape(dL_dfunc_single, newshape=(nfunc, 1))

        ncomp = len(self.model.composite_functions)
        dL_dfunc_comp = np.random.rand(ncomp)
        dL_dfunc_comp_col = np.reshape(dL_dfunc_comp, (ncomp, 1))
        # for scenario in self.model.scenarios:
        #     nfunc = scenario.count_adjoint_functions()
        #     dL_dfunc[scenario.id] = np.random.rand(nfunc)
        #     dL_dfunc_col[scenario.id] = np.reshape(dL_dfunc[scenario.id], newshape=(nfunc, 1))

        # random contravariant tensor d(struct_X)/ds for testing struct shape
        dstructX_ds = np.random.rand(struct_X.shape[0])
        dstructX_ds_row = np.reshape(dstructX_ds, newshape=(1, dstructX_ds.shape[0]))

        """Adjoint method to compute coordinate derivatives and TD"""
        self.driver.solve_forward()
        self.driver.solve_adjoint()
        self.model.evaluate_composite_functions(compute_xpt=True)

        # add coordinate derivatives among scenarios
        adjoint_TD = 0.0

        # for scenario in self.model.scenarios:
        struct_shape_term = body.get_struct_coordinate_derivatives(first_scenario)
        # add in struct coordinate derivatives of this scenario
        adjoint_TD += (dstructX_ds_row @ struct_shape_term @ dL_dfunc_single_col).real

        if ncomp > 0:
            comp_struct_shape_term = body.composite_struct_shape_term
            # add in composite functions coordinate derivatives of this scenario
            adjoint_TD += (
                dstructX_ds_row @ comp_struct_shape_term @ dL_dfunc_comp_col
            ).real

        adjoint_TD = float(adjoint_TD)

        """Complex step to compute coordinate total derivatives"""
        # perturb the coordinate derivatives
        struct_X += 1j * self.epsilon * dstructX_ds
        self.driver.solve_forward()
        self.model.evaluate_composite_functions(compute_grad=False, compute_xpt=False)

        # dfunc_ds = np.array(
        #     [func.value.imag / self.epsilon for func in self.model.get_functions()]
        # )
        dfunc_ds = np.array(
            [func.value.imag / self.epsilon for func in first_scenario.functions]
        )
        complex_step_TD = np.sum(dL_dfunc_single * dfunc_ds)

        # add in composite function terms
        if ncomp > 0:
            dcomp_ds = np.array(
                [
                    func.value.imag / self.epsilon
                    for func in self.model.composite_functions
                ]
            )
            complex_step_TD += np.sum(dL_dfunc_comp * dcomp_ds)

        rel_error = TestResult.relative_error(adjoint_TD, complex_step_TD)
        print(f"\n{test_name}")
        print(f"\tadjoint TD = {adjoint_TD}")
        print(f"\tcomplex step TD = {complex_step_TD}")
        print(f"\trel error = {rel_error}\n")
        return rel_error
