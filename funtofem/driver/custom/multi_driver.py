
__all__ = ["MultiDriver"]

class MultiDriver:
    def __init__(self, driver_list:list):
        """
        call solve_forward, solve_adjoint on multiple drivers

        useful if one scenario uses a coupled driver and the
        other uses an uncoupled driver

        in this way, we can include a mixture of each by combining these
        drivers into one and still using the OptimizationManager
        """
        self.driver_list = driver_list

        # copy comm and solvers from the first driver
        # since these are used in optimizationManager
        first_driver = self.driver_list[0]
        self.comm = first_driver.comm
        self.solvers = first_driver.solvers
        self.model = first_driver.model

    def solve_forward(self):
        driver_list = self.driver_list
        for driver in driver_list:
            driver.solve_forward()

    def solve_adjoint(self):
        self._zero_derivatives()
        driver_list = self.driver_list
        for driver in driver_list:
            driver.solve_adjoint()

    def _zero_derivatives(self):
        """zero all model derivatives"""
        # TODO : only zero derivatives in coupled scenarios when using
        model = self.driver_list[0].model
        for func in model.get_functions(all=True):
            for var in self.model.get_variables():
                func.derivatives[var] = 0.0
        return
