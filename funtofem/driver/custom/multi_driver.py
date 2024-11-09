
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

    def solve_forward(self):
        driver_list = self.driver_list
        for driver in driver_list:
            driver.solve_forward()

    def solve_adjoint(self):
        driver_list = self.driver_list
        for driver in driver_list:
            driver.solve_adjoint()