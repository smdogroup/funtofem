# -------------------------------------------------
# Relaxation scheme objects used in FUNtoFEM.
# Primarily written to be used with TACS interface.
# Previously, relaxations schemes were located in body.
# -------------------------------------------------

__all__ = ["AitkenRelaxationTacs", "UnderRelaxationTacs"]


class AitkenRelaxationTacs:
    """
    Class to define Aitken relaxation settings.
    """

    def __init__(
        self,
        theta_init=1.0,
        theta_min=0.25,
        theta_max=2.0,
        aitken_tol=1e-13,
        debug=False,
        debug_more=False,
        history_file=None,
    ):
        """
        Construct an Aitken relaxation setting object.

        Parameters
        ----------
        theta_init : float
            Initial relaxation parameter.
        theta_min : float
            Minimum relaxation parameter. Defaults to 0.25.
        theta_max : float
            Maximum relaxation parameter. Defaults to 2.0.
        aitken_tol : float
            Tolerance for the denominator in Aitken equation.
            When met, Aitken relaxation stops and theta is set to unity.
        """
        self.theta_init = theta_init
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.aitken_tol = aitken_tol
        self.aitken_debug = debug
        self.aitken_debug_more = debug_more

        self.write_history_flag = False
        self.history_file = history_file

        if self.history_file is not None:
            self.write_history_flag = True

        return
    
class UnderRelaxationTacs:
    def __init__(self, theta_forward=1.0, theta_adjoint=1.0):
        self.theta_forward = theta_forward
        self.theta_adjoint = theta_adjoint
