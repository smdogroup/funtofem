from pyfuntofem.solver_interface import *
from pyfuntofem import TransferScheme
import numpy as np


class FakeAerodynamics(SolverInterface):
    def __init__(self, comm, model, flow_dt=1.0):
        self.c = -100.0

        # Create fake aero_mesh, semi-ellipse centered at (0.5, 0.0)
        model.bodies[0].aero_nnodes = 101
        model.bodies[0].aero_X = np.zeros(
            3 * model.bodies[0].aero_nnodes, dtype=TransferScheme.dtype
        )

        a = 0.5
        b = 0.1
        theta = np.linspace(0.0, np.pi, model.bodies[0].aero_nnodes)

        x = 0.5 - a * np.cos(theta)
        y = np.zeros(model.bodies[0].aero_nnodes)
        z = 0.0 + b * np.sin(theta)

        model.bodies[0].aero_X[0::3] = x[:]
        model.bodies[0].aero_X[1::3] = y[:]
        model.bodies[0].aero_X[2::3] = z[:]

        # Unsteady scenarios
        self.force_hist = {}
        for scenario in model.scenarios:
            self.force_hist[scenario.id] = {}

        self.psi_A = None

        self.fixed_step = 2
        self.fixed_step_psi_A = None

        return

    def initialize(self, scenario, bodies, first_pass=False):
        self.sum_states = 0.0

        return 0

    def get_functions(self, scenario, bodies):
        for function in scenario.functions:
            if (
                function.analysis_type == "aerodynamic"
                and "displacement" in function.name
            ):
                function.value = self.sum_states

    def iterate(self, scenario, bodies, step):
        # qval = scenario.variables['aerodynamic'][6].value

        # Get the aerodynamic node displacements
        Ua = bodies[0].aero_disps

        # Set the aerodynamic loads as a function of the displacements
        bodies[0].aero_loads = (
            -self.c * Ua
        )  # + qval*np.ones(Ua.shape, dtype=TransferScheme.dtype)

        # Add aerodynamic displacements to function
        if step == self.fixed_step:
            self.sum_states += np.sum(Ua)

        # Save this steps forces for the adjoint
        self.force_hist[scenario.id][step] = {}
        self.force_hist[scenario.id][step][bodies[0].id] = bodies[0].aero_loads.copy()

        return 0

    def set_states(self, scenario, bodies, step):
        bodies[0].aero_loads = self.force_hist[scenario.id][step][bodies[0].id]

    def intialize_adjoint(self, scenario, bodies):
        pass

        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        for body in bodies:
            self.psi_A = -body.dLdfa.copy()
            body.dGdua[:, 0] = self.c * self.psi_A.flatten()

            if step == self.fixed_step:
                self.fixed_step_psi_A = self.psi_A.copy()

                for ifunc, func in enumerate(scenario.functions):
                    if (
                        func.analysis_type == "aerodynamic"
                        and "displacement" in func.name
                    ):
                        body.dGdua[:, ifunc] += 1.0

        return 0

    def get_function_gradients(self, scenario, bodies, offset):
        for func, function in enumerate(scenario.functions):
            for body in bodies:
                for vartype in scenario.variables:
                    if vartype == "aerodynamic":
                        for i, var in enumerate(scenario.variables[vartype]):
                            if var.active and "force" in var.name:
                                scenario.derivatives[vartype][offset + func][
                                    i
                                ] = -np.sum(self.fixed_step_psi_A)

        return 0
