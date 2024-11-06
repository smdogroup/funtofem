__all__ = ["TestStructuralSolver"]

import numpy as np
from funtofem import TransferScheme
from .._solver_interface import SolverInterface
import os


class TestStructuralSolver(SolverInterface):
    def __init__(self, comm, model, elastic_k=1.0, thermal_k=1.0, default_mesh=True):
        """
        A test solver that provides the functionality that FUNtoFEM expects from
        a structural solver.

        Forward analysis
        ----------------


        Adjoint analysis
        ----------------


        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        """

        self.comm = comm
        self.npts = 25
        np.random.seed(54321)

        # setup forward and adjoint tolerances
        super().__init__()

        # Get the list of active design variables
        self.variables = model.get_variables()

        # Count the number of structural design variables (if any)
        self.struct_variables = []  # List of the variable objects

        # List of the variable values - converted into an numpy array
        self.struct_dvs = []

        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_variables.append(var)
                self.struct_dvs.append(var.value)

        # Allocate space for the aero dvs
        self.struct_dvs = np.array(self.struct_dvs, dtype=TransferScheme.dtype)

        # elastic and thermal scales 1/stiffness
        elastic_scale = 1.0 / elastic_k
        thermal_scale = 1.0 / thermal_k

        # scenario data for the multi scenario case
        class ScenarioData:
            def __init__(self, npts, struct_dvs):
                self.npts = npts
                self.struct_dvs = struct_dvs

                # choose time step
                self.dt = 0.01

                # Struct disps = Jac1 * struct_forces + b1 * struct_X + c1 * struct_dvs + omega1 * time
                self.Jac1 = (
                    0.01
                    * elastic_scale
                    * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
                )
                self.b1 = (
                    0.01
                    * elastic_scale
                    * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
                )
                self.c1 = (
                    0.01
                    * elastic_scale
                    * (np.random.rand(3 * self.npts, len(self.struct_dvs)) - 0.5)
                )

                # Struct temps = Jac2 * struct_flux + b2 * struct_X + c2 * struct_dvs + omega2 * time
                self.Jac2 = (
                    0.05 * thermal_scale * (np.random.rand(self.npts, self.npts) + 0.5)
                )
                self.b2 = (
                    0.1
                    * thermal_scale
                    * (np.random.rand(self.npts, 3 * self.npts) - 0.5*0)
                )
                self.c2 = (
                    0.5
                    * thermal_scale
                    * (np.random.rand(self.npts, len(self.struct_dvs)) - 0.5*0)
                )

                # Data for output functional values
                self.func_coefs1 = np.random.rand(3 * self.npts)
                self.func_coefs2 = np.random.rand(self.npts)

                # unsteady state variable drift
                rate = 0.001
                self.omega1 = rate * (np.random.rand(3 * self.npts) - 0.5)
                self.omega2 = rate * (np.random.rand(self.npts) - 0.5)

        # create scenario data for each scenario
        self.scenario_data = {}
        for scenario in model.scenarios:
            self.scenario_data[scenario.id] = ScenarioData(self.npts, self.struct_dvs)

        if default_mesh:
            # Set random initial node locations
            self.struct_X = np.random.rand(3 * self.npts).astype(TransferScheme.dtype)

            # Initialize the coordinates of the structural mesh
            struct_id = np.arange(1, self.npts + 1)
            for body in model.bodies:
                body.initialize_struct_nodes(self.struct_X, struct_id)

        else:
            for body in model.bodies:
                # Make TACS Interface first to initialize struct mesh into bodies
                self.struct_X = body.struct_X.copy()

        return

    def set_variables(self, scenario, bodies):
        """Set the design variables for the solver"""

        # Set the aerodynamic design variables
        index = 0
        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_dvs[index] = var.value
                index += 1

        return

    def set_functions(self, scenario, bodies):
        """
        Set the functions to be used for the given scenario.

        In this function, each discipline should initialize the data needed to evaluate
        the given set of functions set in each scenario.
        """

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the functions of interest and set the function values into
        the scenario.functions objects
        """

        time_index = 0 if scenario.steady else scenario.steps

        for func in scenario.functions:
            if func.analysis_type == "structural":
                value = 0.0
                for body in bodies:
                    struct_loads = body.get_struct_loads(scenario, time_index)
                    if struct_loads is not None:
                        value += np.dot(
                            self.scenario_data[scenario.id].func_coefs1, struct_loads
                        )
                    struct_flux = body.get_struct_heat_flux(scenario, time_index)
                    if struct_flux is not None:
                        value += np.dot(
                            self.scenario_data[scenario.id].func_coefs2, struct_flux
                        )
                func.value = self.comm.allreduce(value)
        return

    def get_function_gradients(self, scenario, bodies):
        """
        Evaluate the function gradients and set them into the function classes.

        Note: The function gradients can be evaluated elsewhere (for instance in
        post_adjoint(). This function must get these values and place them into the
        associated function.)
        """

        # Set the derivatives of the functions for the given scenario
        for findex, func in enumerate(scenario.adjoint_functions):
            for vindex, var in enumerate(self.struct_variables):
                for body in bodies:
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        value = np.dot(
                            struct_disps_ajp[:, findex],
                            self.scenario_data[scenario.id].c1[:, vindex],
                        )
                        func.add_gradient_component(var, value)

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        value = np.dot(
                            struct_temps_ajp[:, findex],
                            self.scenario_data[scenario.id].c2[:, vindex],
                        )
                        func.add_gradient_component(var, value)

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Add the contributions to the gradient w.r.t. the structural coordinates
        """

        if step == 0:
            return

        adjoint_map = scenario.adjoint_map
        for ifunc, func in enumerate(scenario.adjoint_functions):
            ifull = adjoint_map[ifunc]
            for body in bodies:
                struct_shape_term = body.get_struct_coordinate_derivatives(scenario)
                struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                if struct_disps_ajp is not None:
                    struct_shape_term[:, ifull] += np.dot(
                        struct_disps_ajp[:, ifunc], self.scenario_data[scenario.id].b1
                    )

                struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                if struct_temps_ajp is not None:
                    struct_shape_term[:, ifull] += np.dot(
                        struct_temps_ajp[:, ifunc], self.scenario_data[scenario.id].b2
                    )

        return

    def initialize(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""

        # Initialize the coordinates of the aerodynamic or structural mesh
        for body in bodies:
            self.struct_X[:] = body.get_struct_nodes()

        return 0

    def iterate(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural solver

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        struct_time = step * self.scenario_data[scenario.id].dt

        for body in bodies:
            # Perform the "analysis"
            struct_loads = body.get_struct_loads(scenario, step)
            struct_disps = body.get_struct_disps(scenario, step)
            if struct_loads is not None:
                struct_disps[:] = np.dot(
                    self.scenario_data[scenario.id].Jac1, struct_loads
                )
                struct_disps[:] += np.dot(
                    self.scenario_data[scenario.id].b1, self.struct_X
                )
                struct_disps[:] += np.dot(
                    self.scenario_data[scenario.id].c1, self.struct_dvs
                )
                if not scenario.steady:
                    struct_disps[:] += (
                        self.scenario_data[scenario.id].omega1 * struct_time
                    )

            # Perform the heat transfer "analysis"
            struct_flux = body.get_struct_heat_flux(scenario, step)
            struct_temps = body.get_struct_temps(scenario, step)
            if struct_flux is not None:
                struct_temps[:] = np.dot(
                    self.scenario_data[scenario.id].Jac2, struct_flux
                )
                struct_temps[:] += np.dot(
                    self.scenario_data[scenario.id].b2, self.struct_X
                )
                struct_temps[:] += np.dot(
                    self.scenario_data[scenario.id].c2, self.struct_dvs
                )
                if not scenario.steady:
                    struct_temps[:] += (
                        self.scenario_data[scenario.id].omega2 * struct_time
                    )

        # This analysis is always successful so return fail = 0
        fail = 0
        return fail

    def post(self, scenario, bodies):
        pass

    def initialize_adjoint(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""
        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural adjoint

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        # If the scenario is unsteady only add the rhs for the final state
        include_rhs = True
        if not scenario.steady and step != scenario.steps:
            include_rhs = False

        for body in bodies:
            struct_disps_ajp = body.get_struct_disps_ajp(scenario)
            struct_loads_ajp = body.get_struct_loads_ajp(scenario)
            if struct_disps_ajp is not None:
                for k, func in enumerate(scenario.adjoint_functions):
                    struct_loads_ajp[:, k] = np.dot(
                        self.scenario_data[scenario.id].Jac1.T, struct_disps_ajp[:, k]
                    )
                    if include_rhs and func.analysis_type == "structural":
                        struct_loads_ajp[:, k] += self.scenario_data[
                            scenario.id
                        ].func_coefs1

            struct_temps_ajp = body.get_struct_temps_ajp(scenario)
            struct_flux_ajp = body.get_struct_heat_flux_ajp(scenario)

            if struct_temps_ajp is not None:
                for k, func in enumerate(scenario.adjoint_functions):
                    struct_flux_ajp[:, k] = np.dot(
                        self.scenario_data[scenario.id].Jac2.T, struct_temps_ajp[:, k]
                    )
                    if include_rhs and func.analysis_type == "structural":
                        struct_flux_ajp[:, k] += self.scenario_data[
                            scenario.id
                        ].func_coefs2

        # add derivative values
        if not scenario.steady:
            if step > 0:
                self.get_function_gradients(scenario, bodies)
            else:  # step == 0
                # want to zero out adjoints used for derivatives, since no analysis done on step 0
                for body in bodies:
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        struct_disps_ajp *= 0.0

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        struct_temps_ajp *= 0.0
        fail = 0
        return fail

    def post_adjoint(self, scenario, bodies):
        return
