__all__ = ["TestAeroOnewayDriver"]

from pyfuntofem.interface import TestAerodynamicSolver


class TestAeroOnewayDriver:
    def __init__(self, solvers, model, transfer_settings=None):
        """
        Aerodynamic driver for unittests (similar to Fun3dOnewayDriver)
        Requires TestAeroSolver in self.flow

        Parameters
        ----------
        solvers: :class:`~interface.solver_manager.SolverManager`
            The various disciplinary solvers.
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data.
        """
        self.solvers = solvers
        self.model = model

        assert isinstance(solvers.flow, TestAerodynamicSolver)
        self.aero_solver = solvers.flow

        comm = solvers.comm
        comm_manager = solvers.comm_manager

        # initialize variables
        for body in self.model.bodies:
            # transfer to fixed structural loads in case the user got only aero loads from the Fun3dOnewayDriver
            body.initialize_transfer(
                comm=comm,
                struct_comm=comm_manager.struct_comm,
                struct_root=comm_manager.struct_root,
                aero_comm=comm_manager.aero_comm,
                aero_root=comm_manager.aero_root,
                transfer_settings=transfer_settings,
            )
            for scenario in model.scenarios:
                body.initialize_variables(scenario)

        # check for unsteady problems
        self._unsteady = False
        for scenario in model.scenarios:
            if not scenario.steady:
                self._unsteady = True
                break

    @property
    def steady(self) -> bool:
        return not (self._unsteady)

    @property
    def unsteady(self) -> bool:
        return self._unsteady

    def solve_forward(self):
        bodies = self.model.bodies
        for scenario in self.model.scenarios:
            # set functions and variables
            self.aero_solver.set_variables(scenario, bodies)
            self.aero_solver.set_functions(scenario, bodies)

            # run the forward analysis via iterate
            self.aero_solver.initialize(scenario, bodies)
            for step in range(1, scenario.steps + 1):
                self.aero_solver.iterate(scenario, bodies, step=step)
            self.aero_solver.post(scenario, bodies)

            # get functions to store the function values into the model
            self.aero_solver.get_functions(scenario, bodies)
        return

    def solve_adjoint(self):
        bodies = self.model.bodies
        # run the adjoint aerodynamic analysis
        functions = self.model.get_functions()

        # Zero the derivative values stored in the function
        for func in functions:
            func.zero_derivatives()

        for scenario in self.model.scenarios:
            # set functions and variables
            self.aero_solver.set_variables(scenario, bodies)
            self.aero_solver.set_functions(scenario, bodies)

            # zero all coupled adjoint variables in the body
            for body in bodies:
                body.initialize_adjoint_variables(scenario)

            # initialize, run, and do post adjoint
            self.aero_solver.initialize_adjoint(scenario, bodies)
            for step in range(1, scenario.steps + 1):
                self.aero_solver.iterate_adjoint(scenario, bodies, step=step)
            self.aero_solver.post_adjoint(scenario, bodies)

            # call get function gradients to store the gradients w.r.t. aero DVs from FUN3D
            self.aero_solver.get_function_gradients(scenario, bodies)
