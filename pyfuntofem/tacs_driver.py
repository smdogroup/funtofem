from .tacs_interface import TacsSteadyInterface, createTacsInterfaceFromBDF
from .funtofem_nlbgs_driver import FUNtoFEMnlbgs
import os
from mpi4py import MPI
import numpy as np


class TacsSteadyAnalysisDriver:
    """
    Class to perform only a TACS analysis with aerodynamic loads and heat fluxes in the body still retained.
    Similar to FUNtoFEMDriver class and FuntoFEMnlbgsDriver.
    Assumed to be ran after one solve_forward from a regular coupled problem, represents uncoupled
    TACS analysis from aerodynamic loads.
    """

    def __init__(self, tacs_interface: TacsSteadyInterface, model):
        self.tacs_interface = tacs_interface
        self.model = model

        # reset struct mesh positions
        for body in self.model.bodies:
            body.update_transfer()

        # zero out previous run data from funtofem
        # self._zero_tacs_data()
        # self._zero_adjoint_data()

    def solve_forward(self):
        """
        solve the forward analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        """

        fail = 0

        # zero all data to start fresh problem, u = 0, res = 0
        self._zero_tacs_data()

        for scenario in self.model.scenarios:

            # set functions and variables
            self.tacs_interface.set_variables(scenario, self.model.bodies)
            self.tacs_interface.set_functions(scenario, self.model.bodies)

            # run the forward analysis via iterate
            self.tacs_interface.initialize(scenario, self.model.bodies)
            self.tacs_interface.iterate(scenario, self.model.bodies, step=0)
            self.tacs_interface.post(scenario, self.model.bodies)

            # get functions to store the function values into the model
            self.tacs_interface.get_functions(scenario, self.model.bodies)

        return 0

    def solve_adjoint(self):
        """
        solve the adjoint analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        Similar to funtofem_driver
        """

        functions = self.model.get_functions()

        # Zero the derivative values stored in the function
        for func in functions:
            func.zero_derivatives()

        # zero adjoint data
        self._zero_adjoint_data()

        for scenario in self.model.scenarios:
            # set functions and variables
            self.tacs_interface.set_variables(scenario, self.model.bodies)
            self.tacs_interface.set_functions(scenario, self.model.bodies)

            # zero all coupled adjoint variables in the body
            for body in self.model.bodies:
                body.initialize_adjoint_variables(scenario)

            # initialize, run, and do post adjoint
            self.tacs_interface.initialize_adjoint(scenario, self.model.bodies)
            self.tacs_interface.iterate_adjoint(scenario, self.model.bodies, step=0)
            self.tacs_interface.post_adjoint(scenario, self.model.bodies)

            # call get function gradients to store  the gradients from tacs
            self.tacs_interface.get_function_gradients(scenario, self.model.bodies)

    def _zero_tacs_data(self):
        """
        zero any TACS solution / adjoint data before running pure TACS
        """

        if self.tacs_interface.tacs_proc:

            # zero temporary solution data
            # others are zeroed out in the tacs_interface by default
            self.tacs_interface.res.zeroEntries()
            self.tacs_interface.ext_force.zeroEntries()
            self.tacs_interface.update.zeroEntries()

            # zero any scenario data
            for scenario in self.model.scenarios:

                # zero state data
                u = self.tacs_interface.scenario_data[scenario].u
                u.zeroEntries()
                self.tacs_interface.assembler.setVariables(u)

    def _zero_adjoint_data(self):

        if self.tacs_interface.tacs_proc:
            # zero adjoint variable
            for scenario in self.model.scenarios:
                psi = self.tacs_interface.scenario_data[scenario].psi
                for vec in psi:
                    vec.zeroEntries()


class TacsSteadyShapeDriver:
    def __init__(
        self,
        comm,
        model,
        n_tacs_procs,
        tacs_aim,
        flow_solver,
        transfer_options,
        initial_bdf=None,
    ):
        self.comm = comm
        self.tacs_aim = tacs_aim
        self.model = model
        self.n_tacs_procs = n_tacs_procs
        self.flow_solver = flow_solver
        self.transfer_options = transfer_options
        self.initial_bdf = initial_bdf

        # temp interface, driver attributes
        self.tacs_driver = None
        self.tacs_interface = None

        # store the shape variables list
        self.shape_variables = [
            var for var in self.model.get_variables() if var.analysis_type == "shape"
        ]

        # prime the driver upon construction
        self._prime_driver()

    @property
    def tacs_comm(self):
        world_rank = self.comm.rank
        if world_rank < self.n_tacs_procs:
            color = 1
            key = world_rank
        else:
            color = MPI.UNDEFINED
            key = world_rank
        return self.comm.Split(color, key)

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == 0

    @property
    def analysis_dir(self):
        direc = None
        if self.root_proc:
            direc = self.tacs_aim.analysisDir
        direc = self.comm.bcast(direc, root=0)
        return direc

    def _prime_driver(self):
        """
        prime the driver by computing the fixed aero loads
        from one forward analysis of the funtofem driver
        """
        # generate the geometry if necessary
        initially_none = self.initial_bdf is None
        if initially_none:

            self.initial_bdf = os.path.join(self.analysis_dir, "nastran_CAPS.dat")

            # run the tacs aim preAnalysis to generate a bdf file
            if self.root_proc:
                self.tacs_aim.preAnalysis()

        # build a tacs interface
        tacs_interface = createTacsInterfaceFromBDF(
            model=self.model,
            comm=self.comm,
            nprocs=self.n_tacs_procs,
            bdf_file=self.initial_bdf,
            callback=None,
            prefix=self.analysis_dir,
            struct_options={},
        )

        # create the solvers dictionary
        solvers = {"flow": self.flow_solver, "structural": tacs_interface}

        # build the funtofem driver
        funtofem_driver = FUNtoFEMnlbgs(
            solvers=solvers,
            comm=self.comm,
            struct_comm=self.tacs_comm,
            struct_root=0,
            aero_comm=self.comm,
            aero_root=0,
            transfer_options=self.transfer_options,
            model=self.model,
        )

        # run the forward analysis in order to obtain the aero loads
        funtofem_driver.solve_forward()

        # run post analysis of tacs aim to prevent CAPS_DIRTY error
        if initially_none and self.root_proc:

            # write the model sensitivity file with zero derivatives
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=os.path.join(self.analysis_dir, "nastran_CAPS.sens"),
                discipline="structural",
            )

    def _transfer_fixed_aero_loads(self):
        """
        transfer fixed aero loads over to the new
        """
        # loop over each body to copy and transfer loads for the new structure
        for body in self.model.bodies:

            # update the transfer schemes for the new mesh size
            body.update_tranfer()

            ns = self.body.struct_nnodes
            dtype = self.body.dtype

            # zero the initial struct loads and struct flux for each scenario
            for scenario in self.model.scenarios:

                # initialize new struct loads
                if self.body.transfer is not None:
                    self.body.struct_loads[scenario.id] = np.zeros(3 * ns, dtype=dtype)

                # initialize new struct heat flux
                if self.body.thermal_transfer is not None:
                    self.body.struct_heat_flux[scenario.id] = np.zeros(ns, dtype=dtype)

                # transfer the loads and heat flux from fixed aero loads to
                # the mesh for the new structural shape
                self.body.transfer_loads(scenario)
                self.body.transfer_heat_flux(scenario)

        return

    def solve_forward(self):
        """
        forward analysis for the given shape and functionals
        assumes shape variables have already been changed
        """

        # set the new shape variables into the model
        if self.root_proc:
            for var in self.shape_variables:
                self.tacs_aim.geometry.despmtr[var.name].value = var.value

            # build the new structure geometry
            self.tacs_aim.preAnalysis()

        # make the new tacs interface of the structural geometry
        self.tacs_interface = createTacsInterfaceFromBDF(
            model=self.model,
            comm=self.comm,
            nprocs=self.n_tacs_procs,
            bdf_file=os.path.join(self.analysis_dir, "nastran_CAPS.dat"),
            callback=None,
            prefix=self.analysis_dir,
            struct_options={},
        )

        # make a tacs driver
        self.tacs_driver = TacsSteadyAnalysisDriver(
            tacs_interface=self.tacs_interface,
            model=self.model,
        )

        # transfer the fixed aero loads
        self._transfer_fixed_aero_loads()

        self.tacs_driver.solve_forward()

    def solve_adjoint(self):
        """
        solve the adjoint analysis for the given shape
        assumes the forward analysis for this shape has already been performed
        """

        # run the adjoint structural analysis
        self.tacs_driver.solve_adjoint()

        # collect the coordinate derivatives for each body
        for body in self.model.bodies:
            body.collect_coordinate_derivatives(comm=self.comm, discipline="structural")

        # write the sensitivity file for the tacs AIM
        self.model.write_sensitivity_file(
            comm=self.comm,
            filename=os.path.join(self.analysis_dir, "nastran_CAPS.sens"),
            discipline="structural",
        )

        # run the tacs aim postAnalysis to compute the chain rule product
        self.tacs_aim.postAnalysis()

        # store the shape variables in the function gradients
        for scenario in self.model.scenarios:
            self.get_function_gradients(scenario)

    def get_function_gradients(self, scenario):
        """
        get shape derivatives together from tacs aim
        """
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(self.shape_variables):
                derivative = self.tacs_aim.dynout[func.name].deriv(var.name)
                func.set_gradient_component(var, derivative)
