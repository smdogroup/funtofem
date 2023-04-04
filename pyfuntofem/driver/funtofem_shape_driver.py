__all__ = ["FuntofemShapeDriver"]

from .funtofem_nlbgs_driver import FUNtoFEMnlbgs

import importlib.util

fun3d_loader = importlib.util.find_spec("fun3d")
tacs_loader = importlib.util.find_spec("tacs")
if fun3d_loader is not None:  # check whether we can import FUN3D
    from pyfuntofem.interface import Fun3dInterface
if tacs_loader is not None:
    from pyfuntofem.interface import (
        TacsSteadyInterface,
        TacsUnsteadyInterface,
        TacsInterface,
    )


class FuntofemShapeDriver(FUNtoFEMnlbgs):
    def __init__(
        self,
        solvers,
        comm_manager=None,
        transfer_settings=None,
        model=None,
    ):
        """
        The FUNtoFEM driver for the Nonlinear Block Gauss-Seidel
        solvers for steady and unsteady coupled adjoint, augmented for ESP/CAPS shape
        optimization with FUN3D + TACS.

        Parameters
        ----------
        solvers: SolverManager
           the various disciplinary solvers
        comm_manager: CommManager
            manager for various discipline communicators
        transfer_settings: TransferSettings
            options of the load and displacement transfer scheme
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        """

        # construct super class
        super(FuntofemShapeDriver, self).__init__(
            solvers, comm_manager, transfer_settings, model
        )

        # make sure the solver interfaces are TACS and FUN3D
        assert isinstance(self.solvers.flow, Fun3dInterface)
        assert isinstance(self.solvers.structural, TacsSteadyInterface) or isinstance(
            self.solvers.structural, TacsUnsteadyInterface
        )

        # get shape variables
        self.shape_variables = [
            var for var in self.model.get_variables() if var.analysis_type == "shape"
        ]

        # get the fun3d aim for changing shape
        if model.flow is None:
            fun3d_aim = None
        else:
            fun3d_aim = model.flow.fun3d_aim
        if model.structural is None:
            tacs_aim = None
        else:
            tacs_aim = model.structural.tacs_aim

        # save both of the discipline aims
        self.fun3d_aim = fun3d_aim
        self.tacs_aim = tacs_aim

        # save both discipline models
        self.fun3d_model = self.model.flow
        self.tacs_model = self.model.structural

        # make sure the fun3d model is setup if needed
        if self.change_shape and self.aero_shape:
            assert self.fun3d_model.is_setup
            self._setup_grid_filepaths()

        return

    @property
    def change_shape(self) -> bool:
        """only do shape optimization if shape variables exist"""
        return len(self.shape_variables) > 0

    @property
    def aero_shape(self) -> bool:
        """whether aerodynamic shape is changing"""
        return self.fun3d_aim is not None and self.change_shape

    @property
    def struct_shape(self) -> bool:
        """whether structural shape is changing"""
        return self.tacs_aim is not None and self.change_shape

    def solve_forward(self):
        """create new aero/struct geometries and run fully-coupled forward analysis"""
        if self.aero_shape:
            # run the pre analysis to generate a new mesh
            self.fun3d_model.apply_shape_variables(self.shape_variables)
            self.fun3d_aim.pre_analysis()

            # move grid files to each scenario location
            # no need to remake Fun3dInterface, it will read in new grid next analysis
            self.fun3d_aim._move_grid_files()

        if self.struct_shape:
            # set the new shape variables into the model using update design to prevent CAPS_CLEAN errors
            input_dict = {var.name: var.value for var in self.model.get_variables()}
            self.tacs_model.update_design(input_dict)
            self.tacs_aim.setup_aim()

            # build the new structure geometry
            self.tacs_aim.pre_analysis()

            # make the new tacs interface of the structural geometry
            self.tacs_interface = TacsInterface.create_from_bdf(
                model=self.model,
                comm=self.comm,
                nprocs=self.nprocs,
                bdf_file=self.dat_file_path,
                output_dir=self.analysis_dir,
            )

            # update the structural solver in FUNtoFEMnlbgs
            self.solvers.structural = self.tacs_interface

        # call solve forward of super class for no shape, fully-coupled analysis
        super(FuntofemShapeDriver, self).solve_forward()

        return

    def solve_adjoint(self):
        """run the fully-coupled adjoint analysis and extract shape derivatives as well"""

        super(FuntofemShapeDriver, self).solve_adjoint()

        if self.struct_shape:
            # write the sensitivity file for the tacs AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=self.tacs_aim.sens_file_path,
                discipline="structural",
            )

            # run the tacs aim postAnalysis to compute the chain rule product
            self.tacs_aim.post_analysis()

            for scenario in self.model.scenarios:
                self._get_struct_shape_derivatives(scenario)

        if self.aero_shape:
            # write the sensitivity file for the FUN3D AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=self.fun3d_aim.sens_file_path,
                discipline="aerodynamic",
            )

            # run the tacs aim postAnalysis to compute the chain rule product
            self.fun3d_aim.post_analysis()

            for scenario in self.model.scenarios:
                self._get_aero_shape_derivatives(scenario)

        return

    def _setup_grid_filepaths(self):
        """setup the filepaths for each fun3d grid file in scenarios"""
        fun3d_dir = self.fun3d_interface.fun3d_dir
        grid_filepaths = []
        for scenario in self.model.scenarios:
            filepath = os.path.join(
                fun3d_dir, scenario.name, "Flow", "funtofem_CAPS.lb8.ugrid"
            )
            grid_filepaths.append(filepath)
        # set the grid filepaths into the fun3d aim
        self.fun3d_aim.grid_filepaths = grid_filepaths
        return

    def _get_struct_shape_derivatives(self, scenario):
        """
        get shape derivatives together from tacs aim
        and store the data in the funtofem model
        """
        gradients = None

        # read shape gradients from tacs aim on root proc
        if self.root_proc:
            gradients = []
            direct_tacs_aim = self.tacs_aim.aim

            for ifunc, func in enumerate(scenario.functions):
                gradients.append([])
                for ivar, var in enumerate(self.shape_variables):
                    derivative = direct_tacs_aim.dynout[func.full_name].deriv(var.name)
                    gradients[ifunc].append(derivative)

        # broadcast shape gradients to all other processors
        gradients = self.comm.bcast(gradients, root=0)

        # store shape derivatives in funtofem model on all processors
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(self.shape_variables):
                derivative = gradients[ifunc][ivar]
                func.add_gradient_component(var, derivative)

        return

    def _get_aero_shape_derivatives(self, scenario):
        """
        get shape derivatives together from FUN3D aim
        and store the data in the funtofem model
        """
        gradients = None

        # read shape gradients from tacs aim on root proc
        fun3d_aim_root = self.fun3d_aim.root
        if self.fun3d_aim.root_proc:
            gradients = []
            direct_fun3d_aim = self.fun3d_aim.aim

            for ifunc, func in enumerate(scenario.functions):
                gradients.append([])
                for ivar, var in enumerate(self.shape_variables):
                    derivative = direct_fun3d_aim.dynout[func.full_name].deriv(var.name)
                    gradients[ifunc].append(derivative)

        # broadcast shape gradients to all other processors
        gradients = self.comm.bcast(gradients, root=fun3d_aim_root)

        # store shape derivatives in funtofem model on all processors
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(self.shape_variables):
                derivative = gradients[ifunc][ivar]
                func.add_gradient_component(var, derivative)

        return
