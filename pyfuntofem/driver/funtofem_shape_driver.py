__all__ = ["FuntofemShapeDriver"]


"""
Unfortunately, FUN3D has to be completely re-initialized for new aerodynamic meshes, so we have
to split our Fun3dOnewayDriver scripts in implementation into two files, a my_funtofem_driver.py and a my_funtofem_analyzer.py.
The file my_funtofem_driver.py is called from a run.pbs script and manages the optimization and AIMs; this file
also uses system calls to the file my_fun3d_analyzer.py which runs the FUN3D analysis for each mesh. There are two 
class methods FuntofemShapeDriver.remote and FuntofemShapeDriver.analysis which build the drivers for each of the two files.

NOTE :  If you need aerodynamic shape derivatives through ESP/CAPS, you should build the driver class using the class method 
FuntofemShapeDriver.remote() and setup a separate script with a driver running the analysis. If you need structural shape derivatives
through ESP/CAPS or no shape derivatives, you can build the driver with the analysis() class method and optimize it without remote
calls and a separate script. This is because only remeshing of the aero side requires a remote driver + analysis driver in order
to reset the FUN3D system environment. More details on these two files are provided below.

my_funtofem_driver.py : main driver script which called from the run.pbs
    NOTE : similar to tests/fun3d_tests/test_funtofem_shape.py
    - Construct the FUNtoFEMmodel
    - Build the Fun3dModel and link with Fun3dAim + AflrAIM and mesh settings, then store in funtofem_model.flow = this
    - Construct bodies and scenarios
    - Register aerodynamic and shape DVs to the scenarios/bodies
    - Construct the SolverManager with comm, but leave flow and structural attributes empty
    - Construct the funtofem shape driver with class method FuntofemShapeDriver.remote to manage system calls to the other script.
    - Build the optimization manager / run the driver

my_funtofem_analysis.py : fun3d analysis script, which is called indirectly from my_fun3d_driver.py
    NOTE : similar to tests/fun3d_tests/run_funtofem_analysis.py
    - Construct the FUNtoFEMmodel
    - Construct the bodies and scenarios
    - Register aerodynamic DVs to the scenarios/bodies (no shape variables added and no AIMs here)
    - Construct the Fun3dInterface
    - Construct the solvers (SolverManager), and set solvers.flow = my_fun3d_interface
    - Construct the a fun3d oneway driver with class method FuntofemShapeDriver.analysis
    - Run solve_forward() and solve_adjoint() on the FuntofemShapeDriver

For an example implementation see tests/fun3d_tests/ folder with test_fun3d_analysis.py for just aero DVs
and (test_funtofem_shape.py => run_funtofem_shape.py) pair of files for the shape DVs using the Fun3dRemote and system calls.
"""

from .funtofem_nlbgs_driver import FUNtoFEMnlbgs
import importlib.util, os, shutil
from pyfuntofem.optimization.optimization_manager import OptimizationManager

fun3d_loader = importlib.util.find_spec("fun3d")
tacs_loader = importlib.util.find_spec("tacs")
if fun3d_loader is not None:  # check whether we can import FUN3D
    from pyfuntofem.interface import Fun3dInterface
    from .fun3d_oneway_driver import Fun3dRemote
if tacs_loader is not None:
    from pyfuntofem.interface import (
        TacsSteadyInterface,
        TacsUnsteadyInterface,
        TacsInterface,
    )


class FuntofemShapeDriver(FUNtoFEMnlbgs):
    @classmethod
    def remote(cls, solvers, model, fun3d_remote):
        """
        build a Fun3dOnewayDriver object for the my_fun3d_driver.py script:
            this object would be responsible for the fun3d, aflr AIMs and

        """
        return cls(solvers, model, fun3d_remote=fun3d_remote)

    @classmethod
    def analysis(
        cls, solvers, model, transfer_settings=None, comm_manager=None, write_sens=True
    ):
        """
        build an Fun3dOnewayDriver object for the my_fun3d_analyzer.py script:
            this object would be responsible for running the FUN3D
            analysis and writing an aero.sens file to the fun3d directory
        """
        return cls(
            solvers,
            model,
            transfer_settings=transfer_settings,
            comm_manager=comm_manager,
            write_sens=write_sens,
        )

    def __init__(
        self,
        solvers,
        comm_manager=None,
        transfer_settings=None,
        model=None,
        fun3d_remote=None,
        write_sens=False,
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

        self.fun3d_remote = fun3d_remote
        self.write_sens = write_sens

        # get shape variables
        self.shape_variables = [
            var for var in self.model.get_variables() if var.analysis_type == "shape"
        ]

        # make sure the solver interfaces are TACS and FUN3D
        if not (self.change_shape) and self.is_remote:
            raise AssertionError(
                "Need shape variables for using the remote system call features for FUN3D."
            )

        if not self.is_remote:
            assert isinstance(self.solvers.flow, Fun3dInterface)
            assert isinstance(
                self.solvers.structural, TacsSteadyInterface
            ) or isinstance(self.solvers.structural, TacsUnsteadyInterface)
            if self.aero_shape and self.root_proc:
                print(
                    f"Warning!! You are trying to remesh the aero shape without using remote system calls of FUN3D, this will likely cause a FUN3D bug."
                )

        # check for unsteady problems
        self._unsteady = False
        for scenario in model.scenarios:
            if not scenario.steady:
                self._unsteady = True
                break

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
            assert fun3d_aim is not None
            assert self.fun3d_model.is_setup
            self._setup_grid_filepaths()

        return

    def solve_forward(self):
        """create new aero/struct geometries and run fully-coupled forward analysis"""
        if self.aero_shape:
            # run the pre analysis to generate a new mesh
            self.fun3d_aim.pre_analysis(self.shape_variables)

        if self.struct_shape:
            # set the new shape variables into the model using update design to prevent CAPS_CLEAN errors
            input_dict = {var.name: var.value for var in self.model.get_variables()}
            self.tacs_model.update_design(input_dict)
            self.tacs_aim.setup_aim()

            # build the new structure geometry
            self.tacs_aim.pre_analysis()

            # move the bdf and dat file to the fun3d_dir
            if self.is_remote:
                bdf_src = os.path.join(
                    self.tacs_aim.analysis_dir, f"{self.tacs_aim.project_name}.bdf"
                )
                bdf_dest = self.fun3d_remote.bdf_file
                shutil.copy(bdf_src, bdf_dest)
                dat_src = os.path.join(
                    self.tacs_aim.analysis_dir, f"{self.tacs_aim.project_name}.bdf"
                )
                dat_dest = self.fun3d_remote.dat_file
                shutil.copy(dat_src, dat_dest)

            if not (self.is_remote):
                # this will almost never get used until we can remesh without having
                # to system call FUN3D, please don't put shape variables in the analysis file
                # make the new tacs interface of the structural geometry
                self.solvers.structural = TacsInterface.create_from_bdf(
                    model=self.model,
                    comm=self.comm,
                    nprocs=self.solvers.structural.nprocs,
                    bdf_file=self.tacs_aim.dat_file_path,
                    output_dir=self.tacs_aim.analysis_dir,
                )

        if self.is_remote:
            # write the funtofem design input file
            self.model.write_design_variables_file(
                self.comm,
                filename=Fun3dRemote.paths(self.fun3d_remote.fun3d_dir).design_file,
                root=0,
            )

            # clear the output file
            if self.root_proc:
                os.remove(self.fun3d_remote.output_file)

            # system call funtofem forward + adjoint analysis
            os.system(
                f"mpiexec_mpt -n {self.fun3d_remote.nprocs} python {self.fun3d_remote.analysis_file} 2>&1 > {self.fun3d_remote.output_file}"
            )
        else:
            # read in the funtofem design input file
            self.model.read_design_variables_file(
                self.comm,
                filename=Fun3dRemote.paths(self.solvers.flow.fun3d_dir).design_file,
                root=0,
            )

            # call solve forward of super class for no shape, fully-coupled analysis
            super(FuntofemShapeDriver, self).solve_forward()

        return

    def solve_adjoint(self):
        """run the fully-coupled adjoint analysis and extract shape derivatives as well"""

        if not self.is_remote:
            # call funtofem adjoint analysis for non-remote driver
            super(FuntofemShapeDriver, self).solve_adjoint()

            if self.struct_shape or self.write_sens:
                # write the sensitivity file for the tacs AIM
                self.model.write_sensitivity_file(
                    comm=self.comm,
                    filename=Fun3dRemote.paths(
                        self.solvers.flow.fun3d_dir
                    ).struct_sens_file,
                    discipline="structural",
                )

            if self.aero_shape or self.write_sens:
                # write sensitivity file for the FUN3D AIM
                self.model.write_sensitivity_file(
                    comm=self.comm,
                    filename=Fun3dRemote.paths(
                        self.solvers.flow.fun3d_dir
                    ).aero_sens_file,
                    discipline="aerodynamic",
                )

        if self.struct_shape:  # either remote or regular
            # move struct sens file to tacs aim directory
            tacs_sens_src = self.fun3d_remote.struct_sens_file
            tacs_sens_dest = self.tacs_aim.sens_file_path
            shutil.copy(tacs_sens_src, tacs_sens_dest)

            # run the tacs aim postAnalysis to compute the chain rule product
            self.tacs_aim.post_analysis()

            for scenario in self.model.scenarios:
                self._get_struct_shape_derivatives(scenario)

        if self.aero_shape:  # either remote or regular
            # run the tacs aim postAnalysis to compute the chain rule product
            self.fun3d_aim.post_analysis(self.fun3d_remote.aero_sens_file)

            for scenario in self.model.scenarios:
                self._get_aero_shape_derivatives(scenario)

        return

    def _setup_grid_filepaths(self):
        """setup the filepaths for each fun3d grid file in scenarios"""
        if self.solvers.flow is not None:
            fun3d_dir = self.solvers.flow.fun3d_dir
        else:
            fun3d_dir = self.fun3d_remote.fun3d_dir
        grid_filepaths = []
        for scenario in self.model.scenarios:
            filepath = os.path.join(
                fun3d_dir,
                scenario.name,
                "Flow",
                f"{scenario.fun3d_project_name}.lb8.ugrid",
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

    @property
    def is_remote(self) -> bool:
        """whether we are calling FUN3D in a remote manner"""
        return self.fun3d_remote is not None

    @property
    def manager(self, hot_start: bool = False):
        return OptimizationManager(self, hot_start=hot_start)

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == 0

    @property
    def fun3d_model(self):
        return self.model.flow
