__all__ = ["FuntofemShapeDriver"]

"""
Unfortunately, FUN3D has to be completely re-initialized for new aerodynamic meshes, so we have
to split our OnewayAeroDriver scripts in implementation into two files, a my_funtofem_driver.py and a my_funtofem_analyzer.py.
The file my_funtofem_driver.py is called from a run.pbs script and manages the optimization and AIMs; this file
also uses system calls to the file my_fun3d_analyzer.py which runs the FUN3D analysis for each mesh. There are two 
class methods FuntofemShapeDriver.remesh and FuntofemShapeDriver.analysis which build the drivers for each of the two files.

NOTE :  If you need aerodynamic shape derivatives through ESP/CAPS, you should build the driver class using the class method 
FuntofemShapeDriver.remesh() and setup a separate script with a driver running the analysis. If you need structural shape derivatives
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
and (test_funtofem_shape.py => run_funtofem_shape.py) pair of files for the shape DVs using the Remote and system calls.
"""

from .funtofem_nlbgs_driver import FUNtoFEMnlbgs
import importlib.util, os, shutil, numpy as np
from funtofem.optimization.optimization_manager import OptimizationManager
from funtofem.interface import Remote
import time

caps_loader = importlib.util.find_spec("pyCAPS")
fun3d_loader = importlib.util.find_spec("fun3d")
tacs_loader = importlib.util.find_spec("tacs")
if fun3d_loader is not None:  # check whether we can import FUN3D
    from funtofem.interface import Fun3dInterface, Fun3dModel
if tacs_loader is not None:
    from funtofem.interface import (
        TacsSteadyInterface,
        TacsUnsteadyInterface,
        TacsInterface,
    )

    if caps_loader is not None:
        from tacs import caps2tacs


class FuntofemShapeDriver(FUNtoFEMnlbgs):
    @classmethod
    def aero_morph(
        cls,
        solvers,
        model,
        transfer_settings=None,
        comm_manager=None,
        struct_nprocs=48,
    ):
        """
        Build a FuntofemShapeDriver object with FUN3D mesh morphing or with no fun3dAIM
        """
        return cls(
            solvers,
            model=model,
            transfer_settings=transfer_settings,
            comm_manager=comm_manager,
            is_paired=False,
            struct_nprocs=struct_nprocs,
        )

    @classmethod
    def aero_remesh(cls, solvers, model, remote):
        """
        Build a FuntofemShapeDriver object for the my_funtofem_driver.py script:
            this object would be responsible for the fun3d, aflr AIMs and

        """
        return cls(solvers, model=model, remote=remote, is_paired=True)

    @classmethod
    def analysis(
        cls,
        solvers,
        model,
        transfer_settings=None,
        comm_manager=None,
    ):
        """
        Build a FuntofemShapeDriver object for the my_funtofem_analysis.py script:
            this object would be responsible for running the FUN3D
            analysis and writing an aero.sens file to the fun3d directory
        """
        return cls(
            solvers,
            model=model,
            transfer_settings=transfer_settings,
            comm_manager=comm_manager,
            is_paired=True,
        )

    def __init__(
        self,
        solvers,
        comm_manager=None,
        transfer_settings=None,
        model=None,
        remote=None,
        is_paired=False,
        struct_nprocs=48,
    ):
        """
        The FUNtoFEM driver for the Nonlinear Block Gauss-Seidel
        solvers for steady and unsteady coupled adjoint, augmented for ESP/CAPS shape
        optimization with FUN3D + TACS.

        NOTE : for using FuntofemShapeDriver driver, put the moving_body.input file on "deform" or "rigid+deform" mesh movement the majority of the time.

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

        self.transfer_settings = transfer_settings
        self.remote = remote
        self.is_paired = is_paired
        self.struct_nprocs = struct_nprocs
        self.shape_variables = [
            var for var in self.model.get_variables() if var.analysis_type == "shape"
        ]

        # make sure the solver interfaces are TACS and FUN3D
        if not (self.change_shape) and self.is_remote:
            raise AssertionError(
                "Need shape variables for using the remote system call features for FUN3D."
            )

        # check for unsteady problems
        self._unsteady = any([not scenario.steady for scenario in model.scenarios])

        # check which aero solver we were given
        self.flow_aim = None
        self._flow_solver_type = None
        if model.flow is None:
            if fun3d_loader is not None:
                if isinstance(solvers.flow, Fun3dInterface):
                    self._flow_solver_type = "fun3d"
            # TBD on new types
        else:  # check with shape change
            if fun3d_loader is not None:
                if isinstance(model.flow, Fun3dModel):
                    self._flow_solver_type = "fun3d"
                    self.flow_aim = model.flow.fun3d_aim
            # TBD on new types

        # figure out which discipline solver we are using
        self.struct_interface = solvers.structural
        self.struct_aim = None
        self._struct_solver_type = None
        if model.structural is None:
            # TACS solver
            if tacs_loader is not None:
                if isinstance(solvers.structural, TacsSteadyInterface) or isinstance(
                    solvers.structural, TacsUnsteadyInterface
                ):
                    self._struct_solver_type = "tacs"
            # TBD more solvers
        # check for structural AIMs
        if caps_loader is not None and model.structural is not None:
            # TACS solver
            if tacs_loader is not None:
                if isinstance(model.structural, caps2tacs.TacsModel):
                    self._struct_solver_type = "tacs"
                    self.struct_aim = model.structural.tacs_aim
            # TBD more solvers

        if not self.is_remote:
            assert solvers.flow is not None
            assert solvers.structural is not None or model.structural is not None
        self._first_forward = True

        # initialize adjoint state variables to zero for writing sens files
        if self.is_paired:
            for scenario in self.model.scenarios:
                for body in self.model.bodies:
                    body.initialize_adjoint_variables(
                        scenario
                    )  # for writing sens files even in forward case

        # make sure the fun3d model is setup if needed
        if self.change_shape and self.aero_shape:
            assert self.flow_aim is not None
            if self.uses_fun3d:
                assert self.model.flow.is_setup
                self._setup_grid_filepaths()

        # initialize all derivative values to zero including off-scenario
        for func in self.model.get_functions(all=True):
            for var in self.model.get_variables():
                func.derivatives[var] = 0.0
        return

    def _initialize_funtofem(self):
        # initialize variables with newly defined aero size
        comm = self.solvers.comm
        comm_manager = self.solvers.comm_manager
        for body in self.model.bodies:
            # transfer to fixed structural loads in case the user got only aero loads from the OnewayAeroDriver
            body.initialize_transfer(
                comm=comm,
                struct_comm=comm_manager.struct_comm,
                struct_root=comm_manager.struct_root,
                aero_comm=comm_manager.aero_comm,
                aero_root=comm_manager.aero_root,
                transfer_settings=self.transfer_settings,  # using minimal settings since we don't use the state variables here (almost a dummy obj)
            )
            for scenario in self.model.scenarios:
                body.initialize_variables(scenario)
                body.initialize_adjoint_variables(
                    scenario
                )  # for writing sens files even in forward case

    def _update_struct_transfer(self):
        """update struct nnodes and transfer scheme if struct remeshing in the loop"""
        # update transfer with the newly created mesh (and new #struct_nodes)
        for body in self.model.bodies:
            # update the transfer schemes for the new mesh size
            body.update_transfer()

            ns = body.struct_nnodes
            dtype = body.dtype

            # need to initialize transfer schemes again with tacs_comm
            comm = self.comm
            comm_manager = self.solvers.comm_manager
            body.initialize_transfer(
                comm=comm,
                struct_comm=comm_manager.struct_comm,
                struct_root=comm_manager.struct_root,
                aero_comm=comm_manager.aero_comm,
                aero_root=comm_manager.aero_root,
                transfer_settings=self.transfer_settings,
            )

            # zero the initial struct loads and struct flux for each scenario
            for scenario in self.model.scenarios:
                # initialize new struct shape term for new ns
                nf = scenario.count_adjoint_functions()
                body.struct_shape_term[scenario.id] = np.zeros(
                    (3 * ns, nf), dtype=dtype
                )

                # initialize new elastic struct vectors
                if body.transfer is not None:
                    body.struct_loads[scenario.id] = np.zeros(3 * ns, dtype=dtype)
                    body.struct_disps[scenario.id] = np.zeros(3 * ns, dtype=dtype)

                # initialize new struct heat flux
                if body.thermal_transfer is not None:
                    body.struct_heat_flux[scenario.id] = np.zeros(ns, dtype=dtype)
                    body.struct_temps[scenario.id] = (
                        np.ones(ns, dtype=dtype) * scenario.T_ref
                    )

    def solve_forward(self):
        """
        Create new aero/struct geometries and run fully-coupled forward analysis.
        """
        if self.aero_shape:
            start_time_aero = time.time()
            if self.comm.rank == 0:
                print("F2F - building aero mesh..", flush=True)
            if self.flow_aim.mesh_morph:
                self.flow_aim.set_design_sensitivity(False, include_file=False)

            # run the pre analysis to generate a new mesh
            self.flow_aim.pre_analysis()

            dt_aero = time.time() - start_time_aero
            if self.comm.rank == 0:
                print(f"F2F - built aero mesh in {dt_aero:.5e} sec", flush=True)

            # for FUN3D mesh morphing now initialize body nodes
            if not (self.is_paired) and self._first_forward:
                if self.uses_fun3d:
                    assert not (self.solvers.flow.auto_coords)
                    self.solvers.flow._initialize_body_nodes(
                        self.model.scenarios[0], self.model.bodies
                    )

                    # initialize funtofem transfer data with new aero_nnodes size
                    self._initialize_funtofem()
                    self._first_forward = False

        if self.struct_shape:
            # self._update_struct_design()
            start_time_struct = time.time()
            if self.comm.rank == 0:
                print("F2F - Building struct mesh")
            input_dict = {var.name: var.value for var in self.model.get_variables()}
            self.model.structural.update_design(input_dict)
            self.struct_aim.setup_aim()
            self.struct_aim.pre_analysis()

            dt_struct = time.time() - start_time_struct
            if self.comm.rank == 0:
                print(f"F2F - Built struct mesh in {dt_struct:.5e} sec", flush=True)

            # move the bdf and dat file to the fun3d_dir
            if self.is_remote:
                self._move_struct_mesh()

            if not (self.is_remote):
                # this will almost never get used until we can remesh without having
                # to system call FUN3D, please don't put shape variables in the analysis file
                # make the new tacs interface of the structural geometry
                if self.uses_tacs:
                    self.solvers.structural = TacsInterface.create_from_bdf(
                        model=self.model,
                        comm=self.comm,
                        nprocs=self.struct_nprocs,
                        bdf_file=self.struct_aim.root_dat_file,
                        output_dir=self.struct_aim.root_analysis_dir,
                    )

                # update the structural part of transfer scheme due to remeshing
                self._update_struct_transfer()

        if self.is_remote:
            if self.comm.rank == 0:
                print("F2F - writing design variables file", flush=True)
            # write the funtofem design input file
            self.model.write_design_variables_file(
                self.comm,
                filename=Remote.paths(self.comm, self.remote.main_dir).design_file,
                root=0,
            )

            # clear the output file
            if self.root_proc and os.path.exists(self.remote.output_file):
                os.remove(self.remote.output_file)

            start_time = time.time()
            if self.comm.rank == 0:
                print(f"Calling remote analysis..", flush=True)
            # system call funtofem forward + adjoint analysis
            os.system(
                f"mpiexec_mpt -n {self.remote.nprocs} python {self.remote.analysis_file} 2>&1 > {self.remote.output_file}"
            )
            elapsed_time = time.time() - start_time
            if self.comm.rank == 0:
                print(
                    f"Done with remote analysis in {elapsed_time:2.5e} sec", flush=True
                )

        else:
            if self.is_paired:
                # read in the funtofem design input file
                self.model.read_design_variables_file(
                    self.comm,
                    filename=Remote.paths(self.comm, self.flow_dir).design_file,
                    root=0,
                )

            # call solve forward of super class for no shape, fully-coupled analysis
            super(FuntofemShapeDriver, self).solve_forward()

        # write sens file for remote to read or if shape change all in one
        if not self.is_remote:
            if not self.is_paired:
                filepath = self.flow_aim.sens_file_path
            else:
                filepath = Remote.paths(self.comm, self.flow_dir).aero_sens_file

            # write the sensitivity file for the FUN3D AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=filepath,
                discipline="aerodynamic",
                write_dvs=False,
            )

        # post analysis for FUN3D mesh morphing
        if self.aero_shape:  # either remote or regular
            # src for movement of sens file or None if not moving it
            sens_file_src = self.remote.aero_sens_file if self.is_paired else None

            # run the tacs aim postAnalysis to compute the chain rule product
            self.flow_aim.post_analysis(sens_file_src)

            # get the analysis function values
            if self.flow_aim.mesh_morph:
                self.flow_aim.unlink()
            else:
                self._get_remote_functions(discipline="aerodynamic")

        # evaluate composite functions
        self.model.evaluate_composite_functions(compute_grad=False)
        return

    def solve_adjoint(self):
        """
        Run the fully-coupled adjoint analysis and extract shape derivatives.
        """
        self._zero_derivatives()

        if self.aero_shape:
            if self.flow_aim.mesh_morph:
                self.flow_aim.set_design_sensitivity(True, include_file=False)

            # run the pre analysis to generate a new mesh
            self.flow_aim.pre_analysis()

        if not self.is_remote:
            # call funtofem adjoint analysis for non-remote driver
            super(FuntofemShapeDriver, self).solve_adjoint()

            # write analysis functions file in analysis or system call
            if self.is_paired:
                self.model.write_functions_file(
                    self.comm, Remote.paths(self.comm, self.flow_dir)._functions_file
                )

            if self.is_paired:
                write_struct = True
                write_aero = True
                struct_sensfile = Remote.paths(
                    self.comm, self.flow_dir
                ).struct_sens_file
                aero_sensfile = Remote.paths(self.comm, self.flow_dir).aero_sens_file
            else:
                if self.struct_shape:
                    write_struct = True
                    struct_sensfile = self.struct_aim.root_sens_file
                else:
                    write_struct = False

                if self.aero_shape:
                    write_aero = True
                    aero_sensfile = self.flow_aim.sens_file_path
                else:
                    write_aero = False

            if write_struct:
                # write the sensitivity file for the tacs AIM
                self.model.write_sensitivity_file(
                    comm=self.comm,
                    filename=struct_sensfile,
                    discipline="structural",
                )

            if write_aero:
                # write sensitivity file for the FUN3D AIM
                self.model.write_sensitivity_file(
                    comm=self.comm,
                    filename=aero_sensfile,
                    discipline="aerodynamic",
                    write_dvs=False,
                )

        if self.struct_shape:  # either remote or regular
            # copy sens file to potetially parallel tacs AIMs
            if self.struct_aim.root_proc:
                if self.is_paired:
                    # move struct sens file to tacs aim directory
                    src = self.remote.struct_sens_file
                else:
                    src = self.struct_aim.root_sens_file

                for proc in self.struct_aim.active_procs[1:]:
                    dest = self.struct_aim.sens_file_path(proc)
                    shutil.copy(src, dest)

            # run the tacs aim postAnalysis to compute the chain rule product
            self.struct_aim.post_analysis()

            self._get_remote_functions(discipline="structural")

            for scenario in self.model.scenarios:
                self._get_struct_shape_derivatives(scenario)

        if self.aero_shape:  # either remote or regular
            # src for movement of sens file if it was the paired driver case
            sens_file_src = self.remote.aero_sens_file if self.is_paired else None

            # run the tacs aim postAnalysis to compute the chain rule product
            self.flow_aim.post_analysis(sens_file_src)

            # self._get_remote_functions(discipline="aerodynamic")

            for scenario in self.model.scenarios:
                self._get_aero_shape_derivatives(scenario)

        # get any remaining aero, struct derivatives from the funtofem.out file (only for analysis functions)
        if self.is_remote and self.is_paired:
            self.model.read_functions_file(self.comm, self.remote._functions_file)

        # evaluate the composite functions
        self.model.evaluate_composite_functions(compute_grad=True)

        # write a functions file
        if self.is_remote and self.is_paired:
            print("Writing funtofem.out file", flush=True)
            self.model.write_functions_file(
                self.comm, self.remote.functions_file, full_precision=False, optim=True
            )

        return

    def _setup_grid_filepaths(self):
        """setup the filepaths for each fun3d grid file in scenarios"""
        if self.solvers.flow is not None:
            fun3d_dir = self.flow_dir
        else:
            fun3d_dir = self.remote.main_dir
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
        self.flow_aim.grid_filepaths = grid_filepaths

        # also setup the mapbc files
        mapbc_filepaths = []
        for scenario in self.model.scenarios:
            filepath = os.path.join(
                fun3d_dir,
                scenario.name,
                "Flow",
                f"{scenario.fun3d_project_name}.mapbc",
            )
            mapbc_filepaths.append(filepath)
        # set the mapbc filepaths into the fun3d aim
        self.flow_aim.mapbc_filepaths = mapbc_filepaths
        return

    def _move_struct_mesh(self):
        if self.struct_aim.root_proc:
            if self.uses_tacs:
                bdf_src = os.path.join(
                    self.struct_aim.root_analysis_dir,
                    f"{self.struct_aim.project_name}.bdf",
                )
                bdf_dest = self.remote.bdf_file
                shutil.copy(bdf_src, bdf_dest)
                dat_src = os.path.join(
                    self.struct_aim.root_analysis_dir,
                    f"{self.struct_aim.project_name}.dat",
                )
                dat_dest = self.remote.dat_file
                shutil.copy(dat_src, dat_dest)

    @property
    def flow_dir(self):
        if self.uses_fun3d:
            return self.solvers.flow.fun3d_dir
        # TBD on other solvers

    def _update_struct_design(self):
        # currently not using this method
        if self.comm.rank == 0:
            aim = self.struct_aim.aim
            input_dict = {var.name: var.value for var in self.model.get_variables()}
            for key in input_dict:
                if aim.geometry.despmtr[key].value != input_dict[key]:
                    aim.geometry.despmtr[key].value = input_dict[key]
        return

    def _get_remote_functions(self, discipline="aerodynamic"):
        """
        Read function values from fun3dAIM when operating in the remote version of the driver.
        Note: it does not matter which AIM we read the function values from since it's the same.
        """
        functions = self.model.get_functions()
        remote_functions = None

        if self.flow_aim.root_proc:
            remote_functions = [
                self.flow_aim.aim.dynout[func.full_name].value for func in functions
            ]

        if self.struct_aim.root_proc:
            remote_functions = [
                self.struct_aim.aim.dynout[func.full_name].value for func in functions
            ]

        # broadcast the function values to other processors
        if discipline == "aerodynamic":
            root = self.flow_aim.root
        else:
            root = self.struct_aim.root_proc_ind
        remote_functions = self.comm.bcast(remote_functions, root=root)

        # update model function values in the remote version of the driver
        for ifunc, func in enumerate(functions):
            func.value = remote_functions[ifunc]
        return

    def _zero_derivatives(self):
        """zero all model derivatives"""
        for func in self.model.get_functions(all=True):
            for var in self.model.get_variables():
                func.derivatives[var] = 0.0
        return

    def _get_struct_shape_derivatives(self, scenario):
        """
        Gather shape derivatives together from TACS AIM and store the data in the FUNtoFEM model.
        """
        variables = self.model.get_variables()

        # read shape gradients from tacs aim among different processors
        # including sometimes parallel versions of the struct AIM
        gradients = []

        for ifunc, func in enumerate(scenario.functions):
            gradients.append([])
            for ivar, var in enumerate(variables):
                derivative = None
                if var.analysis_type == "structural":
                    if self.struct_aim.root_proc:
                        derivative = self.struct_aim.aim.dynout[func.full_name].deriv(
                            var.full_name
                        )
                elif var.analysis_type == "shape":
                    # if tacs aim do this, make this more modular later
                    if self.uses_tacs:  # for parallel tacsAIMs
                        c_proc = self.struct_aim.get_proc_with_shape_var(var.name)
                        if self.comm.rank == c_proc:
                            derivative = self.struct_aim.aim.dynout[
                                func.full_name
                            ].deriv(var.name)
                        # then broadcast the derivative to other processors
                        derivative = self.comm.bcast(derivative, root=c_proc)
                    else:
                        if self.root_proc:
                            derivative = self.struct_aim.aim.dynout[
                                func.full_name
                            ].deriv(var.name)
                else:
                    derivative = 0.0

                # updat the derivatives list
                gradients[ifunc].append(derivative)

        # mpi comm barrier
        self.comm.Barrier()

        # store shape derivatives in funtofem model on all processors
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(variables):
                derivative = gradients[ifunc][ivar]
                func.add_gradient_component(var, gradients[ifunc][ivar])

        return

    def _get_aero_shape_derivatives(self, scenario):
        """
        Gather shape derivatives together from FUN3D AIM and store the data in the FUNtoFEM model.
        """
        gradients = None
        variables = self.model.get_variables()

        # read shape gradients from tacs aim on root proc
        flow_aim_root = self.flow_aim.root
        if self.flow_aim.root_proc:
            gradients = []
            direct_flow_aim = self.flow_aim.aim

            for ifunc, func in enumerate(scenario.functions):
                gradients.append([])
                for ivar, var in enumerate(variables):
                    var_name = (
                        var.name if var.analysis_type == "shape" else var.full_name
                    )
                    # get aerodynamic derivatives from the funtofem.out files instead of Fun3dAim since
                    # it is kind of buggy to create Aerodynamic Analysis DVs currently
                    if var.analysis_type in ["shape"]:  # ["shape", "aerodynamic"]
                        derivative = direct_flow_aim.dynout[func.full_name].deriv(
                            var_name
                        )
                    else:
                        derivative = 0.0
                    gradients[ifunc].append(derivative)

        # broadcast shape gradients to all other processors
        gradients = self.comm.bcast(gradients, root=flow_aim_root)

        # store shape derivatives in funtofem model on all processors
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(variables):
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
        return self.flow_aim is not None and self.change_shape

    @property
    def struct_shape(self) -> bool:
        """whether structural shape is changing"""
        return self.struct_aim is not None and self.change_shape

    @property
    def is_remote(self) -> bool:
        """whether we are calling FUN3D in a remote manner"""
        return self.remote is not None

    @property
    def manager(self, hot_start: bool = False):
        return OptimizationManager(self, hot_start=hot_start)

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == 0

    @property
    def uses_tacs(self) -> bool:
        return self._struct_solver_type == "tacs"

    @property
    def uses_fun3d(self) -> bool:
        return self._flow_solver_type == "fun3d"

    @property
    def tacs_model(self):
        return self.model.structural
