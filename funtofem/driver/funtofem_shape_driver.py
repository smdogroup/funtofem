__all__ = ["FuntofemShapeDriver"]

"""
Written by Sean Engelstad, Georgia Tech 2023

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
    - Construct the Fun3d14Interface
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
    from funtofem.interface import Fun3d14Interface, Fun3dModel
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
        forward_flow_post_analysis=False,
        reload_funtofem_states=False,
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
            forward_flow_post_analysis=forward_flow_post_analysis,
            reload_funtofem_states=reload_funtofem_states,
        )

    @classmethod
    def aero_remesh(
        cls,
        solvers,
        model,
        remote,
        forward_flow_post_analysis=False,
        reload_funtofem_states=False,
    ):
        """
        Build a FuntofemShapeDriver object for the my_funtofem_driver.py script:
            this object would be responsible for the fun3d, aflr AIMs and

        """
        return cls(
            solvers,
            model=model,
            remote=remote,
            is_paired=True,
            forward_flow_post_analysis=forward_flow_post_analysis,
            reload_funtofem_states=reload_funtofem_states,
        )

    @classmethod
    def analysis(
        cls,
        solvers,
        model,
        transfer_settings=None,
        comm_manager=None,
        struct_nprocs=1,
        auto_run: bool = True,
        forward_flow_post_analysis=False,
        reload_funtofem_states=False,
    ):
        """
        Build a FuntofemShapeDriver object for the my_funtofem_analysis.py script:
            this object would be responsible for running the FUN3D
            analysis and writing an aero.sens file to the fun3d directory
        """
        analysis_driver = cls(
            solvers,
            model=model,
            transfer_settings=transfer_settings,
            comm_manager=comm_manager,
            struct_nprocs=struct_nprocs,
            is_paired=True,
            forward_flow_post_analysis=forward_flow_post_analysis,
            reload_funtofem_states=reload_funtofem_states,
        )

        if auto_run:
            analysis_driver.solve_forward()
            analysis_driver.solve_adjoint()
        return analysis_driver

    def __init__(
        self,
        solvers,
        comm_manager=None,
        transfer_settings=None,
        model=None,
        remote=None,
        is_paired=False,
        struct_nprocs=48,
        forward_flow_post_analysis=False,
        reload_funtofem_states=False,
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
        forward_flow_post_analysis: bool
            whether to only do preAnalysis at start of forward and postAnalysis at end of adjoint
            for long optimization iterations (then this would be False). If we want to get analysis function
            values from the forward analysis remote driver this would be True as we want to do both adjoint and flow postAnalysis.
        reload_funtofem_states: bool
            reload funtofem states - struct disps, struct temps to save coupled analysis time
        """

        # construct super class
        super(FuntofemShapeDriver, self).__init__(
            solvers,
            comm_manager,
            transfer_settings,
            model,
            reload_funtofem_states=reload_funtofem_states,
        )

        self.remote = remote
        self.is_paired = is_paired
        self.struct_nprocs = struct_nprocs
        self.shape_variables = [
            var for var in self.model.get_variables() if var.analysis_type == "shape"
        ]
        self.forward_flow_post_analysis = forward_flow_post_analysis

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
                if isinstance(solvers.flow, Fun3d14Interface):
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

        # initial timing data message
        self._iteration = 0
        if self.is_paired and not (self.is_remote):
            pass
        else:  # only write the initial message from the remote driver / non-paired driver
            self._write_timing_data(
                msg="Funtofem Shape Driver timing data..\n", overwrite=True
            )
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

    def _write_timing_data(
        self, msg, overwrite=False, root: int = 0, barrier: bool = False
    ):
        """write to the funtofem timing file"""
        remote = (
            Remote.paths(self.comm, self.flow_dir)
            if self.remote is None
            else self.remote
        )
        if self.comm.rank == root:
            hdl = open(remote.timing_file, "w" if overwrite else "a")
            hdl.write(msg + "\n")
            hdl.flush()
            hdl.close()

        # MPI Barrier for other processors
        if barrier:
            self.comm.Barrier()
        return

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
        # write the initial timing data output
        if self.remote_meshing or self.is_remote:
            self._write_timing_data(msg=f"Iteration {self._iteration}:")
        self._iteration += 1
        self._iteration_start = time.time()

        # build meshes for each discipline (potentially in parallel across each ESP/CAPS AIM instance)
        start_mesh_time = time.time()

        if self.is_remote and not (self.change_shape):
            # case where remote does not perform meshing (write the new design).
            # technically we don't need the separate call here (just needs to be before the system call),
            # but this is more logical and easier to follow
            self.model.write_design_variables_file(
                self.comm,
                filename=Remote.paths(self.comm, self.remote.main_dir).design_file,
                root=0,
            )

        if not (self.is_remote) and self.is_paired:
            if self.change_shape:
                # case where analysis script does the meshing and the remote does not.
                # need to read new shape variable values before doing the meshing
                self.model.read_design_variables_file(
                    self.comm,
                    filename=Remote.paths(self.comm, self.flow_dir).design_file,
                    root=0,
                )

            # remove the _functions_file so remote will fail
            if self.comm.rank == 0:
                analysis_functions_file = Remote.paths(
                    self.comm, self.flow_dir
                )._functions_file
                if os.path.exists(analysis_functions_file):
                    os.remove(analysis_functions_file)
                # also remove capsLock in case meshing is done in the analysis script
                if self.change_shape:
                    os.system("rm -f **/**/capsLock")

        self.comm.Barrier()

        # build aero mesh first
        if self.aero_shape:
            if self.comm.rank == self.flow_aim.root:
                print("F2F - building aero mesh..", flush=True)
            if self.flow_aim.mesh_morph:
                self.flow_aim.set_design_sensitivity(False, include_file=False)

            # run the pre analysis to generate a new mesh
            try:
                self.model.flow.pre_analysis()
                # self.flow_aim.pre_analysis()
                local_fail = False
            except:
                local_fail = True
            if local_fail:
                raise RuntimeError("F2F shape driver aero preAnalysis failed..")

        self.comm.Barrier()

        # then build struct meshes
        if self.struct_shape:
            # self._update_struct_design()
            if self.comm.rank == 0:
                print("F2F - Building struct mesh")
            input_dict = {var.name: var.value for var in self.model.get_variables()}
            self.model.structural.update_design(input_dict)
            self.struct_aim.setup_aim()

            try:
                self.struct_aim.pre_analysis()
                local_fail = False
            except:
                local_fail = True
            if local_fail:
                raise RuntimeError("F2F shape driver struct preAnalysis failed..")

        # done building meshes => report timing data
        self.comm.Barrier()

        if self.aero_shape or self.struct_shape:
            dt_mesh = (time.time() - start_mesh_time) / 60.0
            if self.struct_shape and self.aero_shape:
                msg = f"\tbuilt aero + struct meshes in {dt_mesh:.4f} min"
            elif self.struct_shape:
                msg = f"\tbuilt struct mesh in {dt_mesh:.4f} min"
            else:  # aero shape
                msg = f"\tbuilt aero mesh in {dt_mesh:.4f} min"

            self._write_timing_data(
                msg=msg,
                root=self.flow_aim.root,
                barrier=False,
            )
            if self.comm.rank == 0:
                print(msg, flush=True)

        # rebuild solver interfaces if need be
        # first for the aero / flow interfaces
        if self.aero_shape:
            # for FUN3D mesh morphing / remeshing in analysis driver =>
            #     we initialize the body nodes into F2F
            if (
                self.solvers.flow is not None
                and self.aero_shape
                and self._first_forward
            ):
                if self.uses_fun3d:
                    self.comm.Barrier()

                    assert not (self.solvers.flow.auto_coords)
                    self.solvers.flow._initialize_body_nodes(
                        self.model.scenarios[0], self.model.bodies
                    )

                    # initialize handcrafted mesh coorrdinates
                    if self.model.flow is not None and isinstance(
                        self.model.flow, Fun3dModel
                    ):
                        if self.flow_aim.is_handcrafted:
                            self.flow_aim.handcrafted_mesh_morph._get_hc_coords()

                    # initialize funtofem transfer data with new aero_nnodes size
                    self._initialize_funtofem()
                    self._first_forward = False

        # rebuild the struct solver interface if need be
        if self.struct_shape:
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

        # move on to running the analysis
        self.comm.Barrier()

        if self.is_remote:
            # write the funtofem design input file for new shape design
            # case where remote does meshing
            if self.change_shape:
                if self.comm.rank == 0:
                    print("F2F - writing design variables file", flush=True)
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

            self.comm.Barrier()

            # system call funtofem forward + adjoint analysis
            os.system(
                f"mpiexec_mpt -n {self.remote.nprocs} python {self.remote.analysis_file} 2>&1 > {self.remote.output_file}"
            )
            remote_forward_time = (time.time() - start_time) / 60.0
            self._write_timing_data(f"\tdone with system call forward analysis")
            if self.comm.rank == 0:
                print(
                    f"Done with remote analysis in {remote_forward_time:.4f} min",
                    flush=True,
                )

        else:
            if self.is_paired and not self.change_shape:
                # read in the funtofem design input file
                # case where remote does meshing and analysis does no meshing
                self.model.read_design_variables_file(
                    self.comm,
                    filename=Remote.paths(self.comm, self.flow_dir).design_file,
                    root=0,
                )

            if self.comm.rank == 0:
                print(f"funtofem starting nlbgs forward analysis..", flush=True)

            start_time = time.time()
            # call solve forward of super class for no shape, fully-coupled analysis
            super(FuntofemShapeDriver, self).solve_forward()

            forward_time = (time.time() - start_time) / 60.0
            self._write_timing_data(
                f"\tran nlbgs forward analysis in {forward_time:.4f} min"
            )

        # write sens file for remote to read or if shape change all in one
        if not self.is_remote:
            if not self.remote_meshing:
                filepath = self.flow_aim.sens_file_path
            else:
                filepath = Remote.paths(self.comm, self.flow_dir).aero_sens_file

            # write the sensitivity file for the FUN3D AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=filepath,
                discipline="aerodynamic",
                root=self.flow_aim.root,
                write_dvs=False,
            )

            self.comm.Barrier()

            # hack to do shape change with handcrafted mesh with Fun3dAim
            if self.model.flow is not None and isinstance(self.model.flow, Fun3dModel):
                if self.flow_aim.is_handcrafted:
                    hc_obj = self.flow_aim.handcrafted_mesh_morph
                    for scenario in self.model.scenarios:
                        hc_obj.compute_caps_coord_derivatives(scenario)
                    # overwrite the previous sens file
                    hc_obj.write_sensitivity_file(
                        comm=self.comm,
                        filename=filepath,
                        discipline="aerodynamic",
                        root=self.flow_aim.root,
                        write_dvs=False,
                    )

            self.comm.Barrier()

        # post analysis for FUN3D mesh morphing
        if self.aero_shape and (
            self.forward_flow_post_analysis or self.flow_aim.mesh_morph
        ):
            # src for movement of sens file or None if not moving it
            sens_file_src = self.remote.aero_sens_file if self.remote_meshing else None

            start_time = time.time()

            # run the tacs aim postAnalysis to compute the chain rule product
            self.flow_aim.post_analysis(sens_file_src)

            flow_post1_time = (time.time() - start_time) / 60.0
            self._write_timing_data(
                f"\tflow postAnalysis of forward in time {flow_post1_time:.4f} min"
            )

            # get the analysis function values
            if self.flow_aim.mesh_morph:
                self.flow_aim.unlink()
            else:
                self._get_remote_functions(discipline="aerodynamic")

        # other procs wait for flow aim postAnalysis
        self.comm.Barrier()

        # write analysis functions file in analysis or system call
        # f not(self.remote_meshing) and not(self.is_remote):
        #   # case where we are writing shape derivatives from analysis script which does meshing
        #   # to the remote driver (only writes analysis functions here)
        #   self.model.write_functions_file(
        #       self.comm, Remote.paths(self.comm, self.flow_dir)._functions_file
        #   )

        # get analysis functions from the funtofem.out file
        if self.is_remote and not (self.remote_meshing):
            self.model.read_functions_file(self.comm, self.remote._functions_file)

        self.comm.Barrier()

        # evaluate composite functions
        self.model.evaluate_composite_functions(compute_grad=False)

        return

    def solve_adjoint(self):
        """
        Run the fully-coupled adjoint analysis and extract shape derivatives.
        """
        self._zero_derivatives()

        # the additional adjoint aim preAnalysis is required by mesh morphing, but not necessarily in other cases
        if self.aero_shape and (
            self.forward_flow_post_analysis or self.flow_aim.mesh_morph
        ):
            if self.flow_aim.mesh_morph:
                self.flow_aim.set_design_sensitivity(True, include_file=False)

            start_time = time.time()

            # run the pre analysis to generate a new mesh
            self.model.flow.pre_analysis()
            # self.flow_aim.pre_analysis()

            flow_adjoint_pre_time = (time.time() - start_time) / 60.0
            self._write_timing_data(
                f"\tflow preAnalysis of adjoint in time {flow_adjoint_pre_time:.4f} min"
            )

        # other procs wait for flow pre analysis
        self.comm.Barrier()

        if not self.is_remote:
            start_time = time.time()

            # call funtofem adjoint analysis for non-remote driver
            super(FuntofemShapeDriver, self).solve_adjoint()

            remote_adjoint_time = (time.time() - start_time) / 60.0
            self._write_timing_data(
                f"\tran nlbgs adjoint analysis in {remote_adjoint_time:.4f} min"
            )

            # write analysis functions file in analysis or system call
            if self.is_paired and not self.change_shape:
                # case where we are just writing the non-shape derivatives
                self.model.write_functions_file(
                    self.comm, Remote.paths(self.comm, self.flow_dir)._functions_file
                )

            if self.remote_meshing:
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
                    root=0,
                    discipline="structural",
                )

            if write_aero:
                # write sensitivity file for the FUN3D AIM
                self.model.write_sensitivity_file(
                    comm=self.comm,
                    filename=aero_sensfile,
                    discipline="aerodynamic",
                    root=0,
                    write_dvs=False,
                )

                self.comm.Barrier()

                # hack to do shape change with handcrafted mesh with Fun3dAim
                if self.model.flow is not None and isinstance(
                    self.model.flow, Fun3dModel
                ):
                    if self.flow_aim.is_handcrafted:
                        hc_obj = self.flow_aim.handcrafted_mesh_morph
                        for scenario in self.model.scenarios:
                            hc_obj.compute_caps_coord_derivatives(scenario)
                        # overwrite the previous sens file
                        hc_obj.write_sensitivity_file(
                            comm=self.comm,
                            filename=aero_sensfile,
                            discipline="aerodynamic",
                            root=self.flow_aim.root,
                            write_dvs=False,
                        )

        # mpi barrier before start of post analysis
        self.comm.Barrier()

        if self.struct_shape:  # either remote or regular
            if self.remote_meshing:
                src = self.remote.struct_sens_file
            else:
                src = self.struct_aim.root_sens_file

            # copy sens file to potetially parallel tacs AIMs
            for proc in self.struct_aim.active_procs[1:]:
                dest = self.struct_aim.sens_file_path(proc)

                if self.struct_aim.root_proc:
                    shutil.copy(src, dest)
                # not sure if this barrier is necessary here but just in case
                self.comm.Barrier()

            # if not self.is_remote:
            #     # delete struct interface to free up memory in shape change
            #     # self.solvers.structural._deallocate()
            #     del self.solvers.structural
            #     self.comm.Barrier()

            self.comm.Barrier()
            start_time = time.time()

            # run the tacs aim postAnalysis to compute the chain rule product
            self.struct_aim.post_analysis()

            # wait for all procs to finish their post_analysis before getting results
            self.comm.Barrier()

            self._get_remote_functions(discipline="structural")

            self.comm.Barrier()

            for scenario in self.model.scenarios:
                self._get_struct_shape_derivatives(scenario)

            struct_post_time = (time.time() - start_time) / 60.0
            self._write_timing_data(
                f"\tstruct postAnalysis in {struct_post_time:.4f} min"
            )

        self.comm.Barrier()

        if self.aero_shape:  # either remote or regular
            # src for movement of sens file if it was the paired driver case
            sens_file_src = self.remote.aero_sens_file if self.remote_meshing else None

            start_time = time.time()

            # run the tacs aim postAnalysis to compute the chain rule product
            self.flow_aim.post_analysis(sens_file_src)

            # self._get_remote_functions(discipline="aerodynamic")

            for scenario in self.model.scenarios:
                self._get_aero_shape_derivatives(scenario)

            aero_post_time = (time.time() - start_time) / 60.0
            self._write_timing_data(f"\taero postAnalysis in {aero_post_time:.4f} min")

        self.comm.Barrier()

        # write analysis functions file in analysis or system call
        if self.is_paired and not (self.is_remote):
            # case where we are writing shape derivatives from analysis script which does meshing
            # to the remote driver (only writes analysis functions here)
            self.model.write_functions_file(
                self.comm, Remote.paths(self.comm, self.flow_dir)._functions_file
            )

        # get any remaining aero, struct derivatives from the funtofem.out file (only for analysis functions)
        if self.is_remote and self.is_paired:
            try:
                self.model.read_functions_file(self.comm, self.remote._functions_file)
                local_fail = False
            except:
                local_fail = True
            if local_fail:
                raise RuntimeError(
                    "Failed to read local functions file in remote driver => usually negative cell volumes occured."
                )

            # check if any derivatives have a very large magnitude
            for ifunc, func in enumerate(self.model.get_functions()):
                for ivar, var in enumerate(self.model.get_variables()):
                    deriv = func.derivatives[var]
                    if abs(deriv) > 1e10:
                        raise RuntimeError(
                            f"Funtofem - Derivative d{func.name}/d{var.name} = {deriv} > 1e10.."
                        )

        # evaluate the composite functions
        self.model.evaluate_composite_functions(compute_grad=True)

        # write a functions file for all functionals
        if self.remote is not None:
            print("Writing funtofem.out file", flush=True)
            self.model.write_functions_file(
                self.comm, self.remote.functions_file, full_precision=False, optim=True
            )

        full_iteration_time = (time.time() - self._iteration_start) / 60.0
        prefix = "remote" if self.is_remote else "analysis"
        self._write_timing_data(
            f"\t{prefix} - iteration took {full_iteration_time:.4f} min"
        )

        # mpi barrier for end of post analysis
        self.comm.Barrier()

        return

    def _setup_grid_filepaths(self):
        """setup the filepaths for each fun3d grid file in scenarios"""
        if self.solvers.flow is not None:
            fun3d_dir = self.flow_dir
        else:
            fun3d_dir = self.remote.main_dir
        grid_filepaths = []
        for scenario in self.model.scenarios:
            project_name = scenario.fun3d_project_name
            filepath = os.path.join(
                fun3d_dir,
                scenario.name,
                "Flow",
                f"{project_name}.lb8.ugrid",
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
        if self.uses_tacs:
            bdf_src = os.path.join(
                self.struct_aim.root_analysis_dir,
                f"{self.struct_aim.project_name}.bdf",
            )
            bdf_dest = self.remote.bdf_file
            if self.struct_aim.root_proc:
                shutil.copy(bdf_src, bdf_dest)
            dat_src = os.path.join(
                self.struct_aim.root_analysis_dir,
                f"{self.struct_aim.project_name}.dat",
            )
            dat_dest = self.remote.dat_file
            if self.struct_aim.root_proc:
                shutil.copy(dat_src, dat_dest)

    @property
    def flow_dir(self):
        if self.solvers is None:
            return None
        if self.uses_fun3d:
            return self.solvers.flow.fun3d_dir
        else:
            return None
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

        if discipline == "aerodynamic" and self.flow_aim.root_proc:
            remote_functions = [
                self.flow_aim.aim.dynout[func.full_name].value for func in functions
            ]

        if discipline == "structural" and self.struct_aim.root_proc:
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
                    derivative = self.comm.bcast(
                        derivative, root=self.struct_aim.root_proc_ind
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

                self.comm.Barrier()
                if derivative is None:
                    raise AssertionError(
                        f"F2F shape driver could not get d{func.name}/d{var.full_name} derivative from struct_aim"
                    )

                # updat the derivatives list
                gradients[ifunc].append(derivative)

        # mpi comm barrier
        self.comm.Barrier()

        # store shape derivatives in funtofem model on all processors
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(variables):
                derivative = gradients[ifunc][ivar]
                # only overwrite struct derivatives in remote driver case
                if var.analysis_type == "structural" and self.is_remote:
                    func.set_gradient_component(var, gradients[ifunc][ivar])
                elif var.analysis_type == "shape":
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
                if var.analysis_type == "shape":
                    func.add_gradient_component(var, gradients[ifunc][ivar])
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
        return self.remote is not None and self.is_paired

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
    def remote_meshing(self) -> bool:
        # case in which the remote is doing meshing
        if self.is_paired:
            if self.is_remote and (self.aero_shape or self.struct_shape):
                return True
            elif not (self.is_remote) and not (self.change_shape):
                return True
            else:
                return False
        return False

    @property
    def tacs_model(self):
        return self.model.structural

    def print_summary(self, print_model=False, print_comm=False):
        """
        Print out a summary of the FUNtoFEM driver for inspection.
        """

        print("\n\n==========================================================")
        print("||               FUNtoFEM Driver Summary                ||")
        print("==========================================================")
        print(self)

        self._print_shape_change()
        self._print_transfer(print_comm=print_comm)

        if print_model:
            print(
                "\nPrinting abbreviated model summary. For details print model summary directly."
            )
            self.model.print_summary(print_level=-1, ignore_rigid=True)

        return

    def _print_shape_change(self):
        _num_shape_vars = len(self.shape_variables)
        print("\n--------------------")
        print("|   Shape Change   |")
        print("--------------------")

        print(f"  No. shape variables: {_num_shape_vars}")
        print(f"  Aerodynamic shape change: {self.aero_shape}")
        print(f"  Structural shape change:  {self.struct_shape}")

        print(f"  Meshing:", end=" ")
        if self.is_paired:
            # Remeshing
            print(f" RE-MESH")
            if self.change_shape:
                print(f"    Remote is meshing.")
            else:
                print(f"    Analysis script is meshing.")
        else:
            # Morphing
            print(f" MORPH")

        return

    def __str__(self):
        line1 = f"Driver (<Type>): {self.__class__.__qualname__}"
        line2 = f"  Using remote: {self.is_remote}"
        line3 = f"  Flow solver type: {self._flow_solver_type}"
        line4 = f"  Structural solver type: {self._struct_solver_type}"
        line5 = f"    No. structural procs: {self.struct_nprocs}"

        output = (line1, line2, line3, line4, line5)

        return "\n".join(output)
