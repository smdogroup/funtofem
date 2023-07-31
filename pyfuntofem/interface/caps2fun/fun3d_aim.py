__all__ = ["Fun3dAim", "Fun3dBC"]

import os, shutil


class Fun3dAimMetaData:
    def __init__(self, project_name, analysis_dir):
        self.project_name = project_name
        self.analysis_dir = analysis_dir


class Fun3dBC:
    BC_TYPES = ["inviscid", "viscous", "Farfield"]

    def __init__(self, caps_group, bc_type, wall_spacing=None):
        self.caps_group = caps_group
        self.bc_type = bc_type
        self.wall_spacing = wall_spacing

    @property
    def name(self):
        return self.caps_group

    def register_to(self, obj):
        from .fun3d_model import Fun3dModel

        if isinstance(obj, Fun3dAim):
            obj.include(self)
        elif isinstance(obj, Fun3dModel):
            obj.fun3d_aim.include(self)
        else:
            print(f"warning this object couldn't be registerd.")
        return self

    @classmethod
    def inviscid(cls, caps_group, wall_spacing=None):
        return cls(caps_group, bc_type="inviscid", wall_spacing=wall_spacing)

    @classmethod
    def viscous(cls, caps_group, wall_spacing=None):
        return cls(caps_group, bc_type="viscous", wall_spacing=wall_spacing)

    @classmethod
    def Farfield(cls, caps_group):
        return cls(caps_group, bc_type="Farfield")

    @property
    def BC_dict(self) -> dict:
        if self.wall_spacing is None:
            return {"bcType": self.bc_type}
        else:
            return {"bcType": self.bc_type, "boundaryLayerSpacing": self.wall_spacing}


class Fun3dAim:
    def __init__(self, caps_problem, comm, mesh_morph=False, root=0):
        """Fun3dAim wrapper class for use in FUNtoFEM"""
        self.caps_problem = caps_problem
        self.comm = comm
        self.root = root
        self.mesh_morph = mesh_morph

        self.analysis_ctr = 0

        self._shape_variables = []

        self._aim = None
        self._grid_file = None
        self._grid_filepaths = []

        self._fun3d_dir = None

        self._first_grid_move = True

        self._boundary_conditions = {}

        self._build_aim()

        """setup to do ESP/CAPS sens file reading"""
        self.set_design_sensitivity(flag=True)
        if self.root_proc:
            # set to not overwrite fun3d nml analysis
            self.aim.input.Overwrite_NML = False
            # fun3d design sensitivities settings
            self.aim.input.Mesh_Morph = mesh_morph
            self.aim.input.Mesh_Morph_Combine = mesh_morph

        self._metadata = None
        if self.root_proc:
            self._metadata = Fun3dAimMetaData(
                project_name=self.aim.input.Proj_Name, analysis_dir=self.aim.analysisDir
            )
        self._metadata = self.comm.bcast(self._metadata, root=root)

    @property
    def root_proc(self):
        return self.comm.rank == self.root

    def _build_aim(self):
        self._aim = None
        self._geometry = None
        if self.root_proc:
            self._aim = self.caps_problem.analysis.create(aim="fun3dAIM", name="fun3d")
            self._geometry = self.caps_problem.geometry
        return

    def unlink(self):
        """
        Unlink the volume and surface aerodynamic meshes for mesh morphing.
        """
        if self.root_proc:
            self.aim.input["Mesh"].unlink()
        return

    def set_design_sensitivity(self, flag: bool, include_file=True):
        """toggle design sensitivity for Fun3dAim"""
        if self.root_proc:
            self.aim.input.Design_Sensitivity = flag
            if include_file:
                self.aim.input.Design_SensFile = flag
        return

    @property
    def geometry(self):
        return self._geometry

    def set_config_parameter(self, param_name: str, value: float):
        if self.root_proc:
            self.geometry.cfgpmtr[param_name].value = value
        return

    def get_config_parameter(self, param_name: str):
        value = None
        if self.root_proc:
            value = self.geometry.cfgpmtr[param_name].value
        value = self.comm.bcast(value, root=self.root)
        return value

    def set_boundary_conditions(self):
        """set the boundary conditions into FUN3D AIM"""
        if self.root_proc:
            self.aim.input.Boundary_Condition = self._boundary_conditions
        return

    @property
    def project_name(self):
        return self._metadata.project_name

    @property
    def grid_filepaths(self):
        return self._grid_filepaths

    @grid_filepaths.setter
    def grid_filepaths(self, new_filepaths):
        """set the grid filepaths from each fun3d scenario, from the fun3d interface"""
        self._grid_filepaths = new_filepaths
        return

    def _move_grid_files(self):
        """
        move each of the grid files in the preAnalysis after a new grid is
        destination files are all called fun3d_CAPS.lb8.ugrid
        """
        if self.mesh_morph:  # only move the grid once during mesh morphing case
            if not self._first_grid_move:
                return
            else:
                self._first_grid_move = False
        print(f"copying grid files = {self._first_grid_move}")
        src = self.grid_file
        for dest in self.grid_filepaths:
            shutil.copy(src, dest)
        return

    def _move_sens_files(self, src):
        """move sens files from the fun3d_dir to the fun3d AIM workdir"""
        dest = self.sens_file_path
        if self.root_proc:
            shutil.copy(src, dest)
        return

    def apply_shape_variables(self):
        """apply shape variables to the caps problem"""
        if self.root_proc:
            for shape_var in self._shape_variables:
                self.geometry.despmtr[shape_var.name].value = shape_var.value.real
        return

    def set_variables(self, shape_variables, aero_variables):
        """input list of ESP/CAPS shape variable names into fun3d aim design dict"""
        if len(shape_variables) == 0:
            return
        if self.root_proc:
            DV_dict = self.aim.input.Design_Variable
            if DV_dict is None:
                DV_dict = {}
            for dv in shape_variables:
                DV_dict[dv.name] = {}
            for dv in aero_variables:
                DV_dict[dv.name] = {}
            self.aim.input.Design_Variable = DV_dict
        self._shape_variables += shape_variables
        return

    def include(self, obj):
        if isinstance(obj, Fun3dBC):
            self._boundary_conditions[obj.name] = obj.BC_dict
        else:
            raise AssertionError(
                "No other objects can be registered to a Fun3dAim wrapper."
            )

    @property
    def aim(self):
        return self._aim

    @property
    def analysis_dir(self) -> str:
        return self._metadata.analysis_dir

    @property
    def mesh_morph_filename(self):
        return f"{self.project_name}_body1.dat"

    @property
    def mesh_morph_filepath(self):
        return os.path.join(self.analysis_dir, "Flow", self.mesh_morph_filename)

    def pre_analysis(self):
        if self.root_proc:
            self.apply_shape_variables()
            self.aim.preAnalysis()
            self._move_grid_files()
        self.comm.Barrier()
        return

    def post_analysis(self, sens_file_src=None):
        if self.root_proc:
            # move sens files if need be from fun3d dir to fun3d workdir
            if sens_file_src is not None:
                self._move_sens_files(src=sens_file_src)

            self.aim.postAnalysis()
        self.comm.Barrier()
        return

    @property
    def grid_file(self):
        return self._grid_file

    @grid_file.setter
    def grid_file(self, new_grid_file):
        self._grid_file = new_grid_file

    @property
    def sens_file_path(self):
        """path to fun3d sens file"""
        return os.path.join(self.analysis_dir, f"{self.project_name}.sens")

    @property
    def flow_directory(self):
        return os.path.join(self.analysis_dir, "Flow")

    @property
    def adjoint_directory(self):
        return os.path.join(self.analysis_dir, "Adjoint")
