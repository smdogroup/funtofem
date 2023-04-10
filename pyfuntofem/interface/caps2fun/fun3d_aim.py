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
    def inviscid(cls, caps_group, wall_spacing):
        return cls(caps_group, bc_type="inviscid", wall_spacing=wall_spacing)

    @classmethod
    def viscous(cls, caps_group, wall_spacing):
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
    def __init__(self, caps_problem, comm, root=0):
        """Fun3dAim wrapper class for use in FUNtoFEM"""
        self.caps_problem = caps_problem
        self.comm = comm
        self.root = root

        self._aim = None
        self._grid_file = None
        self._grid_filepaths = []

        self._boundary_conditions = {}

        self._build_aim()

        """setup to do ESP/CAPS sens file reading"""
        if self.root_proc:
            # set to not overwrite fun3d nml analysis
            self.aim.input.Overwrite_NML = False
            # fun3d design sensitivities settings
            self.aim.input.Design_SensFile = True
            self.aim.input.Design_Sensitivity = True

        self.metadata = None
        if self.root_proc:
            self.metadata = Fun3dAimMetaData(
                project_name=self.aim.input.Proj_Name, analysis_dir=self.aim.analysisDir
            )
        self.metadata = self.comm.bcast(self.metadata, root=root)

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
        return self.metadata.project_name

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
        src = self.grid_file
        for dest in self.grid_filepaths:
            print(f"source file = {src}")
            print(f"dest file = {dest}")
            shutil.copy(src, dest)
            print(f"file has been moved!")
        return

    def set_variables(self, shape_var_names):
        """input list of ESP/CAPS shape variable names into fun3d aim design dict"""
        if len(shape_var_names) == 0:
            return
        DV_dict = {}
        for dv_name in shape_var_names:
            DV_dict[dv_name] = {}
        print(f"fun3d aim DV dict = {DV_dict}")
        if self.root_proc:
            self.aim.input.Design_Variable = DV_dict
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
        return self.metadata.analysis_dir

    def pre_analysis(self):
        if self.root_proc:
            self.aim.preAnalysis()
            self._move_grid_files()
        self.comm.Barrier()
        if self.root_proc:
            print(f"done with preAnalysis", flush=True)
        return

    def post_analysis(self):
        if self.root_proc:
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
