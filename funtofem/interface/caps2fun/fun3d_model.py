__all__ = ["Fun3dModel"]

import pyCAPS, os
from .fun3d_aim import Fun3dAim
from .aflr_aim import AflrAim


class Fun3dModel:
    def __init__(self, fun3d_aim, aflr_aim, comm, project_name="caps"):
        self._fun3d_aim = fun3d_aim
        self._aflr_aim = aflr_aim
        self.project_name = project_name
        self.comm = comm
        self._variables = {}

        self.caps_problem = fun3d_aim.caps_problem

        self._set_project_names()

        self._shape_varnames = []
        self._aero_varnames = []
        self._setup = False
        return

    @classmethod
    def build(
        cls,
        csm_file,
        comm,
        project_name="fun3d_CAPS",
        problem_name: str = "capsFluid",
        mesh_morph=False,
    ):
        """
        make a pyCAPS problem with the tacsAIM and egadsAIM on serial / root proc
        Parameters
        ---------------------------------
        csm_file : filepath
            filename / full path of ESP/CAPS Constructive Solid Model or .CSM file
        comm : MPI.COMM
            MPI communicator
        """
        caps_problem = None
        if comm.rank == 0:
            caps_problem = pyCAPS.Problem(
                problemName=problem_name, capsFile=csm_file, outLevel=1
            )
        fun3d_aim = Fun3dAim(caps_problem, comm, mesh_morph=mesh_morph)
        aflr_aim = AflrAim(caps_problem, comm)

        return cls(fun3d_aim, aflr_aim, comm, project_name)

    @classmethod
    def build_morph(
        cls,
        csm_file,
        comm,
        project_name="fun3d_CAPS",
        problem_name: str = "capsFluid",
    ):
        return cls.build(
            csm_file=csm_file,
            comm=comm,
            project_name=project_name,
            problem_name=problem_name,
            mesh_morph=True,
        )

    @property
    def root_proc(self) -> bool:
        return self.fun3d_aim.root_proc

    @property
    def fun3d_aim(self) -> Fun3dAim:
        return self._fun3d_aim

    @property
    def aflr_aim(self) -> AflrAim:
        return self._aflr_aim

    @property
    def mesh_morph(self) -> bool:
        return self.fun3d_aim.mesh_morph

    @property
    def mesh_morph_filename(self):
        return self.fun3d_aim.mesh_morph_filename

    @property
    def mesh_morph_filepath(self):
        return self.fun3d_aim.mesh_morph_filepath

    def _set_project_names(self):
        """set the project names into both aims for grid filenames"""
        if self.fun3d_aim.root_proc:
            self.fun3d_aim.aim.input.Proj_Name = self.project_name
        self.fun3d_aim.metadata.project_name = self.project_name
        if self.aflr_aim.root_proc:
            self.aflr_aim.surface_aim.input.Proj_Name = self.project_name
            self.aflr_aim.volume_aim.input.Proj_Name = self.project_name
        return

    def set_variables(self, shape_varnames, aero_varnames):
        """input list of ESP/CAPS shape variable names into fun3d aim design dict"""
        # add to the list of variable names
        self._shape_varnames += shape_varnames
        self._aero_varnames += aero_varnames
        # update the variables in the AIM
        self.fun3d_aim.set_variables(self._shape_varnames, self._aero_varnames)

    @property
    def is_setup(self) -> bool:
        """whether the fun3d model is setup"""
        return self._setup and len(self._shape_varnames) > 0

    def setup(self):
        """setup the fun3d model before analysis"""
        self._link_aims()
        self.fun3d_aim.set_boundary_conditions()
        self._set_grid_filename()
        self._setup = True
        return

    def _set_grid_filename(self):
        self.fun3d_aim.grid_file = os.path.join(
            self.aflr_aim.analysis_dir, "aflr3_0.lb8.ugrid"
        )
        return

    def _link_aims(self):
        """link the fun3d to aflr aim"""
        self.aflr_aim.link_surface_mesh()
        if self.root_proc:
            self.fun3d_aim.aim.input["Mesh"].link(
                self.aflr_aim.volume_aim.output["Volume_Mesh"]
            )
        return

    @property
    def geometry(self):
        return self.fun3d_aim.geometry
