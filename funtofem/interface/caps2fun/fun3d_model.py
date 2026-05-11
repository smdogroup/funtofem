"""
Written by Sean Engelstad and Brian Burke, Georgia Tech SMDO Lab, 2024.
"""

__all__ = ["Fun3dModel"]

import pyCAPS, os, importlib
from .fun3d_aim import Fun3dAim
from .mesh_aim import MeshAim
from .pointwise_aim import PointwiseAIM
from .aflr_aim import Aflr3Aim
from .hc_mesh_morph import HandcraftedMeshMorph

# optional tacs import for caps2tacs
tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")
if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs


class Fun3dModel:
    SURFACE_AIMS = ["egads", "aflr4"]
    VOLUME_AIMS = ["aflr3"]

    def __init__(
        self,
        fun3d_aim,
        mesh_aim: MeshAim,
        comm,
        project_name="fun3d_CAPS",
        root: int = 0,
    ):
        self._fun3d_aim = fun3d_aim
        self._mesh_aim = mesh_aim
        self._surface_aim = mesh_aim.surface_aim
        self._volume_aim = mesh_aim.volume_aim

        self.project_name = project_name
        self.comm = comm
        self.root = root
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
        volume_mesh="aflr3",
        surface_mesh="aflr4",
        mesh_morph=False,
        root: int = 0,
        verbosity=0,
    ):
        """
        Make a pyCAPS problem with the fun3dAIM and mesh AIMs on serial / root proc
        Parameters
        ---------------------------------
        csm_file : filepath
            Filename / full path of ESP/CAPS Constructive Solid Model or .CSM file.
        comm : MPI.COMM
            MPI communicator.
        project_name : str
            Name of the case that is passed to the flow side, e.g., what is used to name the FUN3D input files.
        problem_name : str
            CAPS problem name, internal name used to define the CAPS problem and determines the name of the directory
            that is created by CAPS to build the fluid mesh, geometry, sensitivity files, etc.
        mesh_morph : bool
            Turn mesh morphing on or off for use with shape variables that alter the fluid geometry
            (e.g., when using mesh deformation rather than remeshing).
        root : int
            The rank of the processor that will control this process.
        verbosity : int
            Parameter passed directly to pyCAPS to determine output level.
        """
        caps_problem = None
        if comm.rank == root:
            caps_problem = pyCAPS.Problem(
                problemName=problem_name, capsFile=csm_file, outLevel=verbosity
            )
        fun3d_aim = Fun3dAim(caps_problem, comm, mesh_morph=mesh_morph, root=root)
        mesh_aim = MeshAim(
            caps_problem,
            comm,
            volume_mesh=volume_mesh,
            surface_mesh=surface_mesh,
            root=root,
        )

        comm.Barrier()

        return cls(fun3d_aim, mesh_aim, comm, project_name, root=root)

    @property
    def root_proc(self) -> bool:
        return self.fun3d_aim.root_proc

    @property
    def fun3d_aim(self) -> Fun3dAim:
        return self._fun3d_aim

    @property
    def aflr_aim(self) -> MeshAim:
        """
        Leaving this here for backwards 'compatibility'.
        """
        return self.mesh_aim

    @property
    def mesh_aim(self) -> MeshAim:
        return self._mesh_aim

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
        """
        Set the project names into both aims for grid filenames.
        """
        if self.fun3d_aim.root_proc:
            self.fun3d_aim.aim.input.Proj_Name = self.project_name
        self.fun3d_aim._metadata["project_name"] = self.project_name
        if self.mesh_aim.root_proc:
            if self.mesh_aim.surface_aim is not None:
                self.mesh_aim.surface_aim.aim.input.Proj_Name = self.project_name
            self.mesh_aim.volume_aim.aim.input.Proj_Name = self.project_name
        return

    def set_variables(self, shape_varnames, aero_varnames):
        """input list of ESP/CAPS shape variable names into fun3d aim design dict"""
        # add to the list of variable names
        self._shape_varnames += shape_varnames
        self._aero_varnames += aero_varnames
        # update the variables in the AIM
        self.fun3d_aim.set_variables(self._shape_varnames, self._aero_varnames)

    def register(self, obj):
        if isinstance(obj, caps2tacs.ShapeVariable):
            self._shape_varnames += [obj.name]

        self.fun3d_aim.register(obj)

    @property
    def is_setup(self) -> bool:
        """whether the fun3d model is setup"""
        return self._setup and len(self._shape_varnames) > 0

    def setup(self):
        """
        Setup the fun3d model before analysis.
        """
        self._link_aims()
        self.fun3d_aim.set_boundary_conditions()
        if self.mesh_aim._dictOptions is not None:
            self.mesh_aim._setDictOptions()
        self._set_grid_filename()
        self._setup = True
        return

    def _set_grid_filename(self):
        if isinstance(self.mesh_aim.volume_aim, Aflr3Aim):
            filename = self.project_name + ".lb8.ugrid"
        elif isinstance(self.mesh_aim.volume_aim, PointwiseAIM):
            filename = "caps.GeomTomesh.lb8.ugrid"

        self.fun3d_aim.grid_file = os.path.join(self.mesh_aim.analysis_dir, filename)

        # also set mapbc file
        self.fun3d_aim.mapbc_file = os.path.join(
            self.fun3d_aim.analysis_dir, "Flow", self.fun3d_aim.project_name + ".mapbc"
        )
        return

    def _link_aims(self):
        """link the fun3d to aflr aim"""
        self.mesh_aim.link_surface_mesh()
        if self.root_proc:
            self.fun3d_aim.aim.input["Mesh"].link(
                self.mesh_aim.volume_aim.aim.output["Volume_Mesh"]
            )
        return

    def pre_analysis(self):
        volume_aim = self.mesh_aim.volume_aim
        if isinstance(volume_aim, PointwiseAIM):
            volume_aim.run_pointwise()

        self.fun3d_aim.pre_analysis()
        return

    @property
    def geometry(self):
        return self.fun3d_aim.geometry

    @property
    def handcrafted_mesh_morph(self) -> HandcraftedMeshMorph:
        return self.fun3d_aim.handcrafted_mesh_morph

    @handcrafted_mesh_morph.setter
    def handcrafted_mesh_morph(self, my_hmm: HandcraftedMeshMorph):
        self.fun3d_aim.handcrafted_mesh_morph = my_hmm

    def print_summary(self, file=None):
        """
        Print a summary of the Fun3dModel including project settings, mesh
        configuration, paths, boundary conditions, and design variables.

        Parameters
        ----------
        file : file-like object or str/path-like, optional
            If a string or path-like object is given the summary is written to
            that file (opened in write mode).  If a file-like object is given
            it is used directly.  If None (default) the summary is printed to
            stdout.
        """
        _opened = False
        if file is not None and isinstance(file, (str, os.PathLike)):
            file = open(file, "w")
            _opened = True

        if self.root_proc:
            p = lambda *args, **kw: print(*args, file=file, **kw)

            p("==========================================================")
            p("||               FUN3D Model Summary                    ||")
            p("==========================================================")

            # --- Top-level settings ---
            p("  Project name       :", self.project_name)
            p("  Mesh morph         :", self.mesh_morph)
            p("  Is setup           :", self._setup)
            p("  Is handcrafted     :", self.fun3d_aim.is_handcrafted)

            # --- Variable names ---
            p("")
            p("  Shape variable names")
            p("  --------------------")
            if self._shape_varnames:
                for name in self._shape_varnames:
                    p(f"    {name}")
            else:
                p("    (none)")

            p("")
            p("  Aero variable names")
            p("  -------------------")
            if self._aero_varnames:
                for name in self._aero_varnames:
                    p(f"    {name}")
            else:
                p("    (none)")

            # --- FUN3D AIM paths ---
            p("")
            p("  FUN3D AIM Paths")
            p("  ---------------")
            p("  Analysis dir       :", self.fun3d_aim.analysis_dir)
            p("  Flow dir           :", self.fun3d_aim.flow_directory)
            p("  Adjoint dir        :", self.fun3d_aim.adjoint_directory)
            p("  Grid file          :", self.fun3d_aim.grid_file)
            p("  Mapbc file         :", self.fun3d_aim.mapbc_file)
            p("  Sens file          :", self.fun3d_aim.sens_file_path)
            if self.fun3d_aim.is_handcrafted:
                p("  HC mesh morph file :", self.fun3d_aim.mesh_morph_filepath)

            # --- Grid file destinations (per scenario) ---
            if self.fun3d_aim._grid_filepaths:
                p("")
                p("  Grid file destinations (per scenario)")
                p("  -------------------------------------")
                for i, fp in enumerate(self.fun3d_aim._grid_filepaths):
                    p(f"    [{i}] {fp}")

            # --- Mesh AIM info ---
            p("")
            p("  Mesh AIM")
            p("  --------")
            vol_aim = self.mesh_aim.volume_aim
            surf_aim = self.mesh_aim.surface_aim
            p(
                "  Volume mesher      :",
                type(vol_aim).__name__ if vol_aim is not None else "None",
            )
            p(
                "  Surface mesher     :",
                type(surf_aim).__name__ if surf_aim is not None else "None",
            )
            p("  Mesh analysis dir  :", self.mesh_aim.analysis_dir)

            # --- Boundary conditions ---
            p("")
            p("  Boundary Conditions")
            p("  -------------------")
            bcs = self.fun3d_aim._boundary_conditions
            if bcs:
                for name, bc_dict in bcs.items():
                    p(f"    {str(name):<30s} : {bc_dict}")
            else:
                p("    (none registered)")

            # --- Shape variables (with values) ---
            p("")
            p("  Shape Variables (with current values)")
            p("  -------------------------------------")
            if self.fun3d_aim._shape_variables:
                for sv in self.fun3d_aim._shape_variables:
                    p(f"    {sv.name:<30s} = {sv.value}")
            else:
                p("    (none registered)")

        if _opened:
            file.close()

        return
