__all__ = ["AflrAim"]


class AflrAim:
    def __init__(self, caps_problem, comm, root=0):
        """MPI wrapper class for AflrAIM from ESP/CAPS"""

        self.caps_problem = caps_problem
        self.comm = comm
        self.root = root

        # holds aflr4 and aflr3 aims
        self._aflr4_aim = None
        self._aflr3_aim = None

        self._build_aim()
        return

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == self.root

    @property
    def volume_aim(self):
        """volume mesh aim aka aflr3 aim"""
        return self._aflr3_aim

    @property
    def surface_aim(self):
        """surface mesher aim aka aflr4 aim"""
        return self._aflr4_aim

    @property
    def analysis_dir(self):
        _analysis_dir = None
        if self.comm.rank == self.root:
            _analysis_dir = self.volume_aim.analysisDir
        _analysis_dir = self.comm.bcast(_analysis_dir, root=self.root)
        return _analysis_dir

    def link_surface_mesh(self):
        """link the surface mesh to volume mesh"""
        if self.root_proc:
            self.volume_aim.input["Surface_Mesh"].link(
                self.surface_aim.output["Surface_Mesh"]
            )

    def _build_aim(self):
        if self.root_proc:
            self._aflr4_aim = self.caps_problem.analysis.create(
                aim="aflr4AIM", name="aflr4"
            )
            self._aflr3_aim = self.caps_problem.analysis.create(
                aim="aflr3AIM", name="aflr3"
            )
        return

    def set_surface_mesh(self, ff_growth=1.4, min_scale=0.05, max_scale=0.5):
        # set surface mesh properties
        if self.root_proc:
            self.surface_aim.input.ff_cdfr = ff_growth
            self.surface_aim.input.min_scale = min_scale
            self.surface_aim.input.max_scale = max_scale
        return self

    def set_boundary_layer(self, initial_spacing=0.001, thickness=0.1, max_layers=1000):
        if self.root_proc:
            self.volume_aim.input.BL_Initial_Spacing = initial_spacing
            self.volume_aim.input.BL_Thickness = thickness
            self.volume_aim.input.BL_Max_Layers = max_layers
        return self
