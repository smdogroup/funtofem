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

        self._dictOptions = None

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

    def set_surface_mesh(
        self,
        ff_growth=1.3,
        min_scale=0.005,
        max_scale=0.1,
        mer_all=1,
        use_quads=False,
        mesh_length=None,
    ):
        # set surface mesh properties
        if self.root_proc:
            self.surface_aim.input.ff_cdfr = ff_growth
            self.surface_aim.input.min_scale = min_scale
            self.surface_aim.input.max_scale = max_scale
            self.surface_aim.input.mer_all = mer_all
            if mesh_length is not None:
                self.surface_aim.input.Mesh_Length_Factor = mesh_length
            if use_quads:
                self.surface_aim.input.Mesh_Gen_Input_String = "mquad=1 mpp=3"

        return self

    def set_boundary_layer(
        self, initial_spacing=0.001, thickness=0.1, max_layers=1000, use_quads=False
    ):
        if self.root_proc:
            self.volume_aim.input.BL_Initial_Spacing = initial_spacing
            self.volume_aim.input.BL_Thickness = thickness
            self.volume_aim.input.BL_Max_Layers = max_layers
        if use_quads and (thickness > 0.0):
            self.volume_aim.input.Mesh_Gen_Input_String = "-blc3"
        return self

    def mesh_sizing(self, fun3d_bc):
        if self.root_proc:
            self.surface_aim.input.Mesh_Sizing = {fun3d_bc.name: fun3d_bc.BC_dict}
        return

    def saveDictOptions(self, dictOptions):
        self._dictOptions = dictOptions

        return self

    def _setDictOptions(self):
        """
        Set AFLR3 and AFLR4 options via dictionaries.
        """
        dictOptions = self._dictOptions

        for ind, option in enumerate(dictOptions["aflr4AIM"]):
            self.surface_aim.input[option].value = dictOptions["aflr4AIM"][option]

        for ind, option in enumerate(dictOptions["aflr3AIM"]):
            self.volume_aim.input[option].value = dictOptions["aflr3AIM"][option]

        return self
