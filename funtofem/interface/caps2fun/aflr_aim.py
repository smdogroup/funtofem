"""
Written by Brian Burke and Sean Engelstad, Georgia Tech SMDO Lab, 2024.
"""

__all__ = ["Aflr3Aim", "Aflr4Aim"]


class Aflr3Aim:
    def __init__(self, caps_problem, comm, root=0):
        """MPI wrapper class for AflrAIM from ESP/CAPS"""

        self.caps_problem = caps_problem
        self.comm = comm
        self.root = root

        # holds aflr3 AIM
        self._aim = None

        self._dictOptions = None

        # self._build_sub_aim()

        return

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == self.root

    @property
    def aim(self):
        return self._aim

    # @property
    # def analysis_dir(self):
    #     _analysis_dir = None
    #     if self.comm.rank == self.root:
    #         _analysis_dir = self.volume_aim.analysisDir
    #     _analysis_dir = self.comm.bcast(_analysis_dir, root=self.root)
    #     return _analysis_dir

    # def link_surface_mesh(self):
    #     """link the surface mesh to volume mesh"""
    #     if self.root_proc:
    #         self.volume_aim.input["Surface_Mesh"].link(
    #             self.surface_aim.output["Surface_Mesh"]
    #         )

    def _build_sub_aim(self):
        if self.root_proc:
            self._aim = self.caps_problem.analysis.create(aim="aflr3AIM", name="aflr3")

            return self._aim

    def set_boundary_layer(
        self, initial_spacing=0.001, thickness=0.1, max_layers=1000, use_quads=False
    ):
        if self.root_proc:
            self.aim.input.BL_Initial_Spacing = initial_spacing
            self.aim.input.BL_Thickness = thickness
            self.aim.input.BL_Max_Layers = max_layers
            if use_quads and (thickness > 0.0):
                self.aim.input.Mesh_Gen_Input_String = "-blc3"
        return self

    def save_dict_options(self, dictOptions):
        self._dictOptions = dictOptions

        return self

    def _set_dict_options(self):
        """
        Set AFLR3 and AFLR4 options via dictionaries.
        """
        if self.root_proc and self._dictOptions is not None:
            dictOptions = self._dictOptions

            if dictOptions["aflr3AIM"] is not None:
                for ind, option in enumerate(dictOptions["aflr3AIM"]):
                    self.aim.input[option].value = dictOptions["aflr3AIM"][option]

        return self


class Aflr4Aim:
    def __init__(self, caps_problem, comm, root=0):
        """
        MPI wrapper class for Aflr4AIM from ESP/CAPS.
        """

        self.caps_problem = caps_problem
        self.comm = comm
        self.root = root

        self._aim = None

        self._dictOptions = None

        # self._build_sub_aim()

        return

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == self.root

    @property
    def aim(self):
        return self._aim

    def _build_sub_aim(self):
        """
        Only call from root_proc.
        """
        if self.root_proc:
            self._aim = self.caps_problem.analysis.create(aim="aflr4AIM", name="aflr4")

            return self._aim

    def save_dict_options(self, dictOptions: dict = None):
        """
        Optional method to set AFLR4 mesh settings using dictionaries.
        Call this before setting up the FUN3D model.
        """
        self._dictOptions = dictOptions

        return self

    def _set_dict_options(self):
        """
        Set AFLR4 options via dictionaries.
        """

        if self.root_proc and self._dictOptions is not None:
            dictOptions = self._dictOptions

            if dictOptions["aflr4AIM"] is not None:
                for ind, option in enumerate(dictOptions["aflr4AIM"]):
                    self.aim.input[option].value = dictOptions["aflr4AIM"][option]

        return self

    def mesh_sizing(self, fun3d_bcs: list):
        if self.root_proc:
            self.aim.input.Mesh_Sizing = {
                fun3d_bc.name: fun3d_bc.BC_dict for fun3d_bc in fun3d_bcs
            }
        return

    def set_surface_mesh(
        self,
        ff_growth=1.3,
        min_scale=0.01,
        max_scale=1,
        mer_all=1,
        use_aflr4_quads=False,
        use_egads_quads=False,
        mesh_length=None,
    ):
        """
        Set surface mesh properties.
        """

        if self.root_proc:
            self.aim.input.ff_cdfr = ff_growth
            self.aim.input.min_scale = min_scale
            self.aim.input.max_scale = max_scale
            self.aim.input.mer_all = mer_all
            if mesh_length is not None:
                self.aim.input.Mesh_Length_Factor = mesh_length
            if use_aflr4_quads:
                self.aim.input.Mesh_Gen_Input_String = "mquad=1 mpp=3"
            if use_egads_quads:
                self.aim.input.EGADS_Quad = True
