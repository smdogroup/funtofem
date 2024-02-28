"""
Written by Brian Burke and Sean Engelstad, Georgia Tech SMDO Lab, 2024.
"""

__all__ = ["EgadsAim"]


class EgadsAim:
    """
    Wrapper class for ESP/CAPS EgadsAim to build surface mesh for Fun3dAim
    Controls the following inputs:
    egadsAim.input.Edge_Point_Min = 15
    egadsAim.input.Edge_Point_Max = 20
    egadsAim.input.Mesh_Elements = "Quad"
    egadsAim.input.Tess_Params = [.25,.01,15]
    """

    def __init__(self, caps_problem, comm, root=0):

        self.caps_problem = caps_problem
        self.comm = comm
        self.root = root

        self._dictOptions = None

        self._aim = None

        return

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == self.root

    @property
    def aim(self):
        return self._aim

    def _build_sub_aim(self):
        if self.root_proc:
            self._aim = self.caps_problem.analysis.create(
                aim="egadsTessAIM", name="egads"
            )

            return self._aim

    def set_surface_mesh(
        self,
        edge_pt_min: int = 15,
        edge_pt_max=20,
        mesh_elements: str = "Quad",
        global_mesh_size: float = 0.25,
        max_surf_offset: float = 0.01,
        max_dihedral_angle: float = 15,
    ):
        """
        Cascaded method to set the mesh input settings to the egadsAim.
        """
        
        if self.root_proc:
            self._aim.input.Edge_Point_Min = edge_pt_min
            self._aim.input.Edge_Point_Max = edge_pt_max
            self._aim.input.Mesh_Elements = mesh_elements
            self._aim.input.Tess_Params = [
                global_mesh_size,
                max_surf_offset,
                max_dihedral_angle,
            ]

        return self

    def save_dict_options(self, dictOptions: dict = None):
        """
        Optional method to set EGADS mesh settings using dictionaries.
        Call this before setting up the FUN3D model. The dictionary should take
        the form of, e.g.:

        dictOptions['egadsTessAIM']['myOption'] = myValue
        """
        self._dictOptions = dictOptions

        return self

    def _set_dict_options(self):
        """
        Set EGADS options via dictionaries.
        """
        if self.root_proc and self._dictOptions is not None:
            dictOptions = self._dictOptions

            if dictOptions["egadsTessAIM"] is not None:
                for option in dictOptions["egadsTessAIM"]:
                    self.aim.input[option].value = dictOptions["egadsTessAIM"][option]
