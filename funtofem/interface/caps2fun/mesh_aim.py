"""
Written by Brian Burke and Sean Engelstad, Georgia Tech SMDO Lab, 2024.
"""

__all__ = ["MeshAim"]

from .aflr_aim import Aflr3Aim, Aflr4Aim
from .pointwise_aim import PointwiseAIM
from .egads_aim import EgadsAim


class MeshAim:
    def __init__(
        self, caps_problem, comm, volume_mesh="aflr3", surface_mesh="aflr4", root=0
    ):
        """
        MPI wrapper class for setting up the mesh AIMs from ESP/CAPS.
        """

        self.caps_problem = caps_problem
        self.comm = comm
        self.root = root

        self._surf_aim = None
        self._vol_aim = None

        # Set volume and surface mesh AIMs
        if volume_mesh == "aflr3":
            self._vol_aim = Aflr3Aim(caps_problem=caps_problem, comm=comm, root=root)
        elif volume_mesh == "pointwise":
            self._vol_aim = PointwiseAIM(
                caps_problem=caps_problem, comm=comm, root=root
            )
        else:
            raise RuntimeError("Unrecognized volume mesher.")

        if surface_mesh == "aflr4":
            self._surf_aim = Aflr4Aim(caps_problem=caps_problem, comm=comm, root=root)
        elif surface_mesh == "egads":
            self._surf_aim = EgadsAim(caps_problem=caps_problem, comm=comm, root=root)
        elif surface_mesh is None:
            self._surf_aim = None
        else:
            raise RuntimeError("Unrecognized surface mesher.")

        self._dictOptions = None

        self._build_aim()
        return

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == self.root

    @property
    def volume_aim(self):
        """
        Volume mesh AIM. Currently only supports AFLR3.
        """
        return self._vol_aim

    @property
    def surface_aim(self):
        """
        Surface mesh AIM. Currently supports EGADS and AFLR4.
        """
        return self._surf_aim

    @property
    def analysis_dir(self):
        _analysis_dir = None
        if self.comm.rank == self.root:
            _analysis_dir = self.volume_aim.aim.analysisDir
        _analysis_dir = self.comm.bcast(_analysis_dir, root=self.root)
        return _analysis_dir

    def link_surface_mesh(self):
        """link the surface mesh to volume mesh"""
        if self.root_proc and self.surface_aim is not None:
            self.volume_aim.aim.input["Surface_Mesh"].link(
                self.surface_aim.aim.output["Surface_Mesh"]
            )

    def _build_aim(self):
        if self.root_proc:
            # self._vol_aim = self.volume_aim._build_sub_aim()
            # self._surf_aim = self.surface_aim._build_sub_aim()
            self.volume_aim._build_sub_aim()
            if self.surface_aim is not None:
                self.surface_aim._build_sub_aim()
        return

    def saveDictOptions(self, dictOptions):
        self._dictOptions = dictOptions

        self.volume_aim.save_dict_options(dictOptions)
        if self.surface_aim is not None:
            self.surface_aim.save_dict_options(dictOptions)

        return self

    def _setDictOptions(self):
        """
        Set AFLR3 and AFLR4 options via dictionaries.
        """
        if self.root_proc:
            dictOptions = self._dictOptions

            self.volume_aim._set_dict_options()
            if self.surface_aim is not None:
                self.surface_aim._set_dict_options()

        return self
