"""
Written by Brian Burke and Sean Engelstad, Georgia Tech SMDO Lab, 2024.
"""

__all__ = ["PointwiseAIM"]

import os
from .fun3d_aim import Fun3dBC
from typing import List


class PointwiseAIM:
    def __init__(self, caps_problem, comm, root=0):
        """MPI wrapper class for AflrAIM from ESP/CAPS"""

        self.caps_problem = caps_problem
        self.comm = comm
        self.root = root

        # holds aflr3 AIM
        self._aim = None

        self._dictOptions = None

        self.root_dir = os.getcwd()

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
                aim="pointwiseAIM", name="pointwise"
            )

            return self._aim

    def run_pointwise(self):
        if self.comm.rank == 0:
            # run AIM pre-analysis
            self.aim.preAnalysis()

            # move to test directory
            os.chdir(self.aim.analysisDir)

            CAPS_GLYPH = os.environ["CAPS_GLYPH"]
            # ranPointwise = False
            for i in range(1):  # can run extra times if having license issues
                os.system(
                    "pointwise -b "
                    + CAPS_GLYPH
                    + "/GeomToMesh.glf caps.egads capsUserDefaults.glf"
                )
                # ranPointwise = os.path.isfile('caps.GeomToMesh.gma') and os.path.isfile('caps.GeomToMesh.ugrid')
                # if ranPointwise: break

            os.chdir(self.root_dir)
            # if (not(ranPointwise)): sys.exit("No pointwise license available")

            # run AIM postanalysis, files in self.pointwiseAim.analysisDir
            self.aim.postAnalysis()

    def main_settings(
        self,
        project_name="Wing",
        mesh_format="AFLR3",
        connector_turn_angle=6.0,
        connector_prox_growth_rate=1.3,
        connector_source_spacing=False,
        domain_algorithm="AdvancingFront",
        domain_max_layers=0,
        domain_growth_rate=1.3,
        domain_iso_type="TriangleQuad",
        domain_decay=0.5,
        domain_trex_AR_limit=200,
        domain_wall_spacing=0,
        block_algorithm="AdvancingFront",
        block_boundary_decay=0.5,
        block_collision_buffer=1.0,
        block_max_skew_angle=175,
        block_edge_max_growth_rate=1.8,
        block_full_layers=0,
        block_max_layers=0,
        block_trex_type="TetPyramid",
    ):
        if self.comm.rank == 0:
            self.aim.input.Proj_Name = project_name
            self.aim.input.Mesh_Format = mesh_format

            # connector level
            self.aim.input.Connector_Turn_Angle = connector_turn_angle
            self.aim.input.Connector_Prox_Growth_Rate = connector_prox_growth_rate
            self.aim.input.Connector_Source_Spacing = connector_source_spacing

            # domain level
            self.aim.input.Domain_Algorithm = (
                domain_algorithm  # "Delaunay", "AdvancingFront", "AdvancingFrontOrtho"
            )
            self.aim.input.Domain_Max_Layers = domain_max_layers
            self.aim.input.Domain_Growth_Rate = domain_growth_rate
            self.aim.input.Domain_TRex_ARLimit = (
                domain_trex_AR_limit  # def 40.0, lower inc mesh size
            )
            self.aim.input.Domain_Decay = domain_decay
            self.aim.input.Domain_Iso_Type = domain_iso_type  # "TriangleQuad"

            self.aim.input.Domain_Wall_Spacing = (
                domain_wall_spacing  # e.g. 1e-5 if turbulent
            )
            # Defined spacing when geometry attributed with PW:WallSpacing $wall (relative to capsMeshLength)

            # block level
            self.aim.input.Block_Algorithm = block_algorithm
            self.aim.input.Block_Boundary_Decay = block_boundary_decay
            self.aim.input.Block_Collision_Buffer = block_collision_buffer
            self.aim.input.Block_Max_Skew_Angle = block_max_skew_angle
            self.aim.input.Block_Edge_Max_Growth_Rate = block_edge_max_growth_rate
            self.aim.input.Block_Full_Layers = block_full_layers
            self.aim.input.Block_Max_Layers = block_max_layers
            self.aim.input.Block_TRexType = block_trex_type

        return

    def bc_mesh_sizing(self, fun3d_bcs: list):
        if self.root_proc:
            self.aim.input.Mesh_Sizing = {
                fun3d_bc.name: fun3d_bc.BC_dict for fun3d_bc in fun3d_bcs
            }
        return

    def save_dict_options(self, dictOptions):
        self._dictOptions = dictOptions

        return self

    def _set_dict_options(self):
        """
        Set AFLR3 and AFLR4 options via dictionaries.
        """
        if self.root_proc and self._dictOptions is not None:
            dictOptions = self._dictOptions

            if dictOptions["pointwiseAIM"] is not None:
                for ind, option in enumerate(dictOptions["pointwiseAIM"]):
                    self.aim.input[option].value = dictOptions["pointwiseAIM"][option]

        return self
