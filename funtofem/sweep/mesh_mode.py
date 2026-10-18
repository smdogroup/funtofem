"""
funtofem.sweep.mesh_mode — MeshMode enumeration.

Controls which mesh-generation callbacks are invoked for each design point
during a parameter sweep.
"""

from enum import Enum


class MeshMode(Enum):
    """Enumeration of mesh-generation modes for a parameter sweep.

    Attributes
    ----------
    FULL_REGEN : str
        Regenerate both the CFD mesh and the structural mesh for every design
        point.  Both the ``cfd_mesh_callback`` and the ``struct_mesh_callback``
        must be registered.
    CFD_ONLY : str
        Regenerate the CFD mesh only and reuse a prior structural mesh.  Only
        the ``cfd_mesh_callback`` is required; ``struct_mesh_callback`` is not
        invoked.
    STRUCT_ONLY : str
        Reuse a prior CFD mesh and regenerate the structural mesh only.  Only
        the ``struct_mesh_callback`` is required; ``cfd_mesh_callback`` is not
        invoked.
    NONE : str
        Use pre-made meshes; neither the CFD mesh nor the structural mesh is
        regenerated.  Neither mesh callback is required or invoked.
    """

    FULL_REGEN = "full_regen"
    CFD_ONLY = "cfd_only"
    STRUCT_ONLY = "struct_only"
    NONE = "none"
