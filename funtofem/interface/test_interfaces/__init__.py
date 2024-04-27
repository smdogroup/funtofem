import importlib

from ._test_aero_solver import *
from ._test_struct_solver import *
from .pistontheory_interface import *

# FUN3D grid deformation interface, with python package "fun3d"
fun3d_loader = importlib.util.find_spec("fun3d")
if fun3d_loader is not None:
    from .fun3d_grid_interface import *
    from .fun3d_14_therm_interface import *
    from .fun3d_ae_interface import *