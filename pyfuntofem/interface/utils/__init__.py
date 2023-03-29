# __init__ file for pyfuntofem/interface/utils
# Provides utilities to each of the analysis interfaces in funtofem

# use importlib package to check available packages
import importlib

fun3d_loader = importlib.util.find_spec("fun3d")

# currently active utilities
from .cart3d_utils import *

# tacs utils
tacs_loader = importlib.util.find_spec("tacs")
if tacs_loader is not None:
    from .funtofem_callback import *

# FUN3D grid deformation interface, with python package "fun3d"
fun3d_loader = importlib.util.find_spec("fun3d")
if fun3d_loader is not None:
    from .fun3d_grid_interface import *

# need to be updated and therefore commmented out for now
# if fun3d_loader is not None: from .fun3d_client import *
