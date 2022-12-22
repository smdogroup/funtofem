# __init__ file for pyfuntofem/fun3d_interface subfolder
# classes and methods included in import * are specified in __all__ at the top of each file
# have to make sure that the subfolder is not called fun3d otherwise interferes with fun3d python interface

# check whether fun3d is available on this machine / server
import importlib

fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

# import the funtofem-fun3d files if available
if has_fun3d:
    # don't import deprecated client class
    # from .fun3d_client import *

    from .fun3d_interface import *
