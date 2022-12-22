# __init__ file for pyfuntofem/fun3d_interface subfolder
# classes and methods included in import * are specified in __all__ at the top of each file
# have to make sure that the subfolder is not called fun3d otherwise interferes with fun3d python interface

# check whether fun3d is available on this machine / server
import imp
try:
    imp.find_module('fun3d')
    #from fun3d.solvers import Flow, Adjoint
    fun3d_avail = True
except ImportError:
    fun3d_avail = False

# import the funtofem-fun3d files if available
if fun3d_avail:
    # don't import deprecated client class
    #from .fun3d_client import *

    from .fun3d_interface import *
