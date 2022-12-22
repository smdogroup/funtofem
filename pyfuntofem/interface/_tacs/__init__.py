# __init__ file for subfolder pyfuntofem/_tacs
# provides TACS steady and unsteady interfaces
# as well as one-way coupled drivers for TACS

# check whether TACS package exists
import imp
try:
    imp.find_module('tacs')
    has_tacs = True
except ImportError:
    has_tacs = False

# if has TACS, proceed with imports
# classes/methods included in import * found in __all__ at the
# top of the file
if has_tacs:
    from .tacs_interface import *
    from .tacs_interface_unsteady import *
    from .tacs_driver import *
