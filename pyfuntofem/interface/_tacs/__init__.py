# __init__ file for subfolder pyfuntofem/_tacs
# provides TACS steady and unsteady interfaces
# as well as one-way coupled drivers for TACS

# check whether TACS package exists
import importlib
tacs_loader = importlib.util.find_spec('tacs')
has_tacs = tacs_loader is not None

# if has TACS, proceed with imports
# classes/methods included in import * found in __all__ at the
# top of the file
if has_tacs:
    from .tacs_interface import *
    from .tacs_interface_unsteady import *
    from .tacs_driver import *
