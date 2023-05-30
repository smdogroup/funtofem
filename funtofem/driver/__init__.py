# __init__.py file for funtofem driver folder
# represents funtofem fully-coupled drivers

# the classes/methods in import * are detailed in __all__ at the
# top of each file

# importlib to check available packages
import importlib

tacs_loader = importlib.util.find_spec("tacs")

# import base funtofem driver
from ._funtofem_driver import *

# import the two fully coupled funtofem drivers
from .funtofem_nlbgs_driver import *
from .funtofem_nlbgs_fsi_subiters_driver import *
from .transfer_settings import *
from .test_drivers import *
from .funtofem_shape_driver import *

# import the two tacs drivers if tacs is available
if tacs_loader is not None:
    from .tacs_oneway_driver import *

fun3d_loader = importlib.util.find_spec("fun3d")
if fun3d_loader is not None:
    from .fun3d_oneway_driver import *
