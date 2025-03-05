# __init__.py file for funtofem driver folder
# represents funtofem fully-coupled drivers

# the classes/methods in import * are detailed in __all__ at the
# top of each file
import importlib

caps_loader = importlib.util.find_spec("pyCAPS")

# import base funtofem driver
from ._funtofem_driver import *
from ._test_drivers import *

# import the two fully coupled funtofem drivers
from .funtofem_nlbgs_driver import *
from .funtofem_nlbgs_fsi_subiters_driver import *
from .transfer_settings import *
from .oneway_struct_driver import *

if caps_loader is not None:
    from .funtofem_shape_driver import *
    from .oneway_aero_driver import *
