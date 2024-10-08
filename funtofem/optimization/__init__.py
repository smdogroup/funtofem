# __init__ file in funtofem/optimization subfolder
# import * classes and methods specified in __all__

import importlib

from .optimization_manager import *
from .pyopt_optimization import *

# openmdao_loader = importlib.util.find_spec("openmdao")
# if openmdao_loader is not None:
#     from .openmdao_component import *

niceplots_loader = importlib.util.find_spec("niceplots")
if niceplots_loader is not None:
    from .utils import *
