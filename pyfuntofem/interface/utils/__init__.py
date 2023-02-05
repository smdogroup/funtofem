# __init__ file for pyfuntofem/interface/utils
# Provides utilities to each of the analysis interfaces in funtofem

# use importlib package to check available packages
import importlib

openmdao_loader = importlib.util.find_spec("openmdao")
fun3d_loader = importlib.util.find_spec("fun3d")

# currently active utilities
from .cart3d_utils import *

# tacs utils
tacs_loader = importlib.util.find_spec("tacs")
if tacs_loader is not None:
    from .funtofem_callback import *

# need to be updated and therefore commmented out for now
# if openmdao_loader is not None: from .openmdao import *
# if fun3d_loader is not None: from .fun3d_client import *
