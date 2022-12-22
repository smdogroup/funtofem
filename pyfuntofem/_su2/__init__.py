# Subfolder for pyfuntofem/_su2 interface
# classes / methods in include * are found in __all__ header

# check whether su2 exists
import imp
try:
    imp.find_module('pysu2')
    has_su2 = True
except ImportError:
    has_su2 = False

# if has su2, proceed with imports
if has_su2:
    from .su2_interface import *
