# __init__ for pyfuntofem/openmdao subfolder

# checks whether openmdao is available
import imp
try:
    imp.find_module('openmdao')
    has_openmdao = True
except ImportError:
    has_openmdao = False

# if openmdao module is available, proceed
if has_openmdao:
    from .openmdao_component import *
