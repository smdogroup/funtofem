#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import each analysis interface subfolder
# each analysis subfolder checks whether the package is available before import

# import the base interface
from .solver_interface import *

# Import all of the funtofem interfaces
# underscores are used here to prevent conflicting imports with fun3d, tacs, etc.
from ._cart3d import *
from ._fun3d import *
#from ._openmdao import *
from ._su2 import *
from ._tacs import *
from ._test import *