pyFUNtoFEM
**********
FUNtoFEM can be executed by writing a Python script. 
Tha main components are the import section, MPI set-up, model creation, instantiation of discipline solvers, and calling the relevant routines.

Import Modules
==============
At the beginning of your Python script, it is important to import the appropriate modules. 
In the following code excerpt, modelTACS refers to a Python class which defines the structural model.

.. code-block:: python

   import os, sys
   from pyfuntofem.model import *
   from pyfuntofem.driver import *
   from pyfuntofem.fun3d_interface import *

   from tacs_model import modelTACS
   from pyOpt import Optimization
   from mpi4py import MPI

MPI
---
FUNtoFEM employs MPI to enable multiple processes to run in parallel.
The next section of the Python script handles the MPI set-up.

.. code-block:: python

   n_tacs_procs = 1
   comm = MPI.COMM_WORLD

   world_rank = comm.Get_rank()
   if world_rank < n_tacs_procs:
        color = 55
        key = world_rank
   else:
        color = MPI.UNDEFINED
        key = world_rank
   tacs_comm = comm.Split(color, key)

