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
===
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

Model Architecture
==================
FUNtoFEM uses model classes to organize the design and coupling data related to a given problem. 
The model is made up of bodies and scenarios. A model is first created by calling the FUNtoFEMmodel function.
Bodies, scenarios, and design variables can then be added to the model. For example:

.. code-block:: python

     # Create model
     model = FUNtoFEMmodel('myModel')

     # Create a body called 'body0'
     body0 = Body('body0', group=0, boundary=1)
     # Add thickness as a structural design variable to the body
     t = 0.025
     svar = Variable('thickness', value=t, lower=1e-3, upper=1.0)
     body0.add_variable('structural', svar)

     # Set number of iterations (steps)
     steps = 20

     # Add a 'cruise' scenario
     cruise = Scenario('cruise', steps=steps)
     model.add_scenario(cruise)

     # Add a 'drag' function
     drag = Function('cd', analysis_type='aerodynamic')
     cruise.add_function(drag)

     # Add the body to the model after the variables
     model.add_body(body0)

Discipline Solvers
==================
After the model has been defined, instantiate the specific discipline solvers with a call to 
Fun3dInterface for the fluid solver and a call to your structural model (e.g., modelTACS ) for the structural solver.

.. code-block:: python

     # Instantiate the flow and structural solvers
     solvers = {}
     solvers['flow'] = Fun3dInterface(comm, model, flow_dt=1.0, qinf=1.0, 
          thermal_scale=1.0, fun3d_dir=None, forward_options=None, adjoint_options=None)
     solvers['structural'] = modelTACS(comm, tacs_comm, model, n_tacs_procs)

Driver Set-up
=============
The problem driver is instantiated with a call to FUNtoFEMnlbgs.

.. code-block:: python

     # Specify the transfer scheme options
     options = {'scheme': 'meld', 'beta': 0.5, 'npts': 50, 'isym': 1}

     # Instantiate the driver
     struct_master = 0
     aero_master = 0
     driver = FUNtoFEMnlbgs(solvers, comm, tacs_comm, struct_master, comm, 
                    aero_master, model=model, transfer_options=options, 
                    theta_init=0.5, theta_min=0.1)

Driver Call
===========
In order to run simulations, calls to the driver are used. 
In this example, a value for the design variable (thickness) is set.
Then :func:`~funtofem_driver.FUNtoFEMDriver.solve_forward` is called to run the forward analysis and 
:func:`~funtofem_driver.FUNtoFEMDriver.solve_adjoint` is called to run the adjoint analysis.

.. code-block:: python

     # Set variable value
     x0 = np.array([0.025])
     model.set_variables(x0)

     # Get the function value
     fail = driver.solve_forward()
     funcs0 = model.get_functions()
     f0vals = []
     for func in funcs0:
          f0vals.append(func.value)
          if comm.rank == 0:
               print('Function value: ', func.value)

     # Evaluate the function gradient
     fail = driver.solve_adjoint()
     grads = model.get_function_gradients()
     if comm.rank == 0:
          print('Adjoint gradient: ', grads)

