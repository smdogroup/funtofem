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
   from pyfuntofem import *
   from pyoptsparse import SNOPT, Optimization
   from mpi4py import MPI

MPI
===
FUNtoFEM employs MPI to enable multiple processes to run in parallel.
The next section of the Python script handles the MPI set-up.

.. code-block:: python

   comm = MPI.COMM_WORLD

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

Shortcuts to creating bodies, scenarios, and a FUNtoFEMmodel are available with new formulation. Classmethods
for variables include :func:`~Variable.structural`, :func:`~Variable.aerodynamic`, :func:`~Variable.shape`.
Classmethods for bodies include :func:`~Body.aeroelastic`, :func:`~Body.aerothermal`, :func:`~Body.aerothermoelastic`.
Classmethods for scenarios include :func:`~Scenario.steady`, :func:`~Scenario.unsteady`. Classmethods for functions
include :func:`~Function.ksfailure`, :func:`~Function.mass`, :func:`~Function.lift`, :func:`~Function.drag`. Registration methods
are available for bodies and scenarios to the model, variables to the body or scenario, and functions to be included in a scenario.

.. code-block:: python

     # Create model and body
     model = FUNtoFEMmodel('myModel')
     body = Body.aeroelastic('body0', boundary=1)

     # Add thickness as a structural design variable to the body
     Variable.structural('thickness').set_bounds(
          lower=1e-3, value=0.025, upper=1.0
     ).register_to(body)

     # register body to model
     body.register_to(model)

     # Add a 'cruise' scenario and register to model
     cruise = Scenario.steady('cruise', steps=20).include(Function.drag()).include(Function.mass())
     cruise.register_to(model)

Discipline Solvers
==================
After the model has been defined, instantiate the specific discipline solvers with a call to 
Fun3dInterface for the fluid solver and a call to TacsSteadyInterface or TacsUnsteadyinterface
for the structural solver.

.. code-block:: python

     # Instantiate the flow and structural solvers
     comm = MPI.COMM_WORLD
     bdf_filename = os.path.join(os.getcwd(), "meshes", "nastran_CAPS.dat") # dat file from tacsAIM includes .bdf file + constraints, loads, dvs

     solvers = SolverManager(comm)
     solvers.flow = Fun3dInterface(comm, model, fun3d_dir=None, forward_options=None, adjoint_options=None)
     solvers.flow.set_units(flow_dt=1.0, qinf=1.0)
     solvers.structural = TacsSteadyInterface.create_from_bdf(model, comm, n_tacs_procs=1, bdf_filename=bdf_filename)

Building a Coupled Funtofem Driver
==================================
The problem driver is instantiated with a call to FUNtoFEMnlbgs.

.. code-block:: python

     # Specify the transfer scheme options
     transfer_settings = TransferSettings(
          elastic_scheme="meld", thermal_scheme="meld",
          beta=0.5, npts=50, isym=1
     )

     # Instantiate the funtofem coupled driver
     funtofem_driver = FUNtoFEMnlbgs(solvers, transfer_settings=transfer_settings, model=model)

Building a Tacs Oneway-Coupled Driver
=====================================
Once a coupled driver is created with the ability to compute aerodynamic loads, the
class method :func:`~TacsSteadyAnalysisDriver.prime_loads` is used to create the driver.
It automatically runs a forward analysis of the coupled driver, saves the aero loads and heat
fluxes as states in the bodies and constructs the driver. An optimization manager for pyoptsparse
or an openmdao component can then be made to proceed to optimization.

.. code-block:: python

     # option 1 use class method to prime loads
     tacs_driver = TacsSteadyAnalysisDriver.prime_loads(funtofem_driver)

     # option 2 prime the loads yourself
     funtofem_driver.solve_forward()
     tacs_driver = TacsSteadyAnalysisDriver(solvers, model)

     # then use solve_forward and solve_adjoint inside an optimizer function
     tacs_driver.solve_forward()
     tacs_driver.solve_adjoint()

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

