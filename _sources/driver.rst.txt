FUNtoFEM Driver
***************

While the FUNtoFEM model holds a lot of the aeroelastic and design data, the driver is what solves the coupled problems. 
The nonlinear block Gauss Seidel is the current driver. The 

To use the driver:

#. Create the SolverManager class
#. (optional) Define a CommManager class. If not, it uses the default CommManager created in the SolverManager.
#. Define the TransferSettings object
#. Instantiate the driver then call the run methods.


Creating the solver manager
-------------------------------
The disciplinary solvers in FUNtoFEM are a collection of the solver interface objects stored in
the `SolverManager` class. The solver interface objects exchange data and ask the disciplinary 
solvers to do tasks like taking forward and adjoint interations and calculating coordinate derivatives.


Here's some pseudocode to create the solvers for the NLBGS driver.

.. code-block:: python

    comm = MPI.COMM_WORLD
    solvers = SolverManager(comm)
    solvers.flow = flow_solver
    solvers.structural = structural_solver

Creating the comm manager
-------------------------------
The disciplinary comms in FUNtoFEM are used in each TransferScheme's in funtofem and setup by the CommManager.
If the CommManager is not defined and input into the FUNtoFEMnlbgs driver then the CommManager will be built
as below with the tacs_comm copied from the tacs_interface. Default CommManager's are not available for other
structural solvers at this time.


Here's some pseudocode to create the solvers for the NLBGS driver.

.. code-block:: python

    comm_manager = CommManager(
        master_comm=comm, 
        struct_comm=tacs_comm, 
        struct_root=0, 
        aero_comm=comm, 
        aero_root=0
    )

Transfer Scheme Set Up
----------------------

Several transfer schemes have been implemented in FUNtoFEM.
The transfer schemes are stored in the FUNtoFEM bodies.
Since each body in the model has it's own transfer scheme instance, different bodies can use different schemes within the same model.
The transfer scheme for each is set up by the driver from a python dictionary of transfer scheme options.
If there are multiple bodies, the *transfer_scheme* argument in the driver can be a list of python dictionaries, one for each body.
If the argument is only one python dictionary but there are multiple bodies, the driver assumes all the bodies will use the same options.
If no transfer scheme options are provide, MELD with the default options is used for all the bodies.


MELD
====
The MELD scheme is the only scheme in FUNtoFEM that currently has implemented all of the derivatives necessary for shape derivatives in optimization.
The default options for MELD are listed below. The argument `isym` is used to specify a symmetry plane. The `beta` argument is a weighting function 
parameter, with lower values of beta giving higher weight among the nearest neighbors. The `npts` argument specifies how many nearest neighbors are 
used in the transfer scheme from structural to aerodynamic mesh and vice versa.

Options for `isym`: 

* -1 for no symmetry
* 0 for symmetry across x = 0
* 1 for symmetry across y = 0
* 2 for symmetry across z = 0

.. code-block:: python

    transfer_settings = TransferSettings(
        elastic_scheme="meld",
        isym=-1,
        beta=0.5,
        npts=200,
    )

Linearized MELD
===============
The linearized version of MELD has a lower computational cost than the full version, but it also more error prone for larger deflections and extrapolations.
It cannot reproduce rigid motion, and should not be used for case such as rotors.

.. code-block:: python

    transfer_settings = TransferSettings(
        elastic_scheme="linearized meld",
        beta=0.5,
        npts=200,
    )


Radial Basis Function
=====================
Radial basis function (RBF) interpolation is available in FUNtoFEM with various basis functions.

The options available for the RBF scheme are the different basis functions:

.. code-block:: python

    # Available basis functions:
    # 'thin plate spline'
    # 'gaussian'
    # 'multiquadric'
    # 'inverse multiquadric'
    transfer_settings = TransferSettings(
        elastic_scheme="rbf",
        beta=0.5,
        npts=200,
        options = {'basis function' : 'thin plate spline'},
    )

Beam
====
The beam transfer scheme can transfer data when all of structural nodes are colinear.
The implemented version is based on the method of Brown.
Unlike the other transfer schemes, the beam transfer has six degrees of freedom at each structural node where the 
additional degrees of freedom are the Euler parameter vector for the rotation.
For the aerodynamic nodes, the force integration and displacement transfer still only three degrees of freedom.

Running the coupled FUNtoFEM driver
-----------------------------------
Once the model, solver dictionary, transfer settings are created, you can instantiate the driver and run the coupled forward and adjoint solvers.
Often times, we use the default comm manager and don't specify it as follows.

.. code-block:: python

   driver = FUNtoFEMnlbgs(solvers, transfer_settings=transfer_settings, model=model)

   fail = driver.solve_forward()

   fail = driver.solve_adjoint()
   
If a user-defined CommManager is built, then the coupled funtofem driver is built as follows.

.. code-block:: python

   driver = FUNtoFEMnlbgs(solvers, comm_manager, transfer_settings, model)

   fail = driver.solve_forward()

   fail = driver.solve_adjoint()

Setting up a design optimization
--------------------------------
See :doc:`model` for explanation of using the driver and model class for a design optimization. There is also an example in the examples directory.

If the user wants a better initial design before optimization with the fully-coupled funtofem driver above, the
`TacsOnewayDriver` can be used to optimize over fixed aerodynamic loads first. While the funtofem driver does not 
include shape variables directly in the driver at the moment, the tacs oneway driver supports shape optimization
with caps2tacs wrapper on the tacsAIM from ESP/CAPS shape variables.

Building and using the TacsOnewayDriver
---------------------------------------
Without shape optimization

.. code-block:: python

   tacs_driver = TacsOnewayDriver.prime_loads(funtofem_driver)

   fail = tacs_driver.solve_forward()

   fail = tacs_driver.solve_adjoint()

With the shape optimization 

.. code-block:: python

   tacs_driver = TacsOnewayDriver.prime_loads_from_shape(
        solvers.flow, 
        tacs_aim, 
        transfer_settings, 
        nprocs
    )

   fail = tacs_driver.solve_forward()

   fail = tacs_driver.solve_adjoint()


Driver Classes
--------------

FUNtoFEM Driver Class
=====================
.. currentmodule:: funtofem.driver

.. autoclass:: FUNtoFEMDriver
    :members:

FUNtoFEM NLBGS Driver Class
===========================
.. currentmodule:: funtofem.driver

.. autoclass:: FUNtoFEMnlbgs
    :members:

TACS Oneway-Coupled Driver Class
================================
.. currentmodule:: funtofem.driver

.. autoclass:: TacsOnewayDriver
    :members:
