FUNtoFEM Driver
***************

While the FUNtoFEM model holds a lot of the aeroelastic and design data, the driver is what solves the coupled problems. 
The nonlinear block Gauss Seidel is the current driver.

To use the driver:

#. Create the solver dictionary
#. Set up the transfer scheme options
#. Instantiate the driver then call the run methods.


Creating the solver dictionary
-------------------------------
The disciplinary solvers in FUNtoFEM are a dictionary of the solver interface objects. 
The solver interface objects exchange data and ask the disciplinary solvers to do tasks like 
taking forward and adjoint interations and calculating coordinate derivatives.


Here's some pseudocode to create the solvers for the NLBGS driver.

.. code-block:: python

    solvers = {}
    solvers['flow'] = flow_solver()
    solvers['structural'] = structural_solver()

Transfer scheme set up
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
The default options for MELD are:


.. code-block:: python

    transfer_options = {'scheme':'MELD'}

    # symmetry plane -1-> no symmetry
    #                 0-> x=0 plane symmetry
    #                 1-> y=0 plane symmetry
    #                 2-> z=0 sym
    transfer_options['isym'] = -1

    # weighting function parameter. As lower values of beta -> more equal weighting
    transfer_options['beta'] = 0.5

    # number of structural nodes each aerodynamic node is connected to
    transfer_options['npts'] = 200


Linearized MELD
===============
The linearized version of MELD has a lower computational cost than the full version, but it also more error prone for larger deflections and extrapolations.
It cannot reproduce rigid motion, and should not be used for case such as rotors.

.. code-block:: python

    transfer_options = {'scheme':'linearized MELD'}

    # weighting function parameter. As lower values of beta -> more equal weighting
    transfer_options['beta'] = 0.5

    # number of structural nodes each aerodynamic node is connected to
    transfer_options['npts'] = 200


Radial Basis Function
=====================
Radial basis function (RBF) interpolation is available in FUNtoFEM with various basis functions.

The options available for the RBF scheme are the different basis functions:

.. code-block:: python

    transfer_options = {'scheme':'RBF'}

    # Available basis functions:
    # 'thin plate spline'
    # 'gaussian'
    # 'multiquadric'
    # 'inverse multiquadric'
    transfer_options['basis function'] = 'thin plate spline'


Beam
====
The beam transfer scheme can transfer data when all of structural nodes are colinear.
The implemented version is based on the method of Brown.
Unlike the other transfer schemes, the beam transfer has six degrees of freedom at each structural node where the 
additional degrees of freedom are the Euler parameter vector for the rotation.
For the aerodynamic nodes, the force integration and displacement transfer still only three degrees of freedom.

Running the driver
------------------
Once the model, solver dictionary, transfer options are created, you can instantiate the driver and run the coupled forward and adjoint solvers.

.. code-block:: python

   driver = FUNtoFEMnlbgs(solvers, comm, transfer_options, model)

   fail = driver.solve_forward()

   fail = driver.solve_adjoint()

Setting up a design optimization
--------------------------------
See :doc:`model` for explanation of using the driver and model class for a design optimization. There is also an example in the examples directory.


Driver Classes
--------------

FUNtoFEM Driver Class
=====================
.. currentmodule:: pyfuntofem.funtofem_driver

.. autoclass:: FUNtoFEMDriver
    :members:

FUNtoFEM NLBGS Driver Class
===========================
.. currentmodule:: pyfuntofem.funtofem_nlbgs_driver

.. autoclass:: FUNtoFEMnlbgs
    :members:
