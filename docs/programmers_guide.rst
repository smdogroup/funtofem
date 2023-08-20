Programmer's Guide
******************

This section explains how to add your solver, shape parameterization, algorithm to the FUNtoFEM framework.
It will be helpful to be familiar with the existing driver and model classes before attempting to add or modify the framework.

Any add new variables you add to the body and scenario classes will be accessible to any of the solvers, so if you need to exchange or just store data, this is a good place to do it.

Create an interface class for a solver
--------------------------------------

The interface class for the solver allows the FUNtoFEM driver to instruct the solver to perform tasks like iterate and has some setters and getters to exchange data.
The setters and getters are written from the perspective of FUNtoFEM, i.e., the direction of the setters and getters is such that the setters put data from FUNtoFEM into the solver and the getters retrieve data from the solver to be used in FUNtoFEM.
If your solver does not need a particular method described below, leaving it as a *pass* as in the base class is acceptable.

The file solver_interface.py contains the base interface class which has the routines necessary for using the nonlinear block Gauss Seidel solver.
All of the methods in this class (except set_mesh) have the current scenario and the list of bodies as inputs.
These objects contain the coupling input data for the solver and are where the solver interface needs to set the coupling output.

Data to be set during initialization
====================================

Before instantiating the driver, the :class:`~body.Body` objects need to contain some information about the meshes.
It is recommended to do this during the instantiation of the solvers in the solver dictionary.
For an aerodynamic solver, the :class:`~body.Body` holds the surface mesh and number of nodes on the processor: body.aero_X and body.aero_nnodes.
aero_X is a flattened array of length 3*aero_nnodes.
For a structural solver, the :class:`~body.Body` holds the entire structural mesh and number of nodes: body.struct_X and body.struct_nnodes.
struct_X is a flattened array of length 3*struct_nnodes.

Solver Interface Class
======================
.. automodule:: funtofem.interface

.. autoclass:: SolverInterface
    :members:

Create a body class with shape parameterization
-----------------------------------------------
To create a shape parameterization, three functions need to added to the body class:

initialize_shape_parameterization()
===================================

update_shape
============

shape_derivative
================

Create a new driver
-------------------
To create a new driver, you only need to add the algorithms use to solve the coupled equations: _solve_steady_forward, _solve_steady_adjoint, _solve_unsteady_forward, _solve_unsteady_adjoint.
If you are only need either a steady or unsteady driver, you do not need to add the other pair of functions.
The functions are hidden (begin with an underscore) since they should only be called by the driver.
The user should call :func:`~funtofem_driver.FUNtoFEMDriver.solve_forward` and :func:`~funtofem_driver.FUNtoFEMDriver.solve_adjoint`.
Within the driver, the programmer has the freedom to make calls in certain order.
