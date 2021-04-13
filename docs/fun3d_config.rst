FUN3D Configuration Requirements
********************************

In order to configure FUNtoFEM to be utilized with FUN3D, two settings are required to be initialized. These are the "fun3d.nml" and "moving_body.input" files. Within the FUN3D example, these files must be located in the following, relative path: "/example/steady/Flow/". The following lines of code must be added to their respective files. 


FUN3D Namelist (fun3d.nml)
==========================

.. code-block:: fortran

   funtofem_include_skin_friction = .true.


Input File (moving_body.input)
==============================

.. code-block:: fortran

   motion_driver(1) = 'funtofem', ! tells fun3d to use motion inputs from python
   mesh_movement(1) = 'deform',  ! can use 'rigid', 'deform', 'rigid+deform' with funtofem interface


