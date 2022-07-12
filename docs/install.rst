Installing FUNtoFEM
*******************

Prerequisites
-------------
The following items are needed to use FUNtoFEM:

* blas and lapack
* numpy
* an MPI compiler and mpi4py
* Cython

Furthermore, FUN3D requires PARMETIS and METIS (which must be installed inside PARMETIS). TACS requires METIS.

Steps to compile
----------------
#. Clone the FUNtoFEM git repository
#. In the base 'funtofem' directory, copy the Makefile.in.info to Makefile.in. Edit the Makefile.in to match your compilers, lapack location, etc.
#. To compile the real version, from the base directory, run *make* then *make interface*
#. Add funtofem to your bashrc script:

.. code-block:: 

    export PYTHONPATH=$PYTHONPATH:~/git/funtofem
    export PYTHONPATH=$PYTHONPATH:~/git/funtofem/pyfuntofem

FUN3D
-----
Currently the FUNtoFEM interface is not in the master branch of FUN3D.
To use FUN3D with FUNtoFEM, you need to checkout the 'funtofem' branch from the FUN3D git repository.
Make sure to include *- -enable-python* in your configuration of FUN3D; otherwise, compile FUN3D as normal.
