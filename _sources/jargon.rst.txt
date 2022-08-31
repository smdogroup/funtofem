FUNtoFEM Parlance
*****************

Retrieving Data 
----------------

**get_** 
When used as a prefix in a method or function, 
*get_* indicates that the method is only used to *return* 
an associated value, rather than performing some computation on a member.

**eval_** is a prefix to indicate that a method is used to compute
something of interest but does not `return` anything.

**extract_**

Storing Data
------------

**set_** is used as a prefix for a method which stores some data 
(often in an object external to the current routine) without *returning* anything.

Adjoint 
-------
Consider a generic adjoint equation:

.. math:: 
    \frac{\partial R}{\partial u}^{T} \psi = - \frac{\partial f}{\partial u}^{T}

We refer to the partial derivative :math:`\frac{\partial R}{\partial u}^{T}`
as `dRdu` or similar. The right hand side of the equation is `adjR_rhs`.

As an example, consider the adjoint equation with respect to the 
nodal forces on the structure:

.. math:: 
    \frac{\partial L}{\partial f_S}^T\psi_L + \frac{\partial S}{\partial f_S}^T\psi_S = 0

We can rearrange this equation as:

.. math:: 
    \frac{\partial L}{\partial f_S}^T\psi_L = -\frac{\partial S}{\partial f_S}^T\psi_S

The partial derivative for the load transfer residual is `dLdfS`.
The right hand side is referred to as `adjL_rhs`.