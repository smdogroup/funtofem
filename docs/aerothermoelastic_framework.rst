Aerothermoelastic Framework
*****************************

.. automodule:: AerothermoelasticFramework

.. figure:: images/aerothermoelastic_framework.png

The surface displacements are computed by solving the displacement transfer residuals and preserving rigid-body motion. The displacement transfer scheme is given by:

.. math::

	{D}({x}, {u}_{S}, {u}_{A}) \triangleq {\xi}({x}, {u}_{S}) - {u}_{A} = 0


To obtain a consistent and conservative load transfer, the load transfer is derived based on the method of virtual work. The residual of the load transfer scheme is:

.. math::

	{L}({x}, {u}_{S}, {f}_{A}, {f}_{S}) \triangleq {\eta}({x}, {u}_{S}, {f}_{A}) - {f}_{S} = 0

MELDThermal links each aerodynamic surface node, where a wall temperature will be specified, to a fixed number of the nearest structural nodes from which the structural temperature will be interpolated. This approach is analogous to the localization property of MELD such that each aerodynamic surface node receives temperature information from a limited number of structural nodes. The temperature of the aerodynamic surface node is then computed from the temperatures of the set of linked structural nodes:

.. math::

	T_{A} = \sum_{i=1}^{N} w_{i} {T_{S}}_{i}

The weights are computed based on the Euclidean distance between the aerodynamic node and the corresponding structural surface nodes:

.. math::

	w_{i} = e^{- \beta d_{i}^2}  \Bigg/ \sum_{j=1}^{N} e^{- \beta d_{j}^2}

The interpolation is repeated for all aerodynamic surface nodes, giving the temperature transfer residual:

.. math::

	{T}({t}_{S}, {t}_{A}) \triangleq  {W} {t}_{S} - {t}_{A} = 0,

The relationship between the area-weighted heat flux at the aerodynamic surface nodes and the resulting heat flux on the structural nodes is calculated in the same manner as the loads. Based on virtual work, the flux produced at a structural node by the force at an aerodynamic surface node is:

.. math::

	{Q}({f}_{T,A}, {f}_{T,S}) \triangleq {W}^{T} {f}_{T,A} - {f}_{T,S} = 0
