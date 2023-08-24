# FUNtoFEM Tests #
Github workflow tests are included in the `unit_tests` folder. More comprehensive FUN3D tests are located in the `fun3d_tests` directory. The adjoint equations, total derivatives, shape derivatives, and verification matrix for FUN3D-TACS coupling are provided below. Adjoint equations for alternative solvers may be provided in the future.

### Adjoint Equations ###
<figure class="image">
  <img src="unit_tests/framework/images/f2f_adjoint_eqns.png" width=\linewidth/>
</figure>

### Discipline Total Derivatives ###
<figure class="image">
  <img src="unit_tests/framework/images/f2f_discipline_totalderivs.drawio.png" width=\linewidth/>
</figure>

### Shape Derivatives ###
<figure class="image">
  <img src="unit_tests/framework/images/f2f_shape_derivs.drawio.png" width=\linewidth/>
</figure>

### Adjoint-Jacobian Product Verification Matrix ###
<figure class="image">
  <img src="unit_tests/framework/images/f2f_adjointJac_tests.drawio.png" width=\linewidth/>
</figure>