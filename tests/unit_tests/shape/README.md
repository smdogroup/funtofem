# Steady-State Shape Derivative Tests #
* `test_framework_unsteady_aero_coord.py` - Test the aerodynamic coordinate derivatives of a fully-coupled unsteady analysis with the TestAero + TestStruct solvers.
* `test_framework_unsteady_struct_coord.py` - Test the structural coordinate derivatives of a fully-coupled unsteady analysis with the TestAero + TestStruct solvers.
* `test_funtofem_aero_coord.py` - Test the aerodynamic coordinate derivatives of steady, fully-coupled analysis using the TestAero + TACS solvers.
* `test_funtofem_struct_coord.py` - Test the structural coordinate derivatives of steady, fully-coupled analysis using the TestAero + TACS solvers.
* `test_funtofem_unsteady_aero_coord.py` - Test the aerodynamic coordinate derivatives of a fully-coupled unsteady analysis with the TestAero + TACS solvers.
* `test_funtofem_unsteady_struct_coord.py` - Test the structural coordinate derivatives of a fully-coupled unsteady analysis with the TestAero + TACS solvers.
* `test_tacs_driver_coordinate.py` - Test the structural coordinate derivatives of steady, oneway-coupled TACS analysis.
* `test_tacs_driver_unsteady_coordinate.py` - Test the structural coordinate derivatives of a oneway-coupled TACS analysis.
* `test_tacs_shape_driver.py` - Test the structural shape derivatives in ESP/CAPS using the TACS AIM in a steady, fully-coupled analysis with TestAero + TACS solvers.
* `test_tacs_unsteady_shape_driver.py` - Test the structural shape derivatives in ESP/CAPS using the TACS AIM in an unsteady, fully-coupled analysis with TestAero + TACS solvers.

### Steady Structural Coordinate Derivatives ###
<figure class="image">
  <img src="images/tacs_coord_derivs_full.drawio.png" width=\linewidth/>
</figure>

### Steady Aerodynamic Coordinate Derivatives ###
<figure class="image">
  <img src="images/f2f_aero_coords.png" width=\linewidth/>
</figure>

### ESP/CAPS Structural Shape Derivatives ###
<figure class="image">
  <img src="images/struct_shape_derivs.drawio.png" width=\linewidth/>
</figure>