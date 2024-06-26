# Simple Subsonic Wing
Testing out fully-coupled aeroelastic optimizations on a simple wing geometry.

## Preliminary Sizing
* Run the _run_flow.py to get the aero loads file
* Use the aero loads file to do a oneway sizing optimization (for panel thicknesses)

## No Shape, Fully Coupled Optimizations
* 1_panel_thickness.py
* 2_aero_aoa.py

## Pure Shape, Fully Coupled Optimizations
Twist variable is not used at root station (fixed there)
* 3_geom_twist.py - twist variables at each station 
* 4_oml_shape.py - twist + airfoil thickness at each station

## Shape + Discipline DVs, Fully Coupled Optimization
Putting it all together
* 5_shape_and_struct.py - geom twist variables + panel thickness struct variables
* 6_shape_and_aero.py - geom twist variables + aero aoa variable + airfoil thickness
* 7_kitchen_sink.py - put all previous variables together : geom twist, airfoil thickness, AOA, panel thickness
