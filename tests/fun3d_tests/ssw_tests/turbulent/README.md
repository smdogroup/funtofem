# "Super Simple Wing"
Testing out fully-coupled aeroelastic optimizations on a simple wing geometry.
This directory is only meant to run the inviscid cases. Script 1 panel thickness should be panel thickness inviscid.py

## Flight Conditions
Here we consider a steady cruise at FL 100 (approximately 3000 meters). 
From 1976 Standard Atmosphere:
* $T_{\infty}=268.338\text{ K}$
* $p_{\infty}=69.68\text{ kPa}$
* $\rho_\infty=0.904637\text{ kg/m}^3$
* $\mu_\infty=1.7115\times10^{-5}\text{ Pa}\cdot\text{s}$

Stipulate flight speed of Mach 0.5 (164.1935 m/s), which results in a dynamic pressure of $q_\infty=12.1945\text{ kPa}$ and a Reynolds number of $Re_L=8.77639\times10^{6}$. Set a flight angle of attack of two degreees.

## Preliminary Sizing
* Run the _run_flow.py to get the aero loads file
* Use the aero loads file to do a oneway sizing optimization (for panel thicknesses)

## No Shape, Fully Coupled Optimizations
* 1_panel_thickness.py
* 2_aero_aoa.py

## Pure Shape, Fully Coupled Optimizations
Twist variable is not used at root station (fixed there):
* 3_geom_twist.py - twist variables at each station 
* 4_oml_shape.py - twist + airfoil thickness at each station

## Shape + Discipline DVs, Fully Coupled Optimization
Putting it all together:
* 5_shape_and_struct.py - geom twist variables + panel thickness struct variables
* 6_shape_and_aero.py - geom twist variables + aero aoa variable + airfoil thickness
* 7_kitchen_sink.py - put all previous variables together : geom twist, airfoil thickness, AOA, panel thickness
