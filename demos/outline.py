
import pyfuntofem2

# make a body and add structDVs to it
wing = pyfuntofem2.Body(name="wing")
thicknesses = [0.01, 0.02, 0.03]
for idx,thickness in enumerate(thicknesses):
    structDV = pyfuntofem2.StructVariable(name=f"thick+{idx}",value=thickness)
    wing.add_variables(vars=structDV)

# make a fluid volume and add aeroDVs to it
fluid_volume = pyfuntofem2.FluidVolume(name="outer-air")
aoa = pyfuntofem2.AeroVariable(name="aoa", value=2.0)
fluid_volume.add_variables(vars=aoa)

fluid_volume.variables

m_model = pyfuntofem2.CoupledModel(body=wing, fluid_volume=fluid_volume)
m_model.add_
