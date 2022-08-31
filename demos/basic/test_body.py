import pyfuntofem2

m_body = pyfuntofem2.Body(name="wing")

aoa = pyfuntofem2.AeroVariable(name="aoa", value=2.0)
thick1 = pyfuntofem2.StructVariable(name="thick1", value=0.01)
thick2 = pyfuntofem2.StructVariable(name="thick2", value=0.02)

m_body.add_variable([aoa, thick1, thick2])


for DV in m_body.variables:
    print(DV)