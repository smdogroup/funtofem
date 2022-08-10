import numpy as np
from tacs import elements, constitutive


def thermoelasticity_callback(
    dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs
):
    # Set constitutive properties
    rho = 4540.0  # density, kg/m^3
    E = 118e9  # elastic modulus, Pa 118e9
    nu = 0.325  # poisson's ratio
    ys = 1050e6  # yield stress, Pa
    kappa = 6.89
    specific_heat = 463.0

    # Create the constitutvie propertes and model
    props_plate = constitutive.MaterialProperties(
        rho=rho, specific_heat=specific_heat, kappp=kappa, E=E, nu=nu, ys=ys
    )

    # Create the basis class
    basis = elements.LinearHexaBasis()

    # Create the elements in an element list
    con = constitutive.SolidConstitutive(props_plate, t=1.0, tNum=dvNum)
    phys_model = elements.LinearThermoelasticity3D(con)

    # Create the element
    element = elements.Element3D(phys_model, basis)

    return element


def elasticity_callback(
    dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs
):
    # Set constitutive properties
    rho = 4540.0  # density, kg/m^3
    E = 118e9  # elastic modulus, Pa 118e9
    nu = 0.325  # poisson's ratio
    ys = 1050e6  # yield stress, Pa
    kappa = 6.89
    specific_heat = 463.0

    # Create the constitutvie propertes and model
    props_plate = constitutive.MaterialProperties(
        rho=rho, specific_heat=specific_heat, kappp=kappa, E=E, nu=nu, ys=ys
    )

    # Create the basis class
    basis = elements.LinearHexaBasis()

    # Create the elements in an element list
    con = constitutive.SolidConstitutive(props_plate, t=1.0, tNum=dvNum)
    phys_model = elements.LinearElasticity3D(con)

    # Create the element
    element = elements.Element3D(phys_model, basis)

    return element


def generateBDF(filename):
    """
    Generate a BDF file
    """
    nx = 11
    ny = 11
    nz = 3

    x_min = 1.0
    x_max = 2.0  # m
    y_min = 0.0
    y_max = 1.0  # m

    z_min = -0.015
    z_max = -0.005

    x = np.linspace(x_min, x_max, num=nx)
    y = np.linspace(y_min, y_max, num=ny)
    z = np.linspace(z_min, z_max, num=nz)
    theta = np.radians(5.0)
    nodes = np.zeros((nx, ny, nz), dtype=int)

    fp = open(filename, "w")
    fp.write("$ Input file for a rectangular plate\n")
    fp.write("SOL 103\nCEND\nBEGIN BULK\n")

    spclist = []
    spclistY = []
    spclistT = []

    # Write the grid points to a file
    node = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Write the nodal data
                spc = " "
                coord_disp = 0
                coord_id = 0
                seid = 0

                xpt = np.cos(theta) * x[i] - np.sin(theta) * z[k] + 1.0
                ypt = y[j]
                zpt = np.sin(theta) * x[i] + np.cos(theta) * z[k]

                nodes[i, j, k] = node
                node += 1

                fp.write(
                    "%-8s%16d%16d%16.9e%16.9e*       \n"
                    % ("GRID*", nodes[i, j, k], coord_id, xpt, ypt)
                )
                fp.write(
                    "*       %16.9e%16d%16s%16d        \n"
                    % (zpt, coord_disp, spc, seid)
                )

                # If the node is on one of the bottom outer edges of the plate,
                # restrain it against displacement in any direction
                if k == 0 and (i == 0 or i == nx - 1):
                    spclist.append(nodes[i, j, k])

                # If the node is on one of the outer edges of the plate,
                # restrain it against displacement in y direction
                if k != 0 and (j == 0 or j == ny - 1):
                    spclistY.append(nodes[i, j, k])

                # Set the temperature along the bottom edge of the plate
                if k == 0:
                    spclistT.append(nodes[i, j, k])

    # Write out the linear hexahedral elements
    elem = 1
    for k in range(nodes.shape[2] - 1):
        for j in range(nodes.shape[1] - 1):
            for i in range(nodes.shape[0] - 1):
                # Set different part numbers for the elements on the
                # lower and volume mesh
                part_id = 1

                # Write the connectivity data
                fp.write(
                    "%-8s%8d%8d%8d%8d%8d%8d%8d%8d%8s\n%-8s%8d%8d\n"
                    % (
                        "CHEXA",
                        elem,
                        part_id,
                        nodes[i, j, k],
                        nodes[i + 1, j, k],
                        nodes[i + 1, j + 1, k],
                        nodes[i, j + 1, k],
                        nodes[i, j, k + 1],
                        nodes[i + 1, j, k + 1],
                        "+",
                        "+",
                        nodes[i + 1, j + 1, k + 1],
                        nodes[i, j + 1, k + 1],
                    )
                )
                elem += 1

    for node in spclist:
        spc = "123"
        fp.write("%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, node, spc, 0.0))

    for node in spclistY:
        spc = "2"
        fp.write("%-8s%8d%8d%8s%8.6f\n" % ("SPC", 1, node, spc, 0.0))

    for node in spclistT:
        spc = "4"
        fp.write("%-8s%8d%8d%8s%8.4f\n" % ("SPC", 1, node, spc, 300.0))

    fp.write("ENDDATA")
    fp.close()

    return
