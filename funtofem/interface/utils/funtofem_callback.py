__all__ = ["f2f_callback", "addLoadsFromBDF"]

import numpy as np
from tacs import constitutive, elements, TACS
from mpi4py import MPI


# define custom funtofem element callback for appropriate assignment of DVs and for elastic/thermoelastic shells
def f2f_callback(fea_assembler, structDV_names, structDV_dict, include_thermal=False):
    def element_callback(
        dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs
    ):
        # Make sure cross-referencing is turned on in pynastran
        # this allows it to read the material cards later on
        if fea_assembler.bdfInfo.is_xrefed is False:
            fea_assembler.bdfInfo.cross_reference()
            fea_assembler.bdfInfo.is_xrefed = True

        # get the property info
        propertyID = kwargs["propID"]
        propInfo = fea_assembler.bdfInfo.properties[propertyID]

        thermal_materials = fea_assembler.bdfInfo.thermal_materials

        # compute the thickness by checking the dvprel has propID equal to the propID from the kwarg of the callback
        # this information is unavailable to a user creating their own element callback without an fea_assembler object
        t = None
        dv_name = None
        for dv_key in fea_assembler.bdfInfo.dvprels:
            propertyID = fea_assembler.bdfInfo.dvprels[dv_key].pid
            dv_obj = fea_assembler.bdfInfo.dvprels[dv_key].dvids_ref[0]
            dv_name = dv_obj.label.lower()

            if propertyID == kwargs["propID"]:
                # only grab thickness from specified DVs
                if dv_name in structDV_names:
                    t = structDV_dict[dv_name]

                    # exit for loop with current t, dv_name
                    break

        if t is not None:
            # get the DV ind from the currently set structDVs (if not all BDF/DAT file DVPRELs are used)
            for dv_ind, name in enumerate(structDV_names):
                if name.lower() == dv_name.lower():
                    break
        else:
            t = propInfo.t
            dv_ind = -1

        # Create a dictionary to sort all elements by property number
        elemDict = {}
        for elementID in fea_assembler.bdfInfo.elements:
            element = fea_assembler.bdfInfo.elements[elementID]
            propertyID = element.pid
            if propertyID not in elemDict:
                elemDict[propertyID] = {}
                elemDict[propertyID]["elements"] = []
                elemDict[propertyID]["dvs"] = {}
            elemDict[propertyID]["elements"].append(element)

        # Callback function to return appropriate tacs MaterialProperties object
        # For a pynastran mat card
        def matCallBack(matInfo):
            # Nastran isotropic material card
            if matInfo.type == "MAT1":
                mid = matInfo.mid
                specific_heat = 921  # default
                kappa = 230  # default
                if (mid + 100) in thermal_materials:
                    thermal_mat = thermal_materials[mid + 100]
                    if thermal_mat.type == "MAT4":
                        kappa = thermal_mat.k
                        specific_heat = thermal_mat.cp

                mat = constitutive.MaterialProperties(
                    rho=matInfo.rho,
                    E=matInfo.e,
                    nu=matInfo.nu,
                    ys=matInfo.St,
                    alpha=matInfo.a,
                    kappa=kappa,
                    specific_heat=specific_heat,
                )

            # TBD find associated MAT4 property, with MID+=100

            # Nastran orthotropic material card
            elif matInfo.type == "MAT8":
                mid = matInfo.mid
                if (mid + 100) in thermal_materials:  # orthotropic thermal
                    thermal_mat = thermal_materials[mid + 100]
                    if thermal_mat.type == "MAT5":
                        kappa1 = thermal_mat.kxx
                        kappa2 = thermal_mat.kyy
                        kappa3 = thermal_mat.kzz
                        specific_heat = thermal_mat.cp

                G12 = matInfo.g12
                G13 = matInfo.g1z
                G23 = matInfo.g2z
                A1 = matInfo.a1
                A2 = matInfo.a2
                A3 = matInfo.a2
                # If out-of-plane shear values are 0, Nastran defaults them to the in-plane
                if G13 == 0.0:
                    G13 = G12
                if G23 == 0.0:
                    G23 = G12
                mat = constitutive.MaterialProperties(
                    rho=matInfo.rho,
                    E1=matInfo.e11,
                    E2=matInfo.e22,
                    nu12=matInfo.nu12,
                    G12=G12,
                    G13=G13,
                    G23=G23,
                    alpha1=A1,
                    alpha2=A2,
                    alpha3=A3,
                    Xt=matInfo.Xt,
                    Xc=matInfo.Xc,
                    Yt=matInfo.Yt,
                    Yc=matInfo.Yc,
                    S12=matInfo.S,
                    kappa1=kappa1,
                    kappa2=kappa2,
                    kappa3=kappa3,
                    specific_heat=specific_heat,
                )
                # Nastran 2D anisotropic material card
            elif matInfo.type == "MAT2":
                C11 = matInfo.G11
                C12 = matInfo.G12
                C22 = matInfo.G22
                C13 = matInfo.G13
                C23 = matInfo.G23
                C33 = matInfo.G33
                nu12 = C12 / C22
                nu21 = C12 / C11
                E1 = C11 * (1 - nu12 * nu21)
                E2 = C22 * (1 - nu12 * nu21)
                G12 = G13 = G23 = C33
                mat = constitutive.MaterialProperties(
                    rho=matInfo.rho,
                    E1=E1,
                    E2=E2,
                    nu12=nu12,
                )

            else:
                raise ValueError(
                    f"Unsupported material type '{matInfo.type}' for material number {matInfo.mid}."
                )

            return mat

        # First we define the material object
        mat = None

        # make either one or more material objects from the
        if hasattr(propInfo, "mid_ref"):
            matInfo = propInfo.mid_ref
            mat = matCallBack(matInfo)
        # This property references multiple materials (maybe a laminate)
        elif hasattr(propInfo, "mids_ref"):
            mat = []
            for matInfo in propInfo.mids_ref:
                mat.append(matCallBack(matInfo))

        if propInfo.type == "PSHELL":
            # make the shell constitutive object for that material, thickness, and dv_ind (for thickness DVs)
            con = constitutive.IsoShellConstitutive(mat, t=t, tNum=dv_ind)

        elif propInfo.type == "PCOMP":
            numPlies = propInfo.nplies
            plyThicknesses = []
            plyAngles = []
            plyMats = []

            if propInfo.lam == "SYM":
                plyIndices = list(range(numPlies // 2))
                plyIndices.extend(plyIndices[::-1])
            else:
                plyIndices = range(numPlies)

            for ply_i in plyIndices:
                plyThicknesses.append(propInfo.thicknesses[ply_i])
                plyMat = constitutive.OrthotropicPly(plyThicknesses[ply_i], mat[ply_i])
                plyMats.append(plyMat)
                plyAngles.append(np.deg2rad(propInfo.thetas[ply_i]))

            # Convert thickness/angles to appropriate numpy array
            plyThicknesses = np.array(plyThicknesses, dtype=TACS.dtype)
            plyAngles = np.array(plyAngles, dtype=TACS.dtype)

            # Get the total laminate thickness
            lamThickness = propInfo.Thickness()
            # Get the offset distance from the ref plane to the midplane
            tOffset = -(propInfo.z0 / lamThickness + 0.5)

            plyFractions = plyThicknesses / lamThickness
            # have to use SmearedCompositeShellConstitutive since the other constitutive class for Composites in TACS
            # is not differentiable
            con = constitutive.SmearedCompositeShellConstitutive(
                plyMats,
                lamThickness,
                plyAngles,
                plyFractions,
                t_offset=tOffset,
                thickness_dv_num=dv_ind,
            )

        # add elements to FEA (assumes all elements are thermal shells by default for aerothermoelastic analysis)
        elemList = []
        transform = None
        for elemDescript in elemDescripts:
            if elemDescript in ["CQUAD4", "CQUADR"]:
                if include_thermal:
                    elem = elements.Quad4ThermalShell(transform, con)
                else:
                    elem = elements.Quad4Shell(transform, con)
                elemList.append(elem)
            elif elemDescript in ["CTRIA3"]:
                if include_thermal:
                    elem = elements.Tri3ThermalShell(transform, con)
                else:
                    elem = elements.Tri3Shell(transform, con)
                elemList.append(elem)
            else:
                print("Uh oh, '%s' not recognized" % (elemDescript))
                elemList.append(elem)

        # Add scale for thickness dv
        scale = [1.0]
        return elemList, scale
        # end of element callback method

    return element_callback


def addLoadsFromBDF(fea_assembler):
    """
    get fixed structural loads for the assembler
    """
    # force vector on the assembler
    assembler = fea_assembler.assembler

    vpn = assembler.getVarsPerNode()
    meshLoader = fea_assembler.meshLoader

    # load vector
    Fvec = assembler.createVec()
    F_array = Fvec.getArray()
    nnodes = assembler.getNumOwnedNodes()

    # get the one loadID, assumes only one loadID
    has_loads = False
    for subCase in fea_assembler.bdfInfo.subcases.values():
        if "LOAD" in subCase:
            # Add loads to problem
            loadsID = subCase["LOAD"][0]
            loadSet, loadScale, _ = fea_assembler.bdfInfo.get_reduced_loads(loadsID)
            has_loads = True  # record that there were some loads

            # Loop through every load in set and add it to problem
            for loadInfo, scale in zip(loadSet, loadScale):
                # Add any point force or moment cards
                if loadInfo.type == "FORCE" or loadInfo.type == "MOMENT":
                    node_ref = loadInfo.node_ref
                    if node_ref is not None:
                        nodeIDs = node_ref.nid
                    else:
                        nodeIDs = [loadInfo.node]

                    loadArray = np.zeros(vpn)
                    if loadInfo.type == "FORCE" and vpn >= 3:
                        F = scale * loadInfo.scaled_vector
                        if loadInfo.cid_ref is not None:
                            loadArray[
                                :3
                            ] += loadInfo.cid_ref.transform_vector_to_global(F)
                        else:
                            loadArray[:3] += loadInfo.xyz * loadInfo.mag
                    elif loadInfo.type == "MOMENT" and vpn >= 6:
                        M = scale * loadInfo.scaled_vector
                        loadArray[3:6] += loadInfo.cid_ref.transform_vector_to_global(M)

                    # self._addLoadToNodes(FVec, nodeID, loadArray, nastranOrdering=True)
                    # Make sure the inputs are the correct shape
                    nodeIDs = np.atleast_1d(nodeIDs)
                    loadArray = np.atleast_2d(loadArray)

                    numNodes = len(nodeIDs)

                    # If the user only specified one force vector,
                    # we assume the force should be the same for each node
                    if loadArray.shape[0] == 1:
                        loadArray = np.repeat(loadArray, [numNodes], axis=0)
                    # If the dimensions still don't match, raise an error
                    elif loadArray.shape[0] != numNodes:
                        raise AssertionError(
                            "Number of forces must match number of nodes,"
                            " {} forces were specified for {} node IDs".format(
                                loadArray.shape[0], numNodes
                            )
                        )

                    if len(loadArray[0]) != vpn:
                        raise AssertionError(
                            "Length of force vector must match varsPerNode specified "
                            "for problem, which is {}, "
                            "but length of vector provided was {}".format(
                                vpn, len(loadArray[0])
                            )
                        )

                    # First find the cooresponding local node ID on each processor
                    localNodeIDs = meshLoader.getLocalNodeIDsFromGlobal(nodeIDs, True)

                    # Flag to make sure we find all user-specified nodes
                    nodeFound = np.zeros(numNodes, dtype=int)

                    F_array2 = F_array.reshape(nnodes, vpn)

                    # Loop through every node and if it's owned by this processor, add the load
                    for i, nodeID in enumerate(localNodeIDs):
                        # The node was found on this proc
                        if nodeID >= 0:
                            # Add contribution to global force array
                            F_array2[nodeID, :] += loadArray[i]
                            nodeFound[i] = 1

                # # Add any gravity loads, TODO : add inertial fixed loads later
                # elif loadInfo.type == "GRAV":
                #     inertiaVec = np.zeros(3, dtype=self.dtype)
                #     inertiaVec[:3] = scale * loadInfo.scale * loadInfo.N
                #     # Convert acceleration to global coordinate system
                #     inertiaVec = loadInfo.cid_ref.transform_vector_to_global(inertiaVec)
                #     self._addInertialLoad(auxElems, inertiaVec)

                # Add any pressure loads
                # Pressure load card specific to shell elements
                elif loadInfo.type == "PLOAD2":
                    elemIDs = loadInfo.eids
                    pressures = scale * loadInfo.pressure
                    # self._addPressureToElements(
                    #     auxElems, elemIDs, pressure, nastranOrdering=True
                    # )

                    # Make sure the inputs are the correct shape
                    elemIDs = np.atleast_1d(elemIDs)
                    pressures = np.atleast_1d(pressures)

                    numElems = len(elemIDs)

                    # If the user only specified one pressure,
                    # we assume the force should be the same for each element
                    if pressures.shape[0] == 1:
                        pressures = np.repeat(pressures, [numElems], axis=0)
                    # If the dimensions still don't match, raise an error
                    elif pressures.shape[0] != numElems:
                        raise AssertionError(
                            "Number of pressures must match number of elements,"
                            " {} pressures were specified for {} element IDs".format(
                                pressures.shape[0], numElems
                            )
                        )

                    # First find the coresponding local element ID on each processor
                    localElemIDs = meshLoader.getLocalElementIDsFromGlobal(
                        elemIDs, nastranOrdering=True
                    )

                    # Flag to make sure we find all user-specified elements
                    elemFound = np.zeros(numElems, dtype=int)

                    # Loop through every element and if it's owned by this processor, add the pressure
                    for i, elemID in enumerate(localElemIDs):
                        # The element was found on this proc
                        if elemID >= 0:
                            elemFound[i] = 1
                            # Get the pointer for the tacs element object for this element
                            elemObj = meshLoader.getElementObjectForElemID(
                                elemIDs[i], nastranOrdering=True
                            )
                            # Create appropriate pressure object for this element type
                            pressObj = elemObj.createElementPressure(0, pressures[i])

    # if it didn't find any loads just return None
    if not has_loads:
        Fvec = None
    return Fvec  # end of addLoadsFromBDF method
