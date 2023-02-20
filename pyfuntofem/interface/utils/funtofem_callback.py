__all__ = ["f2f_callback"]

from tacs import constitutive, elements

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

        # Callback function to return appropriate tacs MaterialProperties object
        # For a pynastran mat card
        def matCallBack(matInfo):
            # Nastran isotropic material card
            if matInfo.type == "MAT1":
                mat = constitutive.MaterialProperties(
                    rho=matInfo.rho,
                    E=matInfo.e,
                    nu=matInfo.nu,
                    ys=matInfo.St,
                    alpha=matInfo.a,
                )

            # Nastran orthotropic material card
            elif matInfo.type == "MAT8":
                G12 = matInfo.g12
                G13 = matInfo.g1z
                G23 = matInfo.g2z
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
                    Xt=matInfo.Xt,
                    Xc=matInfo.Xc,
                    Yt=matInfo.Yt,
                    Yc=matInfo.Yc,
                    S12=matInfo.S,
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

        # make the shell constitutive object for that material, thickness, and dv_ind (for thickness DVs)
        con = constitutive.IsoShellConstitutive(mat, t=t, tNum=dv_ind)

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
            else:
                print("Uh oh, '%s' not recognized" % (elemDescript))

        # Add scale for thickness dv
        scale = [1.0]
        return elemList, scale

    return element_callback
