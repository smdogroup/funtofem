
from __future__ import print_function

import argparse
from tacs import TACS, elements, constitutive
from enum import Enum
import numpy as np
from collections import OrderedDict

# Default location of input files
inp_dir = './inp/'
mesh_dir = './bdf/'

#inp_dir = './'
#mesh_dir = './'

class ShellStiffness:
    """
    Bean that contains information for Shell constitutive object
    """
    def __init__(self,
                 rho, E, nu, kcorr, ys,
                 thickness, tmin, tmax):
        self.rho       = rho
        self.E         = E
        self.nu        = nu
        self.kcorr     = kcorr
        self.ys        = ys
        self.thickness = thickness
        self.tmin      = tmin
        self.tmax      = tmax
        return

class SolidStiffness:
    """
    Bean that contains information for Solid constitutive object
    """
    def __init__(self, rho, E, nu):
        self.rho       = rho
        self.E         = E
        self.nu        = nu
        return

class TACSBodyType(Enum):
    """
    Enumeration for different body types in TACS
    """
    RIGID = 1
    FLEXIBLE = 2
    CONSTRAINT = 3
    SHELL = 4
    BEAM = 5
    SOLID = 6

class TACSBCType(Enum):
    """
    Enumeration for different BC options
    """
    NA = 1
    FACE = 2
    EDGE = 3

class TACSBody:
    """
    A body is composed of many TACSElements. Information such as
    connectivities, node locations, pointers to the end of
    connectivities are packed into this object.
    """
    def __init__(self,
                 id, btype, elems, eid, nodes, xpts, conn,
                 ptr, bcs, name="no name", dof=8,
                 isFixed=False, bcptr = None, bcvars = None,
                 forcing=None):
        """
        Body information is stored in this object
        """
        self.name    = name
        self.id      = id
        self.btype   = btype
        self.elems   = elems
        self.xpts    = xpts
        self.conn    = conn
        self.bcs     = bcs  # flat list
        self.edgeBC  = []   # triplets
        self.faceBC  = []   # triplets
        self.bctype  = []
        self.eids    = eid
        self.ptr     = ptr
        self.dof     = dof
        self.isFixed = isFixed # set this to 'True' for fixed wing bodies
        self.nelems  = len(self.elems)

        self.bcptr = bcptr
        self.bcvars = bcvars

        # Pointer array does not include the start index as it is
        # defaulted to a predetermined offset
        if self.nelems != len(self.ptr)-1:
            raise ValueError('Pointer array does include start index')

        t = {}
        for n in self.conn:
            t[n] = n
        self.nodes = np.array(t.keys(), dtype=np.intc)

        # Determine the type of boundary for elements associated with
        # this body
        self.processBoundaryNodes()

        self.nnodes = len(self.xpts)/3

        # Same size as state vector
        self.forces = np.zeros(self.dof*self.nnodes, dtype=TACS.dtype)
        self.disps = np.zeros(self.dof*self.nnodes, dtype=TACS.dtype)

        # Pointer to function containing the implementation of
        # forces (or moments) <-- G(X,t)
        self.forcing = None

        # Pointer to user supplied write behavior
        self.write = None

        return

    def processBoundaryNodes(self):
        """
        Loop through all the body elements and tag boundaries as
        appropriate
        """
        # The face/edge logic is not applicable for flexible bodies
        if self.btype == TACSBodyType.RIGID or self.btype == TACSBodyType.CONSTRAINT or self.btype == TACSBodyType.SOLID:
            return

        if self.bcs is None or len(self.bcs) == 0:
            '''
            The body does not have nodes at the marked boundary
            '''
            for eidx in range(self.nelems):
                self.bctype.append(TACSBCType.NA)
        else:
            '''
            The body has faces/edges along the boudary
            '''

            # Take care of bc and conn offsets as the body contains
            # lists in global numbering
            conn_offset = min(self.conn)
            ptr_offset = min(self.ptr)

            # Local numbering of connectivities, ptr, bc lists
            conn = [x - conn_offset for x in self.conn]
            ptr = [x - ptr_offset for x in self.ptr]
            bcs = [x - conn_offset for x in self.bcs]

            for eidx in range(self.nelems):

                # Find the start/end pointer to this element from ptr list
                start_idx = ptr[eidx]
                end_idx = ptr[eidx+1]

                # Extract the connectivities of this element
                econn = conn[start_idx:end_idx]

                # Find intesection with BC nodes
                common = self.intersect(econn, bcs)
                ncommon = len(common)

                if ncommon == 0:

                    self.bctype.append(TACSBCType.NA)

                elif ncommon == 3:

                    # Add the edge to the boundary list
                    edge = []
                    for e in econn:
                        if e in common:
                            edge.append(e + conn_offset)

                    self.edgeBC.append(edge)

                    # Set the boundary type of this element
                    self.bctype.append(TACSBCType.EDGE)

                elif ncommon == 9:

                    # Offset to global numbering
                    econn = [x+conn_offset for x in econn]

                    # Set the boundary type of this element
                    self.bctype.append(TACSBCType.FACE)

                    # We can make 4 beam elements along four faces
                    self.faceBC.append([econn[0], econn[1], econn[2]])
                    self.faceBC.append([econn[0], econn[3], econn[6]])
                    self.faceBC.append([econn[2], econn[5], econn[8]])
                    self.faceBC.append([econn[6], econn[7], econn[8]])

                else:
                    self.bctype.append(TACSBCType.NA)

                # The next start index will be the current end index
                start_idx = end_idx

        # Remove duplicate sublists
        map(list, OrderedDict.fromkeys(map(tuple, self.edgeBC)))
        map(list, OrderedDict.fromkeys(map(tuple, self.faceBC)))

        return

    def getBoundaryNodes(self, bctype):
        """
        This function returns a list of bc nodes as selected by the
        user. This information can be used for creating RigidLink
        elements. For creating FlexLink elements the body attributes
        edgeBC and faceBC are directly accessed.
        """
        # The face/edge logic is not applicable for flexible bodies
        if self.btype == TACSBodyType.RIGID or self.btype == TACSBodyType.CONSTRAINT:
            return self.bcs
        else:
            if bctype == TACSBCType.FACE:
                return self.bcs
            else:
                # flatten the Edge BCS
                bcs = [item for sub_bcs in self.edgeBC for item in sub_bcs]

                # remove duplicates and return. Fix the ordering is now
                ## jumbled (adjacent nodes can be further apart in the
                ## list)
                return list(set(bcs))

    def intersect(self, a, b):
        return list(set(a) & set(b))

    def getNodes(self):
        """
        Returns the structural nodes
        """
        return np.array(self.xpts)

    def getDisplacements(self):
        """
        Returns the structural displacements at the current timestep
        """
        return np.array(self.disps)

    def setLoads(self,struct_loads):
        """
        Sets the aerodynamic loads for the current time step
        """
        self.forces = struct_loads
        return

    def toString(self):
        print("name      ", self.name)
        print("id        ", self.id)
        print("type      ", self.btype)
        print("eids      ", self.eids)
        print("elem      ", self.elems)
        print("xpts      ", self.xpts)
        print("conn      ", self.conn)
        print("ptr       ", self.ptr)
        print("bc        ", self.bcs)
        print("edgeBC    ", self.edgeBC)
        print("faceBC    ", self.faceBC)
        print("bctype    ", self.bctype)
        print("forces    ", self.forces)
        print("disps     ", self.disps)
        print("nodes     ", self.nodes)
        print("fixed     ", self.isFixed)
        print("forcing   ", self.forcing)
        print("output    ", self.write)
        print("")
        return

class TACSBuilder:
    """
    Class that returns an instantiated dynamic body
    correspoding to the inputs. This class is targeted to handle the
    bookkeeping related to setting xpts, conn and bcs
    """
    def __init__(self, comm):
        """
        Constructor for a component
        """
        self.comm = comm
        self.ehelper = ElementHelper(self.comm)

        self.next_comp_id = 0
        self.next_elem_id = 0
        self.next_node_id = 0

        self.current_ptr = 0
        self.max_conn = 0

        # List of containing the bodies
        self.body_list = []

        self.dtype = TACS.dtype

        # Track necessary visualizations, rigid, flex
        self.rigid_viz = 0
        self.shell_viz = 0
        self.beam_viz = 0
        self.solid_viz = 0

        # Track number of available dofs
        self.ndof = 0

        return

    def createTACS(self, comm, xpts, conn, ptr, elems, bcs,
                   ordering=TACS.PY_AMD_ORDER,
                   mattype=TACS.PY_DIRECT_SCHUR,
                   print_details=0, ndof=8, bcptr=None, bcvars=None):
        """
        Creates and returns an instance of TACS.
        """
        # Create numpy equivalents for TACS instantiation
        conn = np.array(conn, dtype=np.intc)
        ptr  = np.array(ptr, dtype=np.intc)
        xpts = np.array(xpts)
        bcs  = np.array(bcs, dtype=np.intc)
        if bcptr is not None and bcvars is not None:
            bcptr = np.array(bcptr,dtype=np.intc)
            bcvars = np.array(bcvars,dtype=np.intc)

        # Figure out lengths
        npts   = len(xpts)/3
        nelems = len(elems)

        # Sanity check of data we have
        if max(conn)+1 != npts:
            raise ValueError('Connectivity and number of nodes are inconsistent')
        if nelems != ptr.shape[0]-1:
            raise ValueError('Number of elements and connectivity are inconsistent')

        # Create TACS
        vars_per_node = ndof

        ## Does not work in parallel
        ## tacs = TACS.Assembler.create(comm, vars_per_node, npts, nelems)
        ## tacs.setElementConnectivity(conn, ptr)
        ## tacs.setElements(elems)

        ## # Set BCs
        ## if len(init_bcs) != 0:
        ##     tacs.addInitBCs(init_bcs)

        ## # Reorder the variables using RCM
        ## tacs.computeReordering(ordering, mattype)

        ## # Finish the initialization
        ## tacs.initialize()

        ## # Get the nodes in default ordering
        ## Xpts = tacs.createNodeVec()
        ## X = Xpts.getArray()
        ## X[:] = xpts[:]

        ## # Set the reordered node locations into TACS
        ## tacs.reorderVec(Xpts)
        ## tacs.setNodes(Xpts)

        ## Works in parallel
        creator = TACS.Creator(comm, vars_per_node)
        creator.setReorderingType(ordering, mattype)
        if comm.Get_rank() == 0:
            ids = np.arange(0, nelems, dtype=np.intc)
            creator.setGlobalConnectivity(npts, ptr, conn, ids)
            creator.setNodes(xpts)
            creator.setBoundaryConditions(bcs, bcptr, bcvars)
        creator.setElements(elems)

        # Create TACS using the TACSCreator class
        tacs = creator.createTACS()

        # Get the distributed list of node numbers
        for body in self.body_list:
            body.dist_nodes = creator.getTacsNodeNums(tacs, body.nodes)
            # convert to a local ordering
            rnge = tacs.getOwnerRange()
            body.dist_nodes -= np.min(rnge[comm.Get_rank()])

        if comm.Get_rank() == 0 and print_details > 0:
            print("ordering ", ordering)
            print("mattype  ", mattype)
            print("xpts     ", xpts)
            print("conn     ", conn)
            print("ptr      ", ptr)
            print("elem     ", elems)
            print("bcs      ", bcs)

        # return the tacs instance
        return tacs

    @staticmethod
    def createTACSFromBDF(comm,
                          bdffile, convert_mesh,
                          v, w, g,
                          rho, E, nu, kcorr, ys,
                          thickness, tmin, tmax,
                          ordering=TACS.PY_AMD_ORDER,
                          mattype=TACS.PY_DIRECT_SCHUR,
                          ndof=8):
        """
        Creates and returns an instance of TACS for MITC9 elements.
        """
        velocity = elements.GibbsVector(v[0],v[1],v[2])
        omega    = elements.GibbsVector(w[0],w[1],w[2])
        gravity  = elements.GibbsVector(g[0],g[1],g[2])

        mesh = TACS.MeshLoader(comm)
        mesh.setConvertToCoordinate(convert_mesh)
        mesh.scanBDFFile(mesh_dir+bdffile)

        num_components = mesh.getNumComponents()
        for i in range(num_components):
            descriptor = mesh.getElementDescript(i)
            stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys,
                                         thickness, i, tmin, tmax)
            element = None
            if descriptor in ["CQUAD", "CQUAD9"]:
                element = elements.MITC(stiff,
                                        gravity,
                                        velocity,
                                        omega)
                mesh.setElement(i, element)

        # Create TACS from mesh loader
        return mesh.createTACS(ndof, ordering, mattype)

    def skew(self,x):
        """
        Return a skew symmetric matrix for cross product
        """
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, x[0]],
                         [x[1], x[0], 0]],
                         self.dtype)

    def isPositiveDefinite(self, x):
        # Populate the full matrix
        a11 = x[0]
        a12 = x[1]
        a13 = x[2]

        a21 = a12
        a22 = x[3]
        a23 = x[4]

        a31 = a13
        a32 = a23
        a33 = x[5]

        A = np.array([a11, a12, a13, a21, a22, a23, a31, a32, a33])

        return np.all(np.linalg.eigvals(A.reshape(3,3)) > 0)

    def upperTriangle(self,a):
        """
        convert a 6 entried array in that stores a lower triangular
        matrix's symmetric elements in row major order to an upper
        triangular matrix in row major order.
        """
        return np.array([a[0], a[1], a[3], a[2], a[4], a[5]])

    def getElementID(self):
        """
        Generate next available element ID
        """
        # Generate element id number
        eid = self.next_elem_id

        # Offset the next element by 1
        self.next_elem_id = self.next_elem_id + 1

        return eid

    def getBodyID(self):
        """
        Generate next available Body ID
        """
        # Generate Body id number
        id = self.next_comp_id

        # Offset the next Body by 1
        self.next_comp_id = self.next_comp_id + 1

        return id

    def getNodePointer(self, num_nodes):
        # Offset the pointer based on the number of nodes for this elem
        self.current_ptr = self.current_ptr + num_nodes

        return self.current_ptr

    def setNodePointer(self, num_nodes):
        # Offset the pointer based on the number of nodes for this elem
        self.current_ptr = self.current_ptr + num_nodes

        return

    def getNodeID(self):
        """
        Generate next available node ID based on connectivities. Use
        this to generate node numbers for contraints.
        """
        # Generate node id number
        nid = self.max_conn

        self.max_conn = self.max_conn + 1

        return nid

    def getTACS(self,
                ordering=TACS.PY_AMD_ORDER,
                mattype=TACS.PY_DIRECT_SCHUR,
                print_details=0,ndof=8):
        """
        Order of bodies in the list matters.

        # TODO: generate this list internally to avoid hassle
        """

        self.body_list = remove_duplicates(self.body_list)

        #Create data necessary for TACS creation
        xpts     = []
        conn     = []
        elems    = []
        ptr      = []
        bcs      = []
        nodes    = []

        bcptr    = []
        bcvars   = []
        # Initialize pointer with zero
        ptr.append([0])

        # Loop through all bodies and add CONN, PTR, XPTS, BC
        # information
        for body in self.body_list:
            if print_details != 0:
                body.toString()

            # Append the node locations
            xpts.append(body.xpts[:])

            # Append the connectivities of nodes
            conn.append(body.conn[:])

            # Append the elements
            elems.append([body.elems])

            # Append the pointers into connectivities skipping the
            # start node pointer
            ptr.append(body.ptr[1:])

            # append the nodes
            nodes.append(body.nodes[:])

            # Append only if it is a DRIVER element (identified based
            # on None)
            #if body.bcs is not None and body.btype is TACSBodyType.CONSTRAINT:
            #    bcs.append(body.bcs[:])

            # If this is a fixed body, apply the bcs
            if body.isFixed is True:
                bcs.append(body.bcs[:])

            if body.bcptr is not None and body.bcvars is not None:
                bcptr.extend(body.bcptr.tolist())
                bcvars.extend(body.bcvars.tolist())

        # Create pointers by unpacking connectivities
        tptr = np.zeros(len(conn)+1, dtype=np.intc)
        for i in range(len(conn)):
            tptr[i+1] = tptr[i] + len(conn[i])

        # Compare the two pointer arrays (ensures the consistency of
        # order of bodies with ptr array)
        for entry in zip(ptr, tptr):
            assert(entry[0][-1] == entry[1])

        # Flatten the connectivity and points array
        conn = [item for sublist in conn for item in sublist]
        xpts = [item for x in xpts for item in x]
        ptr  = [item for p in ptr for item in p]
        elems  = [item for e in elems for item in e[0]]
        if len(bcs) != 0:
            bcs = [item for bc in bcs for item in bc]

        # bcptr should not be an empty list rather None (See initialization)
        if len(bcptr) == 0:
            bcptr = None

        # bcvars should not be an empty list rather None (See initialization)
        if len(bcvars) == 0:
            bcvars = None

        # Create and return TACS
        tacs = self.createTACS(self.comm,
                               xpts, conn, ptr, elems,
                               bcs, ordering, mattype,
                               print_details,ndof=ndof,bcptr=bcptr,bcvars=bcvars)
        return tacs

    def addMITCShellBody(self,
                         name, bdffile, bdf_type_flag,
                         shellStiff, isFixed=True):
        """
        Create MITCShell elements for bodies loaded from BDF file
        """
        # Flag that shell element visualization is needed
        self.shell_viz = 1

        # Initialize data
        xpts  = []
        conn  = []
        ptr   = []
        elems = []
        eids  = []
        nodes = []

        # Generate a body ID
        id = self.getBodyID()

        # Load mesh from BDF file
        mesh = TACS.MeshLoader(self.comm)
        # This is necessary if the BDF file is created using gmsh
        mesh.setConvertToCoordinate(bdf_type_flag)
        mesh.scanBDFFile(mesh_dir+bdffile)

        [_ptr, _conn, _cnums, xpts] = mesh.getConnectivity()
        [_bcs, _bcptr, bcvars,  _] = mesh.getBCs()

        # remove duplicate bcs
        #_bcs = np.unique(_bcs)
        _ptr = self.comm.bcast(_ptr, root=0)
        _conn = self.comm.bcast(_conn, root=0)
        cnums = self.comm.bcast(_cnums, root=0)
        xpts = self.comm.bcast(xpts, root=0)
        _bcs = self.comm.bcast(_bcs, root=0)
        _bcptr = self.comm.bcast(_bcptr, root=0)
        bcvars = self.comm.bcast(bcvars, root=0)

        npts = len(xpts)/3
        assert(max(_conn)+1 == npts)
        nelems = _ptr.shape[0]-1
        _conn = np.array(_conn, dtype=np.intc)

        # Shift the ptr array
        ptr = _ptr + self.current_ptr
        bcptr = _bcptr + self.current_ptr

        # Shift the conn array
        conn = _conn + self.max_conn

        # Renumber the bc bodies
        bcs = _bcs + self.max_conn

        # Find the new max in conn
        self.max_conn = max(conn) + 1

        rho       = shellStiff.rho
        E         = shellStiff.E
        nu        = shellStiff.nu
        kcorr     = shellStiff.kcorr
        ys        = shellStiff.ys
        tmin      = shellStiff.tmin
        thickness = shellStiff.thickness
        tmax      = shellStiff.tmax

        # Load the BDF file, get a list of component number to which
        # each element belongs. The length of this array should equal
        # the number of MITC9 elements
        # cnums = getElemCompNumbersFromBDF(mesh_dir+bdffile)
        ncomponents = np.max(cnums) + 1

        # Create as many elements and constitutive objects as the
        # number of components
        shell = []
        for c in range(ncomponents):
            # Create stiffness object
            stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys,
                                         thickness, c,
                                         tmin, tmax)

            # Create the element
            shell_elem = self.ehelper.createMITCShellElement(stiff, c)
            shell.append(shell_elem)

        for i in range(nelems):
            # Set the component ID to which the element belongs
            shell[cnums[i]].setComponentNum(id)

            # Append the element to list
            elems.append(shell[cnums[i]])

            # Generate element id number for each element
            eids.append(self.getElementID())

            # Offset the pointer
            self.setNodePointer(shell_elem.numNodes())

        # Return the flexible body object
        scomp = TACSBody(id, TACSBodyType.FLEXIBLE, elems, eids,
                         nodes, xpts, conn, ptr, bcs, name, isFixed=isFixed, bcptr=bcptr, bcvars=bcvars)
        scomp.dof = 6
        self.body_list.append(scomp)

        return scomp

    def addMITC9ShellBody(self,
                          name, bdffile, gmsh,
                          vInit, wInit, gravity,
                          shellStiff,isFixed=False):
        """
        Create MITC9 elements for bodies loaded from BDF file
        """
        # Flag that shell element visualization is needed
        self.shell_viz = 1

        # Initialize data
        xpts  = []
        conn  = []
        ptr   = []
        elems = []
        eids  = []
        nodes = []

        # Generate a body ID
        id = self.getBodyID()

        # Load mesh from BDF file
        mesh = TACS.MeshLoader(self.comm)
        mesh.setConvertToCoordinate(gmsh)
        mesh.scanBDFFile(mesh_dir+bdffile)

        [_ptr, _conn, _cnums, xpts] = mesh.getConnectivity()
        [_bcs, _, _,  _] = mesh.getBCs()

        # remove duplicate bcs
        _bcs = np.unique(_bcs)

        _ptr = self.comm.bcast(_ptr, root=0)
        _conn = self.comm.bcast(_conn, root=0)
        cnums = self.comm.bcast(_cnums, root=0)
        xpts = self.comm.bcast(xpts, root=0)
        _bcs = self.comm.bcast(_bcs, root=0)

        npts = len(xpts)/3
        # If this assertion fails check whether the elements (CQUADX)
        # exist in the bdf file
        assert(max(_conn)+1 == npts)

        nelems = _ptr.shape[0]-1
        _conn = np.array(_conn, dtype=np.intc)

        # Shift the ptr array
        ptr = _ptr + self.current_ptr

        # Shift the conn array
        conn = _conn + self.max_conn

        # Renumber the bc bodies
        bcs = _bcs + self.max_conn

        # Find the new max in conn
        self.max_conn = max(conn) + 1

        rho       = shellStiff.rho
        E         = shellStiff.E
        nu        = shellStiff.nu
        kcorr     = shellStiff.kcorr
        ys        = shellStiff.ys
        tmin      = shellStiff.tmin
        thickness = shellStiff.thickness
        tmax      = shellStiff.tmax

        # Load the BDF file, get a list of component number to which
        # each element belongs. The length of this array should equal
        # the number of MITC9 elements
        #cnums = getElemCompNumbersFromBDF(mesh_dir+bdffile)
        ncomponents = np.max(cnums) + 1

        # Create as many elements and constitutive objects as the
        # number of components
        shell = []
        for c in range(ncomponents):
            # Create stiffness object
            stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys,
                                         thickness, c,
                                         tmin, tmax)

            # Create the element
            shell_elem = self.ehelper.createMITC9ShellElement(vInit,
                                                              wInit,
                                                              gravity,
                                                              stiff)
            shell.append(shell_elem)

        for i in range(nelems):
            # Set the component ID to which the element belongs
            shell[cnums[i]].setComponentNum(id)

            # Append the element to list
            elems.append(shell[cnums[i]])

            # Generate element id number for each element
            eids.append(self.getElementID())

            # Offset the pointer
            self.setNodePointer(shell_elem.numNodes())

        # Return the flexible body object
        scomp = TACSBody(id, TACSBodyType.FLEXIBLE, elems, eids,
                         nodes, xpts, conn, ptr, bcs, name, isFixed=isFixed)
        self.body_list.append(scomp)

        return scomp

    def addSolidBody(self,
                     name, bdffile, gmsh,
                     solidStiff, order=2, elem_type=elements.PY_LINEAR,isFixed=True):
        """
        Create Solid elements for bodies loaded from BDF file
        """
        # Flag that rigid body visualization is needed
        self.solid_viz = 1

        # Initialize data
        xpts  = []
        conn  = []
        ptr   = []
        elems = []
        eids  = []
        nodes = []

        # Generate a body ID
        id = self.getBodyID()

        # Load mesh from BDF file
        mesh = TACS.MeshLoader(self.comm)
        mesh.setConvertToCoordinate(gmsh)
        mesh.scanBDFFile(mesh_dir+bdffile)

        [_ptr, _conn, _cnums, xpts] = mesh.getConnectivity()
        [_bcs, _bcptr, bcvars,  _] = mesh.getBCs()

        # remove duplicate bcs
        #_bcs = np.unique(_bcs)

        _ptr = self.comm.bcast(_ptr, root=0)
        _conn = self.comm.bcast(_conn, root=0)
        cnums = self.comm.bcast(_cnums, root=0)
        xpts = self.comm.bcast(xpts, root=0)
        _bcs = self.comm.bcast(_bcs, root=0)
        _bcptr = self.comm.bcast(_bcptr, root=0)
        bcvars = self.comm.bcast(bcvars, root=0)

        npts = len(xpts)/3
        assert(max(_conn)+1 == npts)
        nelems = _ptr.shape[0]-1
        _conn = np.array(_conn, dtype=np.intc)

        # Shift the ptr array
        ptr = _ptr + self.current_ptr
        bcptr = _bcptr + self.current_ptr

        # Shift the conn array
        conn = _conn + self.max_conn

        # Renumber the bc bodies
        bcs = _bcs + self.max_conn

        # Find the new max in conn
        self.max_conn = max(conn) + 1

        rho       = solidStiff.rho
        E         = solidStiff.E
        nu        = solidStiff.nu

        # Load the BDF file, get a list of component number to which
        # each element belongs. The length of this array should equal
        # the number of Solid elements
        #cnums = getElemCompNumbersFromBDF(mesh_dir+bdffile)
        ncomponents = np.max(cnums) + 1

        # Create as many elements and constitutive objects as the
        # number of components
        solid = []
        for c in range(ncomponents):
            # Create stiffness object
            stiff = constitutive.isoSolidStiff(rho, E, nu)

            # Create the element
            solid_elem = self.ehelper.createSolidElement(order,
                                                         stiff,
                                                         elem_type,
                                                         c)
            solid.append(solid_elem)

        for i in range(nelems):
            # Set the component ID to which the element belongs
            solid[cnums[i]].setComponentNum(id)

            # Append the element to list
            elems.append(solid[cnums[i]])

            # Generate element id number for each element
            eids.append(self.getElementID())

            # Offset the pointer
            self.setNodePointer(solid_elem.numNodes())

        # Return the flexible body object
        scomp = TACSBody(id, TACSBodyType.SOLID, elems, eids,
                         nodes, xpts, conn, ptr, bcs, name,
                         isFixed=isFixed, bcptr=bcptr, bcvars=bcvars)
        scomp.dof = 3
        self.body_list.append(scomp)

        return scomp

    def addBeamBody(self,
                    name, cbarfile,
                    vInit, wInit, gravity,
                    beamStiff):
        """
        Models the body meshed in cbarfile as Timoshenko beam elements
        """
        # Flag that shell element visualization is needed
        self.beam_viz = 1

        # Generate a body ID for beam body
        id = self.getBodyID()

        # Load the mesh from file
        cbar = CBARLoader(mesh_dir+cbarfile, beamStiff)

        # Obtain data from mesh as list
        xpts = cbar.xpts
        conn = cbar.conn
        bcs = cbar.bcs
        nelems = cbar.nelems

        # Initialize the rest
        eids = []
        elems = []
        ptr = [self.current_ptr]
        nodes = []

        # Get connectivities from cbar file
        conn = np.array(conn, dtype=np.intc)
        bcs = np.array(bcs, dtype=np.intc)

        # Shift the conn and bc array for local indexing
        conn = conn + self.max_conn
        bcs = bcs + self.max_conn

        # Find the new max in conn
        self.max_conn = max(conn) + 1

        if (isinstance(beamStiff, constitutive.Timoshenko)):
            cons = [beamStiff]*nelems
        else:
            cons = cbar.constitutive

        for i in range(nelems):
            # Create a beam element
            beam_element = self.ehelper.createBeamElement(vInit,
                                                          wInit,
                                                          gravity,
                                                          cons[i])

            # Set the body ID to which the element belongs
            beam_element.setComponentNum(id)

            # Append the element to list
            elems.append(beam_element)

            # Generate element id number for each element
            eids.append(self.getElementID())

            # Create pointer into start index of next element
            ptr.append(self.getNodePointer(beam_element.numNodes()))

        # Return the flexible body object
        comp = TACSBody(id, TACSBodyType.FLEXIBLE, elems, eids,
                        nodes, xpts, conn, ptr, bcs, name)

        # Add the body to body list
        self.body_list.append(comp)

        return comp

    def addRigidBody(self, name, mass, c, J, rInit, vInit, wInit,
                     bdffile=None, gmsh=1,
                     grav=None,
                     voffset=None,
                     basepoint=None, xpoint=None, ypoint=None):
        # Flag that rigid body visualization is needed
        self.rigid_viz = 1

        # Increase the number of degrees of Freedom
        self.ndof = self.ndof + 6

        # Initialize data
        xpts  = []
        conn  = []
        ptr   = [self.current_ptr]
        elems = []
        eids  = []
        nodes = []

        # Generate a body ID
        id = self.getBodyID()

        # Create the element
        elem = self.ehelper.createRigidBody(mass, c, J,
                                            rInit, vInit, wInit,
                                            bdffile, gmsh,
                                            grav,
                                            voffset,
                                            basepoint, xpoint, ypoint)

        # Set the body ID to which the element belongs
        elem.setComponentNum(id)

        # Add the element to a list
        elems.append(elem)

        # Generate an element ID
        eids.append(self.getElementID())

        # nodal locations
        xpts.append(rInit[0])
        xpts.append(rInit[1])
        xpts.append(rInit[2])

        node = self.getNodeID()

        # Node numbers
        nodes.append(node)

        # Connectivities
        conn.append(node)

        # Create pointer into starting of next body
        ptr.append(self.getNodePointer(elem.numNodes()))

        # return the body object
        comp = TACSBody(id, TACSBodyType.RIGID, elems, eids,
                        nodes, xpts, conn, ptr, conn, name)

        # Add the body to list
        self.body_list.append(comp)

        return comp

    def addRigidLink(self, rigidCompA, bodyB, type=TACSBCType.FACE):
        """
        A rigid link has to be between (1) a rigid body and another
        rigid body OR (2) between a rigid body and a flexible
        body.

        The key is to create the link by passing the rigid body as the
        argument into the constructor for RigidLink and associate the
        other body using connectivities.

        In case (1), there will be just one rigid link, whereas in
        case (2) there will be num(SPC) rigid links between the rigid
        body and flexible boundary nodes. The collection of rigid
        links are treated as a body within this framework.

        The optional BCType allows to construct links along all the
        edge nodes or all the face nodes.
        """
        # The two supplied bodies must be different
        assert(rigidCompA != bodyB)
        assert(len(bodyB.bcs) != 0)

        # Reduce 6 dof
        self.ndof = self.ndof - 6

        # The first body must be rigid
        assert(rigidCompA.btype == TACSBodyType.RIGID)

        # Generate a body ID
        id = self.getBodyID()

        # Create the rigid link
        link = self.ehelper.createRigidLink(rigidCompA.elems[0])

        # Set the body ID to which the element belongs
        link.setComponentNum(id)

        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [self.current_ptr]
        elems = []
        eids  = []
        elems = []
        nodes = []

        # Get the node at the boundary of body
        bcs = bodyB.getBoundaryNodes(type)

        # Iterate through the boundary nodes and create link and
        # associated data
        for bc in bcs:
            # Append the element to list
            elems.append(link)

            # Generate element id number for each element
            eids.append(self.getElementID())

            # Dummy link locations
            xpts.append(0.0)
            xpts.append(0.0)
            xpts.append(0.0)

            # Create a node number for this element and append to list
            node = self.getNodeID()
            nodes.append(node)

            # Append connectivities (links are three noded)
            conn.append(rigidCompA.conn[0])
            conn.append(bc)
            conn.append(node)

            # Create pointer into starting of next body
            ptr.append(self.getNodePointer(link.numNodes()))

        # return the body object
        name =  "rigid link between %s and %s" % (rigidCompA.name, bodyB.name)
        comp =  TACSBody(id, TACSBodyType.CONSTRAINT, elems, eids,
                         nodes, xpts, conn, ptr, None, name)
        self.body_list.append(comp)

        return comp

    def addFlexLink(self,
                    rigidBody, flexBody,
                    location, frame, moments_flag,
                    type=TACSBCType.EDGE):
        """
        Adds a flexible attachment between a rigidbody and flexible
        body. This attachment enforces an averaged displacement at the
        attachment surface as opposed to a rigid link that absolutely
        restricts all the nodes along the marked boundary surface.
        """
        # Sanity checks
        assert(rigidBody       != flexBody)
        assert(rigidBody.btype == TACSBodyType.RIGID)
        assert(flexBody.btype  == TACSBodyType.FLEXIBLE)

        # Create the flexible link element
        flink = self.ehelper.createFlexLink(rigidBody.elems[0],
                                            location, frame,
                                            moments_flag)

        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [self.current_ptr]
        elems = []
        eids  = []
        nodes = []

        # One nodal location for all constraints as the same lambda is
        # used for all
        xpts.append(0.0)
        xpts.append(0.0)
        xpts.append(0.0)

        # One ID all constraints
        conid = self.getNodeID()
        nodes.append(conid)
        flink.setComponentNum(conid)

        # Obtain the boundary nodes from the flexible body to which
        # the flexible links are to be attached
        if type == TACSBCType.EDGE:
            bc_list = flexBody.edgeBC
        elif type == TACSBCType.EDGE:
            bc_list = flexBody.faceBC
        else:
            raise ValueError("unsupported connection type for flexible lnk")

        # For each triplet of bc, associate a flexible link element
        for bc in bc_list:
            # Append the element to list
            elems.append(flink)

            # Generate element id number for each element
            eids.append(self.getElementID())

            # Append connectivities (flex links are 5 noded)
            conn.append(rigidBody.conn[0])   # rigid attachment node
            conn.append(bc[0]) # beam node 1
            conn.append(bc[1]) # beam node 2
            conn.append(bc[2]) # beam node 3
            conn.append(conid) # flex attachment node

            # Create pointer into starting of next component
            ptr.append(self.getNodePointer(flink.numNodes()))

        # Return the body object
        name = "flexible link between %s and %s" % (rigidBody.name,
                                                    flexBody.name)
        comp = TACSBody(conid, TACSBodyType.CONSTRAINT,
                        elems, eids, nodes, xpts, conn, ptr,
                        None, name, 8, False)
        self.body_list.append(comp)

        return comp

    def addRevoluteConstraint(self, location, revaxis,
                              bodyA, bodyB=None):
        '''
        Add a revolute contraint body in between two rigid
        bodies. This can not be generated between rigid and
        flexible bodies. If there is a flexible body
        involved, create a rigid body adjoining the flexible
        body and then use the newly created rigid body to
        construct this revolute constaint.
        '''
        # Make sure the hinge axis is of unit length
        assert(np.allclose(np.linalg.norm(revaxis), 1.0)==1)

        # Reduce dof of bodies
        self.ndof = self.ndof - 5

        # Check element types first
        assert(bodyA.btype == TACSBodyType.RIGID)
        if bodyB is not None:
            assert(bodyB.btype == TACSBodyType.RIGID)

        # Generate a body ID
        id = self.getBodyID()

        # Create the revolute constraint
        if bodyB is None:
            revcon = self.ehelper.createRevoluteConstraint(location,
                                                           revaxis,
                                                           bodyA.elems[0])
            name =  "revolute constraint between %s and point" % (bodyA.name)

        else:
            revcon = self.ehelper.createRevoluteConstraint(location,
                                                           revaxis,
                                                           bodyA.elems[0],
                                                           bodyB.elems[0])
            name =  "revolute constraint between %s and %s" % (bodyA.name,
                                                               bodyB.name)

        # Set the body ID to which the element belongs
        revcon.setComponentNum(id)

        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [ self.current_ptr ]
        elems = []
        eids  = []
        nodes = []

        # Append the element to list
        elems.append(revcon)

        # Generate element id number for each element
        eids.append(self.getElementID())

        # Dummy link locations
        xpts.append(location[0])
        xpts.append(location[1])
        xpts.append(location[2])

        # Create a node number for this element and append to list
        node = self.getNodeID()
        nodes.append(node)

        # Append connectivities (revcons are two/three noded)
        conn.append(bodyA.conn[0])
        if bodyB is not None:
            conn.append(bodyB.conn[0])
        conn.append(node)

        # Create pointer into starting of next body
        ptr.append(self.getNodePointer(revcon.numNodes()))

        # Return the body object
        comp = TACSBody(id, TACSBodyType.CONSTRAINT, elems,
                        eids, nodes,
                        xpts, conn, ptr, None, name)
        self.body_list.append(comp)

        return comp

    def addCylindricalConstraint(self, location, revaxis,
                                 bodyA, bodyB=None):
        '''
        Add a cylindrical contraint body in between two rigid
        bodies. This can not be generated between rigid and
        flexible bodies. If there is a flexible body
        involved, create a rigid body adjoining the flexible
        body and then use the newly created rigid body to
        construct this cylindrical constaint.
        '''
        # Make sure the hinge axis is of unit length
        assert(np.linalg.norm(revaxis) == 1.0)

        # Reduce dof of bodies
        self.ndof = self.ndof - 4

        # Check element types first
        assert(bodyA.btype == TACSBodyType.RIGID)
        if bodyB is not None:
            assert(bodyB.btype == TACSBodyType.RIGID)

        # Generate a body ID
        id = self.getBodyID()

        # Create the cylindrical constraint
        if bodyB is None:
            revcon = self.ehelper.createCylindricalConstraint(location,
                                                              revaxis,
                                                              bodyA.elems[0])
            name = "cylindrical constraint between %s and point"%(bodyA.name)

        else:
            revcon = self.ehelper.createCylindricalConstraint(location,
                                                              revaxis,
                                                              bodyA.elems[0],
                                                              bodyB.elems[0])
            name = "cylindrical constraint between %s and %s"%(bodyA.name,
                                                               bodyB.name)

        # Set the body ID to which the element belongs
        revcon.setComponentNum(id)

        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [ self.current_ptr ]
        elems = []
        eids  = []
        elems = []
        nodes = []

        # Append the element to list
        elems.append(revcon)

        # Generate element id number for each element
        eids.append(self.getElementID())

        # Dummy link locations
        xpts.append(location[0])
        xpts.append(location[1])
        xpts.append(location[2])

        # Create a node number for this element and append to list
        node = self.getNodeID()
        nodes.append(node)

        # Append connectivities (revcons are two/three noded)
        conn.append(bodyA.conn[0])
        if bodyB is not None:
            conn.append(bodyB.conn[0])
        conn.append(node)

        # Create pointer into starting of next body
        ptr.append(self.getNodePointer(revcon.numNodes()))

        # Return the body object
        comp = TACSBody(id, TACSBodyType.CONSTRAINT, elems,
                        eids, nodes, xpts, conn, ptr, None, name)
        self.body_list.append(comp)

        return comp

    def addPrismaticConstraint(self, location, revaxis,
                               bodyA, bodyB=None):
        '''
        Prismatic constraint between two bodies
        '''
        # Make sure the hinge axis is of unit length
        assert(np.linalg.norm(revaxis) == 1.0)

        # Reduce dof of bodies
        self.ndof = self.ndof - 5

        # Check element types first
        assert(bodyA.btype == TACSBodyType.RIGID)
        if bodyB is not None:
            assert(bodyB.btype == TACSBodyType.RIGID)

        # Generate a body ID
        id = self.getBodyID()

        # Create the cylindrical constraint
        if bodyB is None:
            revcon = self.ehelper.createPrismaticConstraint(location,
                                                            revaxis,
                                                            bodyA.elems[0])
            name =  "prismatic constraint between %s and point" % (bodyA.name)

        else:
            revcon = self.ehelper.createPrismaticConstraint(location,
                                                            revaxis,
                                                            bodyA.elems[0],
                                                            bodyB.elems[0])
            name =  "prismatic constraint between %s and %s" % (bodyA.name,
                                                                bodyB.name)

        # Set the body ID to which the element belongs
        revcon.setComponentNum(id)

        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [ self.current_ptr ]
        elems = []
        eids  = []
        elems = []
        nodes = []

        # Append the element to list
        elems.append(revcon)

        # Generate element id number for each element
        eids.append(self.getElementID())

        # Dummy link locations
        xpts.append(location[0])
        xpts.append(location[1])
        xpts.append(location[2])

        # Create a node number for this element and append to list
        node = self.getNodeID()
        nodes.append(node)

        # Append connectivities (revcons are two/three noded)
        conn.append(bodyA.conn[0])
        if bodyB is not None:
            conn.append(bodyB.conn[0])
        conn.append(node)

        # Create pointer into starting of next body
        ptr.append(self.getNodePointer(revcon.numNodes()))

        # Return the body object
        comp = TACSBody(id, TACSBodyType.CONSTRAINT, elems,
                        eids, nodes, xpts, conn, ptr, None, name)
        self.body_list.append(comp)

        return comp

    def addSlidingPivotConstraint(self, location, revaxis,
                                  bodyA, bodyB=None):
        '''
        Add a sliding pivot contraint
        '''
        # Make sure the hinge axis is of unit length
        assert(np.linalg.norm(revaxis) == 1.0)

        # Reduce dof of bodies
        self.ndof = self.ndof - 2

        # Check element types first
        assert(bodyA.btype == TACSBodyType.RIGID)
        if bodyB is not None:
            assert(bodyB.btype == TACSBodyType.RIGID)

        # Generate a body ID
        id = self.getBodyID()

        # Create the cylindrical constraint
        if bodyB is None:
            revcon = self.ehelper.createSlidingPivotConstraint(location,
                                                               revaxis,
                                                               bodyA.elems[0])
            name =  "sliding pivot constraint between %s and point" % (bodyA.name)

        else:
            revcon = self.ehelper.createSlidingPivotConstraint(location,
                                                               revaxis,
                                                               bodyA.elems[0],
                                                               bodyB.elems[0])
            name =  "sliding pivot constraint between %s and %s" % (bodyA.name,
                                                                    bodyB.name)

        # Set the body ID to which the element belongs
        revcon.setComponentNum(id)

        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [ self.current_ptr ]
        elems = []
        eids  = []
        elems = []
        nodes = []

        # Append the element to list
        elems.append(revcon)

        # Generate element id number for each element
        eids.append(self.getElementID())

        # Dummy link locations
        xpts.append(location[0])
        xpts.append(location[1])
        xpts.append(location[2])

        # Create a node number for this element and append to list
        node = self.getNodeID()
        nodes.append(node)

        # Append connectivities (revcons are two/three noded)
        conn.append(bodyA.conn[0])
        if bodyB is not None:
            conn.append(bodyB.conn[0])
        conn.append(node)

        # Create pointer into starting of next body
        ptr.append(self.getNodePointer(revcon.numNodes()))

        # Return the body object
        comp = TACSBody(id, TACSBodyType.CONSTRAINT, elems,
                        eids, nodes, xpts, conn, ptr, None, name)
        self.body_list.append(comp)

        return comp

    def addFixedConstraint(self, location, bodyA):
        '''
        Add a fixed contraint body to a rigid body
        '''
        # Check element types first
        assert(bodyA.btype == TACSBodyType.RIGID)

        # Reduce 6 dofs
        self.ndof = self.ndof - 6

        # Generate a body ID
        id = self.getBodyID()

        # Create the spherical constraint
        fcon = self.ehelper.createFixedConstraint(location,
                                                  bodyA.elems[0])
        name =  "fixed constraint between %s and point" % (bodyA.name)

        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [ self.current_ptr ]
        elems = []
        eids  = []
        elems = []
        nodes = []

        # Set the body ID to which the element belongs
        fcon.setComponentNum(id)

        # Append the element to list
        elems.append(fcon)

        # Generate element id number for each element
        eids.append(self.getElementID())

        # Dummy link locations
        xpts.append(location[0])
        xpts.append(location[1])
        xpts.append(location[2])

        # Create a node number for this element and append to list
        node = self.getNodeID()
        nodes.append(node)

        # Append connectivities (fcons are two noded)
        conn.append(bodyA.conn[0])
        conn.append(node)

        # Create pointer into starting of next body
        ptr.append(self.getNodePointer(fcon.numNodes()))

        # Return the body object
        comp = TACSBody(id, TACSBodyType.CONSTRAINT, elems,
                        eids, nodes, xpts, conn, ptr, None, name)
        self.body_list.append(comp)

        return comp

    def addSphericalConstraint(self, location, bodyA, bodyB=None):
        '''
        Add a spherical contraint body in between two rigid
        bodies. This can not be generated between rigid and
        flexible bodies. If there is a flexible body
        involved, create a rigid body adjoining the flexible
        body and then use the newly created rigid body to
        construct this spherical constaint.
        '''
        # Check element types first
        assert(bodyA.btype == TACSBodyType.RIGID)

        # Reduce dof of bodies
        self.ndof = self.ndof - 3

        if bodyB is not None:
            assert(bodyB.btype == TACSBodyType.RIGID)

        # Generate a body ID
        id = self.getBodyID()

        # Create the spherical constraint
        if bodyB is None:
            spcon = self.ehelper.createSphericalConstraint(location,
                                                           bodyA.elems[0])
            name =  "spherical constraint between %s and point" % (bodyA.name)
        else:
            spcon = self.ehelper.createSphericalConstraint(location,
                                                           bodyA.elems[0],
                                                           bodyB.elems[0])
            name =  "spherical constraint between %s and %s" % (bodyA.name,
                                                                bodyB.name)


        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [ self.current_ptr ]
        elems = []
        eids  = []
        elems = []
        nodes = []

        # Set the body ID to which the element belongs
        spcon.setComponentNum(id)

        # Append the element to list
        elems.append(spcon)

        # Generate element id number for each element
        eids.append(self.getElementID())

        # Dummy link locations
        xpts.append(location[0])
        xpts.append(location[1])
        xpts.append(location[2])

        # Create a node number for this element and append to list
        node = self.getNodeID()
        nodes.append(node)

        # Append connectivities (spcons are two/three noded)
        conn.append(bodyA.conn[0])
        if bodyB is not None:
            conn.append(bodyB.conn[0])
        conn.append(node)

        # Create pointer into starting of next body
        ptr.append(self.getNodePointer(spcon.numNodes()))

        # Return the body object
        comp = TACSBody(id, TACSBodyType.CONSTRAINT, elems,
                        eids, nodes, xpts, conn, ptr, None, name)
        self.body_list.append(comp)

        return comp

    def addRevoluteDriver(self, revaxis, speed, drivenComp):
        """
        Revolute driver is an element that drives the connected
        body at a specified speed about the revaxis.
        """
        # Generate a body ID
        id = self.getBodyID()

        # Reduce ## check
        self.ndof = self.ndof - 5

        # Create the driver element
        driver = self.ehelper.createRevoluteDriver(revaxis, speed)

        # Set the body ID to which the element belongs
        driver.setComponentNum(id)

        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [ self.current_ptr ]
        elems = []
        eids  = []
        elems = []
        bcs   = []
        nodes = []

        for bc in drivenComp.bcs:
            # Append the element to list
            elems.append(driver)

            # Generate element id number for each element
            eids.append(self.getElementID())

            # Dummy link locations
            xpts.append(0.0)
            xpts.append(0.0)
            xpts.append(0.0)

            # Get a new node number for this element
            nid = self.getNodeID()
            nodes.append(nid)

            # Append connectivities (links are three noded)
            conn.append(bc)
            conn.append(nid)

            # Append the boundary conditions
            bcs.append(nid)
            bcs.append(bc)

            # Create pointer into starting of next body
            ptr.append(self.getNodePointer(driver.numNodes()))

        # Return the body object
        name =  "revolute driver for %s" % (drivenComp.name)
        comp = TACSBody(id, TACSBodyType.CONSTRAINT, elems,
                        eids, nodes, xpts, conn, ptr, bcs, name)
        self.body_list.append(comp)

        return comp

    def addMotionDriver(self, dir, speed, drivenComp, arrest_rotations=False, linearized=False):
        """
        Motion driver is an element that drives the connected
        body at a specified speed about the revaxis.
        """
        # Generate a body ID
        id = self.getBodyID()

        # Reduce 3 degrees of freedom ## check
        if not linearized:
            self.ndof = self.ndof - 3

        # Create the driver element
        if linearized:
            mdriver = self.ehelper.createLinearizedMotionDriver(dir, speed, arrest_rotations)
        else:
            mdriver = self.ehelper.createMotionDriver(dir, speed, arrest_rotations)

        # Set the body ID to which the element belongs
        mdriver.setComponentNum(id)

        # Initialize lists
        xpts  = []
        conn  = []
        ptr   = [ self.current_ptr ]
        elems = []
        eids  = []
        elems = []
        bcs   = []
        nodes = []

        for bc in drivenComp.bcs:
            # Append the element to list
            elems.append(mdriver)

            # Generate element id number for each element
            eids.append(self.getElementID())

            # Dummy link locations
            xpts.append(dir[0])
            xpts.append(dir[1])
            xpts.append(dir[2])

            # Get a new node number for this element
            nid = self.getNodeID()
            nodes.append(nid)

            # Append connectivities (links are three noded)
            conn.append(bc)
            conn.append(nid)

            # Append the boundary conditions
            bcs.append(nid)
            bcs.append(bc)

            # Create pointer into starting of next body
            ptr.append(self.getNodePointer(mdriver.numNodes()))

        # Return the body object
        name =  "motion driver for %s" % (drivenComp.name)
        comp = TACSBody(id, TACSBodyType.CONSTRAINT, elems,
                        eids, nodes, xpts, conn, ptr, bcs, name)
        self.body_list.append(comp)

        return comp

    # Functions to handle loading of body and dynamic pameters from a file
    def rigidBody(self, filename):
        """
        Load the data from file and create a rigid body
        """
        # Open file, load content and close
        bodyParams = BodyParams(inp_dir+filename)

        #-------------------------------------------------------------#
        # EVALUTE MASS and CENTER OF MASS
        #-------------------------------------------------------------#

        mass = bodyParams.vol*bodyParams.rho
        xcg  = np.array(bodyParams.xcg, self.dtype)

        #-------------------------------------------------------------#
        # Specify simulation REF point fixed to the body. One can pick
        # any inertial reference point regardless of whether it is
        # geometrically significant or not. Examples: xcg or
        # np.random.rand(3), or simply (0,0,0)
        #-------------------------------------------------------------#

        xref = xcg

        # Initial conditions
        #
        # when using xref velocity components due to omega add up
        # when using x=0 velocity components are purely translational -- might be better for kinematic compatibility

        # Inertial properties
        #
        # when using xcg the first moment of mass is zero by definition
        # when using xref other than xcg the first moment of mass is non zero

        #xref = np.array([0.0,0.0,0.9]) # top of the shaft
        #self.xref = np.random.rand(3)

        #-------------------------------------------------------------#
        # First moment of mass
        #-------------------------------------------------------------#

        # Load the first moment of mass computed about the inertial
        # origin (0,0,0). Also, compute manually based on CG location
        # and check if they are same (sanity check)
        corig = np.array(bodyParams.c)*bodyParams.rho # Loaded
        cxcg = mass*np.array([xcg[0], xcg[1], xcg[2]]) # Computed
        assert(np.allclose(np.linalg.norm(corig-cxcg), 0.0) == 1)

        # Compute cref about the "ref" point manually mass*(xcg - xref)
        cref = cxcg - mass*np.array([xref[0], xref[1], xref[2]])
        if xref is xcg:
            # sanity check: must be zero
            assert(np.allclose(np.linalg.norm(cref), 0.0) == 1)
        else:
            # sanity check: must be non zero
            assert(np.allclose(np.linalg.norm(cref), 0.0) == 0)

        #-------------------------------------------------------------#
        # Second moment of mass
        #-------------------------------------------------------------#

        # Assert if J loaded from file is positive definite
        JCG = np.array(bodyParams.J, self.dtype)*bodyParams.rho
        assert(self.isPositiveDefinite(JCG)==1)

        # The matrix of inertia is returned in the central coordinate
        # system (G, Gx, Gy, Gz) where G is the centre of mass of the
        # system and Gx, Gy, Gz the directions parallel to the
        # X(1,0,0) Y(0,1,0) Z(0,0,1) directions of the absolute
        # cartesian coordinate system. It is possible to compute the
        # matrix of inertia at another location point using the
        # Huyghens theorem

        # Compute the offset of CG from the ref point of the body
        x0 = xcg[0] - xref[0]
        y0 = xcg[1] - xref[1]
        z0 = xcg[2] - xref[2]

        Jref = np.zeros(6, self.dtype)
        Jref[0] = JCG[0] + mass*(y0*y0 + z0*z0)
        Jref[3] = JCG[3] + mass*(x0*x0 + z0*z0)
        Jref[5] = JCG[5] + mass*(y0*y0 + x0*x0)
        Jref[1] = JCG[1] - mass*y0*x0
        Jref[2] = JCG[2] - mass*z0*x0
        Jref[4] = JCG[4] - mass*z0*y0

        #-------------------------------------------------------------#
        # Now every thing is sane to create the rigid body
        #-------------------------------------------------------------#

        body = self.addRigidBody(bodyParams.name,
                                 mass,
                                 cref, # negative or positive
                                 Jref,
                                 xref,
                                 np.array(bodyParams.vel, self.dtype),
                                 np.array(bodyParams.omega, self.dtype),
                                 bodyParams.mesh[0],
                                 bodyParams.mesh[1],
                                 np.array(bodyParams.grav, self.dtype),
                                 xref) # we subtract all

        # Return the rigid body
        return body

    def shellBody(self, filename, shell_type='MITC9'):
        """
        Load the data from file and create a flexible body.
        """
        # Create constitutive props for shell
        rho       = 2500.0
        E         = 70.0e9
        nu        = 0.3
        kcorr     = 5.0/6.0
        ys        = 350.0e6
        tmin      = 1.0e-3
        thickness = 1.0e-2
        tmax      = 3.0e-2
        #stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys,
        #                             thickness, i, tmin, tmax)
        shellStiff = ShellStiffness(rho, E, nu, kcorr, ys,
                                    thickness, tmin, tmax)

        # Open file, load content and close
        bodyParams = BodyParams(inp_dir+filename)

        # Create the shell body
        if shell_type == 'MITC9':
            body = self.addMITC9ShellBody(bodyParams.name,
                                          bodyParams.mesh[0],
                                          bodyParams.mesh[1],
                                          np.array(bodyParams.vel),
                                          np.array(bodyParams.omega),
                                          np.array(bodyParams.grav),
                                          shellStiff)
        else:
            body = self.addMITCShellBody(bodyParams.name,
                                         bodyParams.mesh[0],
                                         bodyParams.mesh[1],
                                         shellStiff)
        return body

    def beamBody(self, filename, constitutive_props=None):
        """
        Load the data from cbar file and model the body using
        beam.
        """
        if constitutive_props is None:
            # Create constitutive props for beam
            axis = np.array([0.0,0.0,1.0])

            # mass per unit span
            mass = 3.67 # kg/m

            # moments of inertia per unit span
            Ixx = 4.0e-4 # kg.m
            Iyy = 4.0e-4 # kg.m
            Ixy = 0.0

            # six stiffness props
            EA = 1.26e8 # axial stiffness
            GJ = 3.8e2 # torsional stiffness

            # bending stiffness
            EIx = 3.0e3
            EIy = 1.4e4

            # shear stiffness
            kGAx = 6.3e6
            kGAy = 6.3e6
            beamStiff = constitutive.Timoshenko(mass,
                                                Ixx, Iyy, Ixy,
                                                EA, GJ,
                                                EIx, EIy,
                                                kGAx, kGAy,
                                                axis)
        else:
            beamStiff = constitutive_props

        # Open file, load content and close
        bodyParams = BodyParams(inp_dir+filename)

        # Create the shell body
        body = self.addBeamBody(bodyParams.name, bodyParams.mesh[0],
                                np.array(bodyParams.vel, self.dtype),
                                np.array(bodyParams.omega, self.dtype),
                                np.array(bodyParams.grav, self.dtype),
                                beamStiff)
	# Return the rigid body
	return body

    def body(self, filename, type):
        '''
        This is a wrapper around three functions rigidBody, shellBody,
        beamBody. The of body of specified type will be created and
        returned.
        '''

        body = None
        if type == TACSBodyType.RIGID:
            body = self.rigidBody(filename)
        elif type == TACSBodyType.SHELL:
            body = self.shellBody(filename, 'MITC9')
        elif type == TACSBodyType.BEAM:
            body = self.beamBody(filename)
        else:
            raise ValueError("unsupported body type to create")

        return body

class ElementHelper:
    """
    Class to help create elements. This class provides functions
    that can be used to create different element types.
    """
    def __init__(self, comm=None):
        # TACS Communicator
        self.comm = comm

        ## # Default origin is at zero
        ## if origin is not None:
        ##     self.origin = elements.GibbsVector(origin[0], origin[1], origin[2])

        ##     # Define useful axes
        ##     self.xaxis = elements.GibbsVector(origin[0]+1.0, origin[1], origin[2])
        ##     self.yaxis = elements.GibbsVector(origin[0], origin[1]+1.0, origin[2])
        ##     self.zaxis = elements.GibbsVector(origin[0], origin[1], origin[2]+1.0)

        ## else:
        ##     self.origin = elements.GibbsVector(0.0,0.0,0.0)

        ##     # Define useful axes
        ##     self.xaxis = elements.GibbsVector(1.0, 0.0, 0.0)
        ##     self.yaxis = elements.GibbsVector(0.0, 1.0, 0.0)
        ##     self.zaxis = elements.GibbsVector(0.0, 0.0, 1.0)

        ## # Create a global ref frame (unused currently)
        ## self.globalFrame = elements.RefFrame(self.origin, self.xaxis, self.yaxis)

        return

    def createRigidBody(self,
                        mass, c, J,
                        rInit, vInit, wInit,
                        bdffile=None, gmsh=1,
                        gravity=None,
                        voffset=None,
                        basepoint=None, xpoint=None, ypoint=None):
        # acceleration due to gravity [kg.m.s-2]
        if gravity is not None:
            g = elements.GibbsVector(gravity[0], gravity[1], gravity[2])
        else:
            g = elements.GibbsVector(0.0,0.0,-9.81)

        # Convert to gibbs vectors
        r = elements.GibbsVector(rInit[0], rInit[1], rInit[2])
        v = elements.GibbsVector(vInit[0], vInit[1], vInit[2])
        w = elements.GibbsVector(wInit[0], wInit[1], wInit[2])

        # Create a frame attached to the body -- bodyframe. The
        # default bodyframe is located at the rInit of the body but
        # aligned with the global frame. The inertial properties must
        # be supplied in the body frame.
        if (basepoint or xpoint or ypoint) is None:
            o = elements.GibbsVector(rInit[0], rInit[1], rInit[2])
            x = elements.GibbsVector(rInit[0]+1.0, rInit[1], rInit[2])
            y = elements.GibbsVector(rInit[0], rInit[1]+1.0, rInit[2])
        else:
            o = elements.GibbsVector(basepoint[0], basepoint[0], basepoint[2])
            x = elements.GibbsVector(xpoint[0], xpoint[1], xpoint[2])
            y = elements.GibbsVector(ypoint[0], ypoint[1], ypoint[2])

        # Create the body frame
        bodyFrame = elements.RefFrame(o, x, y)

        # Create the body
        body = elements.RigidBody(bodyFrame, mass, c, J, r, v, w, g)

        # Create visualization for the rigid body using BDF mesh
        if bdffile is not None:
            mesh = TACS.MeshLoader(self.comm)
            mesh.setConvertToCoordinate(gmsh)
            mesh.scanBDFFile(mesh_dir+bdffile)
            [ptr, conn, cnums, xpts] = mesh.getConnectivity()

            ptr = self.comm.bcast(ptr, root=0)    #MPI.ROOT
            cnums = self.comm.bcast(cnums, root=0)
            conn = self.comm.bcast(conn, root=0)
            xpts = self.comm.bcast(xpts, root=0)

            npts = len(xpts)/3
            nelems = ptr.shape[0]-1
            conn = np.array(conn, dtype=np.intc)

            if voffset is not None:
                voffset = elements.GibbsVector(voffset[0], voffset[1], voffset[2])

            # Create the visualization object using hte mesh points
            viz = elements.RigidBodyViz(npts, nelems, xpts, conn, voffset)
            body.setVisualization(viz)

        else:
            # Create a placeholder visualization object as cuboid
            viz = elements.RigidBodyViz(Lx=0.1,Ly=0.1,Lz=0.1)
            body.setVisualization(viz)

        return body

    def createRigidLink(self, driver):
        return elements.RigidLink(driver)

    def createFlexLink(self, bodyA, loc, frame, use_moments):
        orig = elements.GibbsVector(loc[0], loc[1], loc[2])
        return elements.AverageConstraint(bodyA, orig, frame, use_moments)

    def createFixedConstraint(self, loc, bodyA):
        orig = elements.GibbsVector(loc[0], loc[1], loc[2])
        return elements.FixedConstraint(orig, bodyA)

    def createSphericalConstraint(self, loc, bodyA, bodyB=None):
        orig = elements.GibbsVector(loc[0], loc[1], loc[2])
        constraint = elements.SphericalConstraint(orig, bodyA, bodyB)
        return constraint

    def createRevoluteConstraint(self, loc, revaxis, bodyA, bodyB=None):
        orig = elements.GibbsVector(loc[0], loc[1], loc[2])
        axis = elements.GibbsVector(revaxis[0], revaxis[1], revaxis[2])
        return elements.RevoluteConstraint(orig, axis, bodyA, bodyB)

    def createCylindricalConstraint(self, loc, revaxis, bodyA, bodyB=None):
        orig = elements.GibbsVector(loc[0], loc[1], loc[2])
        axis = elements.GibbsVector(revaxis[0], revaxis[1], revaxis[2])
        return elements.CylindricalConstraint(orig, axis, bodyA, bodyB)

    def createPrismaticConstraint(self, loc, revaxis, bodyA, bodyB=None):
        orig = elements.GibbsVector(loc[0], loc[1], loc[2])
        axis = elements.GibbsVector(revaxis[0], revaxis[1], revaxis[2])
        return elements.PrismaticConstraint(orig, axis, bodyA, bodyB)

    def createSlidingPivotConstraint(self, loc, revaxis, bodyA, bodyB=None):
        orig = elements.GibbsVector(loc[0], loc[1], loc[2])
        axis = elements.GibbsVector(revaxis[0], revaxis[1], revaxis[2])
        return elements.SlidingPivotConstraint(orig, axis, bodyA, bodyB)

    def createCylindricalConstraint(self, loc, revaxis, bodyA, bodyB=None):
        orig = elements.GibbsVector(loc[0], loc[1], loc[2])
        axis = elements.GibbsVector(revaxis[0], revaxis[1], revaxis[2])
        return elements.CylindricalConstraint(orig, axis, bodyA, bodyB)

    def createMITC9ShellElement(self, vInit, wInit, gravity, shellStiff):
        assert(isinstance(shellStiff, constitutive.isoFSDT)==1)
        v = elements.GibbsVector(vInit[0], vInit[1], vInit[2])
        w = elements.GibbsVector(wInit[0], wInit[1], wInit[2])
        g = elements.GibbsVector(gravity[0], gravity[1], gravity[2])
        element = elements.MITC(shellStiff, g, v, w)
        return element

    def createSolidElement(self, order, solidStiff, elem_type, c):
        assert(isinstance(solidStiff, constitutive.SolidStiff)==1)
        element = elements.Solid(order, solidStiff, elem_type, c)
        return element

    def createMITCShellElement(self, shellStiff, c):
        assert(isinstance(shellStiff, constitutive.isoFSDT)==1)
        element = elements.MITCShell(2,shellStiff, component_num=c)
        return element

    def createBeamElement(self, vInit, wInit, gravity, beamStiff):
        assert(isinstance(beamStiff, constitutive.Timoshenko)==1)
        v = elements.GibbsVector(vInit[0], vInit[1], vInit[2])
        w = elements.GibbsVector(wInit[0], wInit[1], wInit[2])
        g = elements.GibbsVector(gravity[0], gravity[1], gravity[2])
        element = elements.MITCBeam(beamStiff, g, v, w)
        return element

    def createRevoluteDriver(self, revaxis, speed):
        axis = elements.GibbsVector(revaxis[0], revaxis[1], revaxis[2])
        return elements.RevoluteDriver(axis, speed)

    def createMotionDriver(self, diraxis, speed, arrest_rotations):
        axis = elements.GibbsVector(diraxis[0], diraxis[1], diraxis[2])
        return elements.MotionDriver(axis, speed, arrest_rotations)

    def createLinearizedMotionDriver(self, amplitude, omega, arrest_rotations):
        amp = elements.GibbsVector(amplitude[0], amplitude[1], amplitude[2])
        return elements.LinearizedMotionDriver(amp, omega, arrest_rotations)

class TACSSolver:
    '''
    Solves the problem in time
    '''
    @staticmethod
    def createSolver(tacs, args):
        """
        Create the Integrator (solver) and configure it with objective
        and functions to evaluate
        """
        start_time           = args.start_time
        num_steps_per_rev    = args.num_steps_per_rev
        num_revs             = args.num_revs
        speed                = args.speed
        integration_order    = args.integration_order
        solver_rel_tol       = args.solver_rel_tol
        solver_abs_tol       = args.solver_abs_tol
        max_newton_iters     = args.max_newton_iters
        print_level          = args.print_level
        output_freq          = args.output_frequency
        output_dir           = args.output_dir
        states_dir           = args.states_dir
        femat                = args.femat

        # Parse the matrix ordering
        if args.ordering == "TACS_AMD":
            ordering = TACS.PY_TACS_AMD_ORDER
        elif args.ordering == "AMD":
            ordering = TACS.PY_AMD_ORDER
        elif args.ordering == "RCM":
            ordering = TACS.PY_RCM_ORDER
        elif args.ordering == "NATURAL":
            ordering = TACS.PY_NATURAL_ORDER
        else:
            print("wrong ordering specified: ", args.ordering)

        # Figure out the end time and number of steps for creating
        # integrator
        num_steps = num_steps_per_rev*num_revs
        h = 2.0*np.pi/(speed*num_steps_per_rev)
        end_time = h*num_steps

        # Create an integrator for TACS
        if args.integrator == 'DIRK':
            # Backout the stages from user-requested order
            if integration_order > 1:
                stages = integration_order - 1
            else:
                stages = 1
            integrator = TACS.DIRKIntegrator(tacs,
                                             start_time, end_time,
                                             num_steps,
                                             stages)
        elif args.integrator == 'BDF':
            integrator = TACS.BDFIntegrator(tacs,
                                            start_time, end_time,
                                            num_steps,
                                            integration_order)
        elif args.integrator == 'ABM':
            integrator = TACS.BDFIntegrator(tacs,
                                            start_time, end_time,
                                            num_steps,
                                            integration_order)
        elif args.integrator == 'NBG':
            integrator = TACS.BDFIntegrator(tacs,
                                            start_time, end_time,
                                            num_steps,
                                            integration_order)
        else:
            print("wrong integrator type requested:", args.integrator)

        # Set other parameters for integration
        integrator.setRelTol(solver_rel_tol)
        integrator.setAbsTol(solver_abs_tol)
        integrator.setMaxNewtonIters(max_newton_iters)
        integrator.setUseFEMat(femat,ordering)
        integrator.setPrintLevel(print_level)
        integrator.setOutputFrequency(output_freq)
        integrator.setOutputPrefix(output_dir)
        #integrator.setJacAssemblyFreq(3)
        #integrator.setUseLapack(1)
        return integrator

def remove_duplicates(x):
    '''
    Removes duplicates in a list
    '''
    a = []
    for i in x:
        if i not in a:
            a.append(i)
        else:
            print("ERROR: Duplicate body ", i)
            exit(0)
    return a

def getElemCompNumbersFromBDF(file):
    '''
    Returns a list of component numbers for elements in the BDF file
    '''
    try:
        inpFile = open(file, "r")
        content = list(inpFile.readlines())
        inpFile.close()
    except:
        raise IOError

    # Loop through contents and get the connectivities
    cnums = {}
    for line in content:
        entry = line.split()
        if entry[0] == "CQUAD" or entry[0] == "CQUAD9"or entry[0] == "CQUADR":
            # reduce by one to fix one based numbering
            cnums[int(entry[1])-1] = (int(entry[2])-1)
        if entry[0] == "CHEXA":
            # reduce by one to fix one based numbering
            cnums[int(entry[1])-1] = (int(entry[2])-1)

    # return connectivities of beam elements
    return cnums

class CBARLoader:
    """
    Class to load CBAR mesh and handle related information
    """
    def __init__(self, cbarfile, stiffness_function=None):
        # Store the filename
        self.filename = cbarfile

        # Load the file
        try:
            inpFile = open(self.filename, "r")
            content = list(inpFile.readlines())
            inpFile.close()
        except:
            print("Error: File not found", cbarfile)
            raise IOError

        self.start_node = None

        # Loop through contents
        self.xpts = []
        nids = []
        conn = []
        bcs  = []
        eids = []

        # Parse the file content and extract name
        for line in content:
            # Split the line into tuples
            entry = line.split()

            # Node Locations
            if entry[0] == "GRID":
                nids.append(int(entry[1]))
                self.xpts.append(float(entry[2]))
                self.xpts.append(float(entry[3]))
                self.xpts.append(float(entry[4]))

            # Connectivities and eids
            if entry[0] == "CBAR":
                eids.append(int(entry[1]))
                conn.append(int(entry[2]))
                conn.append(int(entry[3]))
                conn.append(int(entry[4]))

            # Boundary conditions
            if entry[0] == "SPC":
                bcs.append(int(entry[1]))

        # sanity check
        assert(len(nids) == len(list(set(conn))))
        assert(len(conn)/3 == len(eids))

        # number of elements
        self.nelems = len(eids)

        # Correct to zero based numbering
        start_node = min(nids)
        start_eid = min(eids)

        # Offset and store as class variables
        self.conn = []
        for c in conn:
            self.conn.append(c-start_node)

        self.bcs = []
        for bc in bcs:
            self.bcs.append(bc-start_node)

        self.eids = []
        for e in eids:
            self.eids.append(e-start_eid)

        if not isinstance(stiffness_function, constitutive.Timoshenko):
            # List of central nodes in 3-noded beam element
            cnodes = []
            for i in range(len(self.conn)/3):
                cnodes.append(self.conn[i*3+1])

            # Node location of these central nodes
            X = []
            for n in cnodes:
                x = [self.xpts[n*3+0], self.xpts[n*3+1], self.xpts[n*3+2]]
                X.append(x)

            # Evaluate the stiffness funtion to get the list of constitutive objects
            self.constitutive = stiffness_function(X)

        return

    def toString(self):
        print("filename  ", self.filename)
        print("xpts      ", self.xpts)
        print("eids      ", self.eids)
        print("conn      ", self.conn)
        print("bcs       ", self.bcs)
        return

# maps label to attribute name and types
label_attr_map = {
    "name": [ "name", str],
    "density": [ "rho", float],
    "volume": [ "vol", float],
    "xcg": ["xcg", float, float, float],
    "vel": ["vel", float, float, float],
    "omega": ["omega", float, float, float],
    "grav": ["grav", float, float, float],
    "c": ["c", float, float, float],
    "J": ["J", float, float, float, float, float, float],
    "mesh": [ "mesh", str, int]
}

# Class to load the input parameters to create rigid and flexible
# bodies
class BodyParams(object):
    def __init__(self, input_file_name):
        """
        Parse the input data and create a dictionary
        """
        with open(input_file_name, 'r') as input_file:
            for line in input_file:
                row = line.split()
                label = row[0]
                data = row[1:]  # rest of row is data list

                attr = label_attr_map[label][0]
                datatypes = label_attr_map[label][1:]

                values = [(datatypes[i](data[i])) for i in range(len(datatypes))]
                self.__dict__[attr] = values if len(values) > 1 else values[0]
        return

class TACSProblem(object):
    pass

class TACSStaticsProblem(TACSProblem):
    pass

class TACSDynamicsProblem(TACSProblem):
    '''
    This class wraps around 3 important objects pertaining to a
    flexible multibody dynamic simulation in TACS.

    ==================================================================
    Variables
    ==================================================================
    s
    helper     : created during constructor invocation and handles
                 book keeping of xpts, conn, ptr, elem_list etc used
                 for TACS creation
    tacs       : instance of TACS, created after calling initialize()
    integrator : instance of time integration scheme, created after
                 calling initialize()

    ==================================================================
    Methods
    ==================================================================

    initialize() : creates TACS and Integator
    solve()      : Solves the problem in time
    '''
    def __init__(self, comm, write_states=False, hot_start=False):
        """
        Create a TACS dynamics problem by adding bodies and
        constraints
        """
        # Communicator
        self.comm = comm

        # Datatype used
        self.dtype = TACS.dtype

        # Create an intance of helper class to handle connectivities
        # and pointers, bc, xpts internally
        self.builder = TACSBuilder(comm)

        # Store a reference to body list (will be referenced during initialize call)
        self.tacs_body_list = None

        # Set TACS to None (will be referenced during initialize call)
        self.tacs  = None

        # Set integrator to None (will be referenced during initialize
        # call)
        self.integrator = None
        self.nsteps = 0

        # Hotstart using stored states?
        self.hot_start = hot_start

        # Write States to disk
        self.write_states = write_states

        return

    def toString(self):
        print("================================================================================")
        print(" TACS Dynamic Analysis Problem ")
        print("================================================================================")
        print(" > TACS                  ", self.tacs)
        print(" > Integrator            ", self.integrator)
        print(" > Initialized?          ", (self.integrator is not None and self.tacs is not None))
        print(" > Hot start?            ", self.hot_start)
        print(" > Write states to disk? ", self.write_states)
        print(" > Num Time Steps        ", self.nsteps)
        print(" > Bodies                ", ['%d, %s' % (body.id, body.name) for body in self.tacs_body_list])
        print("================================================================================")
        return

    def getArguments(self):
        """
        Commandline arguments are used to control the solution process
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--start_time'            , type=float  , default=0.00       , help='Start time of simulation')
        parser.add_argument('--num_revs'              , type=float  , default=1.0        , help='Number of revolutions of blade')
        parser.add_argument('--num_steps_per_rev'     , type=float  , default=360.0      , help='Number of steps to take per time step')
        parser.add_argument('--speed'                 , type=float  , default=109.12     , help='Angular speed of the rotors')
        parser.add_argument('--solver_rel_tol'        , type=float  , default=1.0e-7     , help='The relative reduction in residual for stopping nonlinear solution')
        parser.add_argument('--solver_abs_tol'        , type=float  , default=1.0e-4     , help='The absolute reduction in residual for stopping nonlinear solution')
        parser.add_argument('--max_newton_iters'      , type=int    , default=30         , help='Maximum iterations for newton_solve')
        parser.add_argument('--output_frequency'      , type=int    , default=0          , help='Fraction of number of time steps to write the f5 output file')
        parser.add_argument('--output_dir'            , type=str    , default='results'  , help='Directory for tecplot output files')
        parser.add_argument('--states_dir'            , type=str    , default='states'   , help='Directory for TACS state vectors')
        parser.add_argument('--print_level'           , type=int    , default=1          , help='Amount of print. 0 : off, 1 = report after each time step, 2= report after each Newton iteration')
        parser.add_argument('--femat'                 , type=int    , default=1          , help='0 : uses femat, 1 : uses serial matrix')
        parser.add_argument('--flexible'              , type=int    , default=1          , help='0 : rigid simulation, 1 : flexible simulation')
        parser.add_argument('--moment_flag'           , type=int    , default=3          , help='7: all axes, 1, is xaxis, 3 is xy')
        parser.add_argument('--ordering'              , type=str    , default='TACS_AMD' , help='Ordering of the matrices: TACS_AMD, AMD, NATURAL, RCM')
        parser.add_argument('--integrator'            , type=str    , default='BDF'      , help='Type of integrator to use: DIRK, BDF, ABM, NBG')
        parser.add_argument('--integration_order'     , type=int    , default=2          , help='The order of accuracy of the integration scheme')
        return parser.parse_args()

    def initialize(self):
        """
        Create an instance of TACS and Integator.
        """
        # Setup commandline arguments
        self.args = self.getArguments()
        if self.args.ordering == "TACS_AMD":
            ordering = TACS.PY_TACS_AMD_ORDER
        elif self.args.ordering == "AMD":
            ordering = TACS.PY_AMD_ORDER
        elif self.args.ordering == "RCM":
            ordering = TACS.PY_RCM_ORDER
        elif self.args.ordering == "NATURAL":
            ordering = TACS.PY_NATURAL_ORDER
        else:
            print("wrong ordering specified: ", self.args.ordering)
            raise ValueError('Specified ordering does not exist')

        # Everything to do for TACS Creation
        self.tacs = self.builder.getTACS(ordering, TACS.PY_DIRECT_SCHUR)

        # Things for configuring time marching
        self.integrator = TACSSolver.createSolver(self.tacs, self.args)

        # Control F5 output
        if self.builder.rigid_viz == 1:
            flag = (TACS.ToFH5.NODES|
                    TACS.ToFH5.DISPLACEMENTS)
            rigidf5 = TACS.ToFH5(self.tacs, TACS.PY_RIGID, flag)
            self.integrator.setRigidOutput(rigidf5)

        if self.builder.shell_viz == 1:
            flag = (TACS.ToFH5.NODES|
                    TACS.ToFH5.DISPLACEMENTS|
                    TACS.ToFH5.STRAINS|
                    TACS.ToFH5.STRESSES|
                    TACS.ToFH5.EXTRAS)
            shellf5 = TACS.ToFH5(self.tacs, TACS.PY_SHELL, flag)
            self.integrator.setShellOutput(shellf5)

        if self.builder.beam_viz == 1:
            flag = (TACS.ToFH5.NODES|
                    TACS.ToFH5.DISPLACEMENTS|
                    TACS.ToFH5.STRAINS|
                    TACS.ToFH5.STRESSES|
                    TACS.ToFH5.EXTRAS)
            beamf5 = TACS.ToFH5(self.tacs, TACS.PY_TIMOSHENKO_BEAM, flag)
            self.integrator.setBeamOutput(beamf5)

        if self.builder.solid_viz == 1:
            flag = (TACS.ToFH5.NODES|
                    TACS.ToFH5.DISPLACEMENTS|
                    TACS.ToFH5.STRESSES|
                    TACS.ToFH5.EXTRAS)
            solidf5 = TACS.ToFH5(self.tacs, TACS.PY_SOLID, flag)
            self.integrator.setSolidOutput(solidf5)

        # store the refernce to body list after initializations are complete
        self.tacs_body_list  = self.builder.body_list

        # Get the new ordering of nodes from TACS
        self.newNodeIndices = self.tacs.getReordering()

        self.nsteps = self.integrator.getNumTimeSteps()

        # Print basic details about the problem
        if self.comm.rank ==0:
            self.toString()

        return

    def getForces(self, tindex=None):
        """
        Returns a BVEC of forces based on the current forces set into
        each body object.

        Currently we apply forces only on flexible components.

        #TODO: Generalize for both rigid and flexible
        """

        # create a distributed vector (aka bvec) to contain the
        # force/moment values
        forces = self.tacs.createVec()

        # Get a pointer to the array to place the values inside
        farray = forces.getArray()

        # Set the forces from each body
        for body in self.tacs_body_list:

            # Use user provided F(x,t) during body creation (call
            # funtofem directly here??)
            if body.btype == TACSBodyType.FLEXIBLE:

                if body.forcing is not None:
                    body.forcing(tindex, body)
                else:
                    body.forces[:] = 0.0

                for n in range(body.nnodes):
                    nn = body.dist_nodes[n]
                    farray[nn*8:nn*8+3] = body.forces[n*8:n*8+3]

        return forces

    def setDisplacements(self, tacs_disp_vec):
        """
        The displacement is provided in reordered bvec. This function
        gets the reordered array, maps it with the old ordering and
        sets into each TACSBody's local displacement array.
        """
        u = tacs_disp_vec.getArray()
        for body in self.tacs_body_list:
            if body.btype == TACSBodyType.FLEXIBLE:
                for n in range(body.nnodes):
                    nn = body.dist_nodes[n]
                    body.disps[n*8:(n+1)*8] = u[nn*8:(nn+1)*8]
        return

    def step(self, tindex):
        """
        Take a step in time using TACS and update bodies
        """
        if self.tacs is None or self.integrator is None:
            self.initialize()

        # Zero out error flags from execution
        load_states_failed = 0
        flag = 0

        # Try loading from disk if configured
        if self.hot_start is True:
            print( " >> Reading TACS state vector from disk from %s %d/%d" %
                   (self.args.states_dir, tindex, self.nsteps))
            # Load states from disk
            load_states_failed = self.integrator.loadStates(tindex,
                                                            self.args.states_dir)

        if (load_states_failed == 1):
            print("Loading states failed. Will try to integrate")

        # If loading failed do a direct integration
        if (self.hot_start is False) or (load_states_failed == 1):
            # Take a step time by actual integration
            flag = self.integrator.iterate(tindex, self.getForces(tindex))
            if flag != 0:
                if tindex == 1:
                    print("Time marching failed at step", flag , ". Did you call step(0) first?")
                else:
                    print("Time marching failed at step", flag)
                    raise RuntimeError('Time marching failed at step %d'%(flag))

            # Write the states to disk if configured or if we are
            # doing actual integration for a failed file-load
            if self.write_states is True or load_states_failed == 1:
                self.integrator.persistStates(tindex, self.args.states_dir)

        # Get the displacement vector from TACS and set into the
        # bodies
        t, disps, vels, accs = self.integrator.getStates(tindex)
        self.setDisplacements(disps)

        # Write the output of the body as configured by the user
        for body in self.tacs_body_list:
            if body.write is not None:
                body.write(body, tindex)

        return flag

    def march(self):
        """
        Iterates continously until end time
        """
        for k in range(self.nsteps+1):
            flag = self.step(k)
            if flag != 0:
                raise RuntimeError("integration failed")
        return flag

    def solve(self, dvs=None, funcs=None, dfdx=None,
              check_gradient=False):
        """
        Initialize to make sure TACS and Integrator are created. Then
        integrate forward in time
        """
        # Create TACS and Integrator if does not exist
        if self.tacs is None or self.integrator is None:
            # Create TACS and integrator
            self.initialize()

        # Set the design variable vector if supplied
        if dvs is not None:
            self.tacs.setDesignVars(dvs)

        # Set the functions to evaluate
        if funcs is not None:
            self.integrator.setFunctions(funcs, len(dvs))

        # If space for storing derivative is not supplied
        if dfdx is None:
            # Solve the problem forward in  time
            flag = self.march() #integrator.integrate()
            # Evaluate the function values if function is set
            if funcs is not None:
                return self.integrator.evalFunctions(funcs)
            else:
                return None
        else:
            # Perform function gradient evaluation using adjoint
            nvars = dvs.size
            funcVals = np.zeros(len(funcs), self.dtype)
            if check_gradient is True:
                if TACS.dtype == np.complex:
                    dh = 1.0e-30
                else:
                    dh = 1.0e-6
                self.integrator.checkGradients(dh)
                return None
            else:
                # March forward and evaluate functions
                flag = self.march() #integrator.integrate()
                if flag != 0:
                    raise RuntimeError('integration failed')
                funcVals = self.integrator.evalFunctions(funcs)

                # March backward and evaluate gradient
                self.integrator.integrateAdjoint()
                self.integrator.getGradient(dfdx)

                return funcVals

    def getFuncGrad(self, dvs, funcs, dfdx, funcOnly=False):
        """
        This function can be used within a optimizer for obtaining
        function and gradient values at each iteration.
        """

        dfdx[:] = 0.0

        # Set the current dv values into TACS
        self.tacs.setDesignVars(dvs)

        # March forward and evaluate functions
        self.integrator.integrate()
        #flag = self.march()
        fvals = self.integrator.evalFunctions(funcs)

        if funcOnly is not True:
            # March backward and evaluate gradient
            self.integrator.integrateAdjoint()
            self.integrator.getGradient(dfdx)

        return fvals

if __name__ == "__main__":

    """
    An example of creating TACS using TACSMeshLoader class. For
    sophisticated flexible multibody simulations use TACSBuilder class
    to create TACS.
    """
    from mpi4py import MPI
    from tacs import functions

    # Inputs
    bdffilename = 'rotor1_1dv.bdf'
    bdftype = 0 # 0: coordinate ordering, 1: GMSH ordering. This is
                # the first thing to check if you get Nan in first
                # Newton iteration

    # Specify material properties
    rho=2500.0
    E=70.0e9
    nu=0.3
    kcorr=5.0/6.0
    ys=350.0e6
    thickness=0.01
    tmin=1.0e-4
    tmax=1.0
    tdv=0

    # dynamics
    vel   = np.array([0.0,0.0,0.0], TACS.dtype)
    omega = np.array([0.0,0.0,0.0], TACS.dtype)
    grav  = np.array([0.0,0.0,-9.81], TACS.dtype)

    print("Creating instance of TACS from BDF file...", bdffilename)

    # Create an instance of TACS
    tacs = TACSBuilder.createTACSFromBDF(MPI.COMM_WORLD,
                                         bdffilename, bdftype,
                                         vel, omega, grav,
                                         rho, E, nu, kcorr, ys,
                                         thickness, tdv, tmin, tmax)

    start_time = 0.0
    end_time = 1.0
    num_steps = 1000
    order = 2
    integrator = TACS.BDFIntegrator(tacs,
                                    start_time, end_time,
                                    num_steps,
                                    order)
    # Configure output as f5 files
    flag = (TACS.ToFH5.NODES|
            TACS.ToFH5.DISPLACEMENTS|
            TACS.ToFH5.STRAINS|
            TACS.ToFH5.STRESSES|
            TACS.ToFH5.EXTRAS)
    shellf5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
    integrator.setShellOutput(shellf5)
    integrator.setOutputFrequency(1)

    # Assuming 1 variable in the BDF file
    dvs = np.array([0.015], dtype=TACS.dtype)
    nvars = dvs.size

    # Create a list of functions
    funcs = []
    funcs.append(functions.KSFailure(tacs, 100.0))
    funcs.append(functions.InducedFailure(tacs, 20.0))
    funcs.append(functions.Compliance(tacs))
    funcs.append(functions.StructuralMass(tacs))

    # Set the functions into the integrator/TACS
    integrator.setFunctions(funcs, len(dvs))

    # Make space for function values and derivatives
    funcVals = np.zeros(len(funcs), TACS.dtype)

    # Integrate forward in time and evaluate the functions on the
    # model
    integrator.integrate()
    funcVals = integrator.evalFunctions(funcs)
    print("Function values : ", np.real(funcVals))
