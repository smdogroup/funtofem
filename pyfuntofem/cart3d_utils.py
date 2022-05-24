# -------------------------------------------------
# Cart3D import, export routines needed by FUNtoFEM
# -------------------------------------------------
# Primarily written by George Anderson, modified and augmented by Jan Kiviaho

from __future__ import print_function
import numpy as np
import struct
import os

def ReadTriangulation(filepath):
    '''
    Import an unstructured surface triangulation.
    Auto-detects TRI ASCII and Binary formats
    '''
    filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        print("Filepath does not exist:", filepath)
        return

    # Try all three versions of .TRI files
    try:
        return ReadTri(filepath)
    except Exception as e_ascii:
        try:
            return ReadTriBinary(filepath)
        except Exception as e_binary:
            print("Can't read triangulation in any format:")
            print("ASCII: ", e_ascii)
            print("Binary: ", e_binary)

def ReadTri(filepath):
    '''
    Read an ASCII TRI file

    Parameters
    ----------
    filepath : string
        Input Filepath

    Returns
    -------
    verts : np.ndarray
        list of vertices
    faces : np.ndarray
        mesh connectivity
    scalars : np.ndarray
        node-centered data

    '''
    # Pull the entire file into memory
    with open(filepath,'r') as f:
        # Determine number of verts, faces and scalars from first line in file
        info = f.readline().split()
        numVerts = int(info[0])
        numFaces = int(info[1])
        numScalars = int(info[2]) if len(info) > 2 else 0

        # Read vertices
        verts = []
        for i in range(numVerts):
            vert = f.readline().split()
            verts.append((float(vert[0]),float(vert[1]),float(vert[2])))

        # Read connectivity
        faces = []
        for i in range(numFaces):
            face = f.readline().split()
            # Convert from 1-index in TRI to 0-index
            faces.append((int(face[0])-1,int(face[1])-1,int(face[2])-1))

        # Read component numbers
        comps = []
        try:
            comps = [int(f.readline()) for i in range(numFaces)]
        except:
            Warn("No components ")

        # Read scalars
        scalars = []
        if numScalars > 0:
            # Read all scalars in at once
            for line in f:
                scalars.extend([float(x) for x in line.split()])

    # Convert to numpy arrays
    verts = np.array(verts)
    faces = np.array(faces)
    comps = np.array(comps)
    scalars = np.array(scalars)
    if len(scalars) > 0:
        scalars = scalars.reshape((-1,numScalars))

    return verts, faces, comps, scalars

def ReadTriBinary(filepath):
    '''
    Read a binary TRI geometry

    Parameters
    ----------
    filepath : string
        input filepath

    Returns
    -------
    verts : np.ndarray
        list of vertices
    faces : np.ndarray
        mesh connectivity

    '''
    verts = [] # Stores vert list
    faces = [] # Stores face list
    comps = []
    scalars = []

    with open(filepath,'rb') as f: #data = f.read()
         f.read(4) # Throw away 4 bytes
         n_verts, = struct.unpack('i',f.read(4))
         n_faces, = struct.unpack('i',f.read(4))
         f.read(8) # Throw away 8 bytes (probably gives number of tags)

         for i in range(n_verts):
             x, = struct.unpack('f',f.read(4))
             y, = struct.unpack('f',f.read(4))
             z, = struct.unpack('f',f.read(4))
             verts.append([x,y,z])
         f.read(8) # Throw away 8 bytes

         for i in range(n_faces):
             v1, = struct.unpack('i',f.read(4))
             v2, = struct.unpack('i',f.read(4))
             v3, = struct.unpack('i',f.read(4))
             faces.append([v1-1,v2-1,v3-1]) # Switch to 0-index
         f.read(8) # Throw away 8 bytes
         for i in range(n_faces):
             tag, = struct.unpack('i',f.read(4))
             comps.append(tag)

    # Convert to numpy arrays
    verts = np.array(verts)
    faces = np.array(faces)
    comps = np.array(comps)
    scalars = np.array(scalars)

    return verts, faces, comps, scalars

def ComputeAeroLoads(verts, faces, scalars, pinf, gamma):
    '''
    Compute node-centered loads by integrating over dual area

    Parameters
    ----------
    verts : np.ndarray
        list of vertices
    faces : np.ndarray
        mesh connectivity
    scalars : np.ndarray
        node-centered data
    pinf : float
        freestream pressure
    gamma : float
        ratio of specific heats

    Returns
    -------
    aero_loads : np.ndarray
        node-centered loads

    '''
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    n_scalars = scalars.shape[0]

    pressures = scalars[:,5]
    aero_loads = np.zeros((n_verts, 3), dtype=np.float64)

    # Loop over the triangles
    for i in range(n_faces):
        # Extract the vertices of the triangle
        n1 = faces[i,0]
        n2 = faces[i,1]
        n3 = faces[i,2]
        v1 = verts[n1,:]
        v2 = verts[n2,:]
        v3 = verts[n3,:]

        # Compute the sides of the triangle, a and b
        a = v2 - v1
        b = v3 - v1

        # Take cross product of a and b to compute surface normal and area
        axb = np.cross(a, b)

        # Extract pressures on the vertices of the triangle
        p1 = pressures[n1]
        p2 = pressures[n2]
        p3 = pressures[n3]

        # Dimensionalize the pressures
        p1 *= gamma*pinf
        p2 *= gamma*pinf
        p3 *= gamma*pinf

        # Compute contribution to load at each node from this triangle
        f1 = -(1.0/6.0)*(p1-pinf)*axb
        f2 = -(1.0/6.0)*(p2-pinf)*axb
        f3 = -(1.0/6.0)*(p3-pinf)*axb

        # Accumulate contributions
        aero_loads[n1,:] += f1
        aero_loads[n2,:] += f2
        aero_loads[n3,:] += f3

    return aero_loads

def WriteTri(verts, faces, comps, filepath):
    '''
    Write objects to an ASCII TRI file

    Parameters
    ----------
    verts : list
        list of vertices
    faces : list 
        mesh connectivity
    comps: list
        list of which component each face belongs to
    filepath : string
        output TRI filepath (including extension)

    '''
    with open(filepath, 'w') as out:
        out.write('%i %i\n' % (len(verts), len(faces)))
        for v in verts.tolist():
            out.write('{:16.9E} {:16.9E} {:16.9E}\n'.format(v[0],v[1],v[2]))
        # Increment face indices by 1 for 1-indexing in TRI format
        for face in faces.tolist():
            out.write('%i %i %i\n' % (face[0]+1,face[1]+1,face[2]+1))
        for comp in comps.tolist():
            out.write('%i\n' % comp)

def RMS(qn, qnm1):
    """
    Compute RMS error from state at step n and state at step (n-1) as a measure
    of the convergence of the steady aeroelastic solution

    """
    return np.sqrt(np.mean(np.square(qn - qnm1)))
