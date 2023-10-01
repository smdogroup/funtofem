/*
  This file is part of the package FUNtoFEM for coupled aeroelastic simulation
  and design optimization.

  Copyright (C) 2015 Georgia Tech Research Corporation.
  Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
  All rights reserved.

  FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
  you may not use this software except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "TransferScheme.h"

#include <stdio.h>
#include <stdlib.h>

#include <cstring>

#include "LocatePoint.h"
#include "funtofemlapack.h"

// Initialize object counter to zero
int TransferScheme::object_count = 0;

TransferScheme::~TransferScheme() {
  // Free the aerodynamic data
  if (Xa) {
    delete[] Xa;
  }

  // Free the structural data
  if (Xs) {
    delete[] Xs;
  }
  if (Xs_local) {
    delete[] Xs_local;
  }
}

/*
  Set the aerodynamic surface node locations
*/
void TransferScheme::setAeroNodes(const F2FScalar *aero_X, int aero_nnodes) {
  na = aero_nnodes;

  // Free the aerodynamic data if any is allocated
  if (Xa) {
    delete[] Xa;
  }

  // Global number of aerodynamic nodes
  na_global = 0;
  MPI_Allreduce(&na, &na_global, 1, MPI_INT, MPI_SUM, global_comm);

  // Allocate memory for aerodynamic data, copy in node locations, initialize
  // displacement and load arrays
  if (na > 0) {
    Xa = new F2FScalar[3 * na];
    memcpy(Xa, aero_X, 3 * na * sizeof(F2FScalar));
  } else {
    Xa = NULL;
  }
}

/*
  Set the initial structural node locations
*/
void TransferScheme::setStructNodes(const F2FScalar *struct_X,
                                    int struct_nnodes) {
  mesh_update = 1;
  ns_local = struct_nnodes;

  // Free the structural data if any is allocated
  if (Xs_local) {
    delete[] Xs_local;
  }

  // Allocate memory for aerodynamic data, copy in node locations, initialize
  // displacement and load arrays
  if (struct_nnodes > 0) {
    Xs_local = new F2FScalar[3 * ns_local];
    memcpy(Xs_local, struct_X, 3 * ns_local * sizeof(F2FScalar));
  } else {
    Xs_local = NULL;
  }
}

void TransferScheme::distributeStructuralMesh() {
  // Check if the mesh has been updated
  MPI_Allreduce(MPI_IN_PLACE, &mesh_update, 1, MPI_INT, MPI_SUM, global_comm);

  if (mesh_update > 0) {
    mesh_update = 0;

    // Compute the number of structural nodes on the structural root processor
    if (struct_comm != MPI_COMM_NULL) {
      MPI_Reduce(&ns_local, &ns, 1, MPI_INT, MPI_SUM, 0, struct_comm);
    }

    // Broadcast the number of nodes across all the processors
    MPI_Bcast(&ns, 1, MPI_INT, struct_root, global_comm);

    // Allocate memory for structural data, initialize displacement array
    if (Xs) {
      delete[] Xs;
    }

    Xs = new F2FScalar[3 * ns];
    memset(Xs, 0, 3 * ns * sizeof(F2FScalar));

    structGatherBcast(3 * ns_local, Xs_local, 3 * ns, Xs);
  }
}

/*
  This code takes data in an array that is duplicated across aero_comm, adds
  all the values so that they are consistent and distributes the added values
  to processors on aero_comm.

  Here struct_comm and aero_comm may be overlapping or non-overlapping
  communicators on global_comm. The ranks struct_root and aero_root are the
  ranks in global_comm of the rank-0 processors in struct_comm and aero_comm,
  respectively.

  Here we assume that the struct_data is of length struct_len and is a global
  duplicated array across all struct_comm processors. This data is first summed
  via MPI_Reduce to the rank-0 struct processor.

  Arguments
  ---------
  global_len   : length of structural data (same on all struct_comm)
  global_data  : array of the structural data (to be added together)
  local_len    : local length of distributed structural data

  Output
  ------
  local_data   : local added/scattered data
*/
void TransferScheme::structAddScatter(int global_len, F2FScalar *global_data,
                                      int local_len, F2FScalar *local_data) {
  // Reduce values on global_comm to the struct_root processor
  int global_rank;
  MPI_Comm_rank(global_comm, &global_rank);

  if (global_rank == struct_root) {
    MPI_Reduce(MPI_IN_PLACE, global_data, global_len, F2F_MPI_TYPE, MPI_SUM,
               struct_root, global_comm);
  } else {
    MPI_Reduce(global_data, NULL, global_len, F2F_MPI_TYPE, MPI_SUM,
               struct_root, global_comm);
  }

  // Scatter the values across all aero_comm processors
  if (struct_comm != MPI_COMM_NULL) {
    // Gather how many values each processor has
    int struct_nprocs, struct_rank;
    MPI_Comm_size(struct_comm, &struct_nprocs);
    MPI_Comm_rank(struct_comm, &struct_rank);

    int *nvalues = new int[struct_nprocs];
    memset(nvalues, 0, struct_nprocs * sizeof(int));

    MPI_Gather(&local_len, 1, MPI_INT, nvalues, 1, MPI_INT, 0, struct_comm);

    // Collect the values nodes on the root
    int *disps = new int[struct_nprocs];
    memset(disps, 0, struct_nprocs * sizeof(int));

    if (struct_rank == 0) {
      for (int proc = 0; proc < struct_nprocs; proc++) {
        if (proc > 0) {
          disps[proc] = disps[proc - 1] + nvalues[proc - 1];
        }
      }
    }

    // Scatter from the root aero processor
    MPI_Scatterv(global_data, nvalues, disps, F2F_MPI_TYPE, local_data,
                 local_len, F2F_MPI_TYPE, 0, struct_comm);

    delete[] nvalues;
    delete[] disps;
  }
}

/*
  This code takes distributed structural data and broadcasts it to all
  processors in global_comm.

  Arguments
  ---------
  local_len    : local length of distributed structural data
  local_data   : local distributed array of data
  global_len   : global length of the data

  Output
  ------
  global_data  : global array of the structural data
*/
void TransferScheme::structGatherBcast(int local_len,
                                       const F2FScalar *local_data,
                                       int global_len, F2FScalar *global_data) {
  // Collect how many structural nodes every processor has
  if (struct_comm != MPI_COMM_NULL) {
    int struct_nprocs, struct_rank;
    MPI_Comm_size(struct_comm, &struct_nprocs);
    MPI_Comm_rank(struct_comm, &struct_rank);

    int *nvalues = new int[struct_nprocs];
    memset(nvalues, 0, struct_nprocs * sizeof(int));

    MPI_Gather(&local_len, 1, MPI_INT, nvalues, 1, MPI_INT, 0, struct_comm);

    // Collect the values nodes on the root
    int *disps = new int[struct_nprocs];
    memset(disps, 0, struct_nprocs * sizeof(int));

    if (struct_rank == 0) {
      for (int proc = 0; proc < struct_nprocs; proc++) {
        if (proc > 0) {
          disps[proc] = disps[proc - 1] + nvalues[proc - 1];
        }
      }
    }

    MPI_Gatherv(local_data, local_len, F2F_MPI_TYPE, global_data, nvalues,
                disps, F2F_MPI_TYPE, 0, struct_comm);

    delete[] nvalues;
    delete[] disps;
  }

  // Broadcast the global list to all the processors
  MPI_Bcast(global_data, global_len, F2F_MPI_TYPE, struct_root, global_comm);
}

/*
  Reduce vector to get the total across all aero procs then distribute to the
  aerodynamic processors
*/
void TransferScheme::aeroScatter(int global_len, F2FScalar *global_data,
                                 int local_len, F2FScalar *local_data) {
  if (aero_comm != MPI_COMM_NULL) {
    // Collect how many nodes each aerodynamic processor has
    int aero_nprocs, aero_rank;
    MPI_Comm_size(aero_comm, &aero_nprocs);
    MPI_Comm_rank(aero_comm, &aero_rank);

    int *nvalues = new int[aero_nprocs];
    memset(nvalues, 0, aero_nprocs * sizeof(int));

    MPI_Gather(&local_len, 1, MPI_INT, nvalues, 1, MPI_INT, 0, aero_comm);

    // Collect the values nodes on the root
    int *disps = new int[aero_nprocs];
    memset(disps, 0, aero_nprocs * sizeof(int));

    if (aero_rank == 0) {
      for (int proc = 0; proc < aero_nprocs; proc++) {
        if (proc > 0) {
          disps[proc] = disps[proc - 1] + nvalues[proc - 1];
        }
      }
    }

    MPI_Scatterv(global_data, nvalues, disps, F2F_MPI_TYPE, local_data,
                 local_len, F2F_MPI_TYPE, aero_root, aero_comm);

    delete[] nvalues;
    delete[] disps;
  }
}

/*
  Collect a aerodynamic vector to create a global image then distribute to the
  aerodynamic processors
*/
void TransferScheme::aeroGatherBcast(int local_len, const F2FScalar *local_data,
                                     int global_len, F2FScalar *global_data) {
  // Collect how many structural nodes every processor has
  if (aero_comm != MPI_COMM_NULL) {
    int aero_nprocs, aero_rank;
    MPI_Comm_size(aero_comm, &aero_nprocs);
    MPI_Comm_rank(aero_comm, &aero_rank);

    int *nvalues = new int[aero_nprocs];
    memset(nvalues, 0, aero_nprocs * sizeof(int));

    MPI_Gather(&local_len, 1, MPI_INT, nvalues, 1, MPI_INT, 0, struct_comm);

    // Collect the values nodes on the root
    int *disps = new int[aero_nprocs];
    memset(disps, 0, aero_nprocs * sizeof(int));

    if (aero_rank == 0) {
      for (int proc = 0; proc < aero_nprocs; proc++) {
        if (proc > 0) {
          disps[proc] = disps[proc - 1] + nvalues[proc - 1];
        }
      }
    }

    MPI_Gatherv(local_data, local_len, F2F_MPI_TYPE, global_data, nvalues,
                disps, F2F_MPI_TYPE, 0, aero_comm);

    delete[] nvalues;
    delete[] disps;
  }

  // Broadcast the global list to all the processors
  MPI_Bcast(global_data, global_len, F2F_MPI_TYPE, aero_root, global_comm);
}

/*
  Builds aerostructural connectivity through LocatePoint search, linking each
  aerodynamic node with a specified number of nearest structural nodes

  Arguments
  ---------
  isymm  : Symmetry index
  nn     : The number of nearest neighbors
  tol    : Symmetry plane tolerance

  Returns
  -------
  conn   : aerostructural connectivity
*/
void TransferScheme::computeAeroStructConn(int isymm, int nn, int *conn,
                                           double tol) {
  // Copy or duplicate and reflect the unique structural nodes
  F2FScalar *Xs_dup = NULL;
  int num_locate_nodes = 0;
  int *locate_to_reflected_index = NULL;

  if (isymm >= 0) {
    Xs_dup = new F2FScalar[6 * ns];
    memcpy(Xs_dup, Xs, 3 * ns * sizeof(F2FScalar));

    locate_to_reflected_index = new int[ns];

    num_locate_nodes = 0;
    for (int k = 0; k < ns; k++) {
      if (fabs(F2FRealPart(Xs_dup[3 * k + isymm])) > tol) {
        // Node is not on the plane of symmetry, so copy it
        int kk = num_locate_nodes + ns;
        memcpy(&Xs_dup[3 * kk], &Xs_dup[3 * k], 3 * sizeof(F2FScalar));

        // reflect the point
        Xs_dup[3 * kk + isymm] *= -1.0;

        // Record the index
        locate_to_reflected_index[num_locate_nodes] = k + ns;
        num_locate_nodes++;
      }
    }
    num_locate_nodes += ns;
  } else {
    Xs_dup = new F2FScalar[3 * ns];
    memcpy(Xs_dup, Xs, 3 * ns * sizeof(F2FScalar));
    num_locate_nodes = ns;
  }

  // Create instance of LocatePoint class to perform the following searches
  int min_bin_size = 10;
  LocatePoint *locator =
      new LocatePoint(Xs_dup, num_locate_nodes, min_bin_size);

  // Indices of nearest nodes
  int *indx = new int[nn];
  F2FScalar *dist = new F2FScalar[nn];

  // For each aerodynamic node, copy the indices of the nearest n structural
  // nodes into the conn array
  for (int i = 0; i < na; i++) {
    F2FScalar xa0[3];
    memcpy(xa0, &Xa[3 * i], 3 * sizeof(F2FScalar));

    locator->locateKClosest(nn, indx, dist, xa0);

    for (int k = 0; k < nn; k++) {
      if (indx[k] >= ns) {
        int kk = indx[k] - ns;
        conn[nn * i + k] = locate_to_reflected_index[kk];
      } else {
        conn[nn * i + k] = indx[k];
      }
    }
  }

  // Free the duplicate array
  delete[] Xs_dup;

  if (locate_to_reflected_index) {
    delete[] locate_to_reflected_index;
  }

  // Delete the LocatePoint object and release memory
  delete[] indx;
  delete[] dist;
  delete locator;
}

/*
  Computes weights of structural nodes

  Arguments
  ---------
  beta   : Exponential factor
  isymm  : Symmetry index
  nn     : The number of nearest neighbors
  conn   : Connectivity array aero nodes -> structural nodes
  tol    : Symmetry plane tolerance

  Returns
  --------
  W : weights
*/
void TransferScheme::computeWeights(double beta, int isymm, int nn,
                                    const int *conn, F2FScalar *W, double tol) {
  for (int i = 0; i < na; i++) {
    const F2FScalar *xa0 = &Xa[3 * i];
    const int *local_conn = &conn[i * nn];
    F2FScalar *w = &W[i * nn];

    // Compute the weights based on the difference between the aerodynamic
    // point location and the local undeformed structural point
    F2FScalar wtotal = 0.0;
    F2FScalar v[3];

    // Find the average squared distance for normalization
    F2FScalar v2_avg = 0.0;

    for (int j = 0; j < nn; j++) {
      if (local_conn[j] < ns) {
        const F2FScalar *xs0 = &Xs[3 * local_conn[j]];
        vec_diff(xs0, xa0, v);
      } else {
        F2FScalar rxs0[3];  // Reflected xs0
        memcpy(rxs0, &Xs[3 * (local_conn[j] - ns)], 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(rxs0, xa0, v);
      }
      v2_avg += vec_dot(v, v) / (1.0 * nn);
    }

    // Make sure we don't divide by zero
    if (F2FRealPart(v2_avg) < tol) {
      v2_avg = tol;
    }

    for (int j = 0; j < nn; j++) {
      if (local_conn[j] < ns) {
        const F2FScalar *xs0 = &Xs[3 * local_conn[j]];
        vec_diff(xs0, xa0, v);
      } else {
        F2FScalar rxs0[3];  // Reflected xs0
        memcpy(rxs0, &Xs[3 * (local_conn[j] - ns)], 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(rxs0, xa0, v);
      }
      w[j] = exp(-beta * vec_dot(v, v) / v2_avg);
      wtotal += w[j];
    }

    // Normalize the weights
    wtotal = 1.0 / wtotal;
    for (int j = 0; j < nn; j++) {
      w[j] *= wtotal;
    }
  }
}

/*
  Transform a set of aerodynamic surface displacements into a least-squares
  fit of rotation and translation plus elastic deformations

  Arguments
  ----------
  aero_disps : aerodynamic surface node displacements

  Returns
  -------
  R          : rotation matrix
  t          : translation
  u          : elastic deformations
*/
void LDTransferScheme::transformEquivRigidMotion(const F2FScalar *aero_disps,
                                                 F2FScalar *R, F2FScalar *t,
                                                 F2FScalar *u) {
  // Gather aerodynamic node locations and displacements
  F2FScalar *Xa_global = new F2FScalar[3 * na_global];
  aeroGatherBcast(3 * na, Xa, 3 * na_global, Xa_global);

  F2FScalar *Ua_global = new F2FScalar[3 * na_global];
  aeroGatherBcast(3 * na, aero_disps, 3 * na_global, Ua_global);

  // Adds aerodynamic displacements to aerodynamic node locations
  F2FScalar *Xad_global = new F2FScalar[3 * na_global];
  for (int i = 0; i < 3 * na_global; i++) {
    Xad_global[i] = Xa_global[i] + Ua_global[i];
  }

  // Allocate global array of elastic displacements
  F2FScalar *u_global = new F2FScalar[3 * na_global];

  // Compute centroids of the original and displaced node locations
  F2FScalar x0_bar[3];
  memset(x0_bar, 0.0, 3 * sizeof(F2FScalar));
  F2FScalar x_bar[3];
  memset(x_bar, 0.0, 3 * sizeof(F2FScalar));

  for (int j = 0; j < na_global; j++) {
    x0_bar[0] += Xa_global[3 * j + 0];
    x0_bar[1] += Xa_global[3 * j + 1];
    x0_bar[2] += Xa_global[3 * j + 2];
    x_bar[0] += Xad_global[3 * j + 0];
    x_bar[1] += Xad_global[3 * j + 1];
    x_bar[2] += Xad_global[3 * j + 2];
  }

  x0_bar[0] *= 1.0 / na_global;
  x0_bar[1] *= 1.0 / na_global;
  x0_bar[2] *= 1.0 / na_global;
  x_bar[0] *= 1.0 / na_global;
  x_bar[1] *= 1.0 / na_global;
  x_bar[2] *= 1.0 / na_global;

  // Form the covariance matrix
  F2FScalar H[9];
  memset(H, 0.0, 9 * sizeof(F2FScalar));

  for (int j = 0; j < na_global; j++) {
    F2FScalar q[3];
    q[0] = Xa_global[3 * j + 0] - x0_bar[0];
    q[1] = Xa_global[3 * j + 1] - x0_bar[1];
    q[2] = Xa_global[3 * j + 2] - x0_bar[2];

    F2FScalar p[3];
    p[0] = Xad_global[3 * j + 0] - x_bar[0];
    p[1] = Xad_global[3 * j + 1] - x_bar[1];
    p[2] = Xad_global[3 * j + 2] - x_bar[2];

    H[0] += p[0] * q[0];
    H[1] += p[1] * q[0];
    H[2] += p[2] * q[0];
    H[3] += p[0] * q[1];
    H[4] += p[1] * q[1];
    H[5] += p[2] * q[1];
    H[6] += p[0] * q[2];
    H[7] += p[1] * q[2];
    H[8] += p[2] * q[2];
  }

  for (int k = 0; k < 9; k++) {
    H[k] *= 1.0 / na_global;
  }

  // Compute rotation matrix
  computeRotation(H, R, Saero);

  // Compute translation
  t[0] = x_bar[0] - R[0] * x0_bar[0] - R[3] * x0_bar[1] - R[6] * x0_bar[2];
  t[1] = x_bar[1] - R[1] * x0_bar[0] - R[4] * x0_bar[1] - R[7] * x0_bar[2];
  t[2] = x_bar[2] - R[2] * x0_bar[0] - R[5] * x0_bar[1] - R[8] * x0_bar[2];

  // Compute elastic deformations (deviation from rigid motion)
  for (int j = 0; j < na_global; j++) {
    F2FScalar Xa_rigid[] = {t[0], t[1], t[2]};
    F2FScalar *x = &Xa_global[3 * j];
    Xa_rigid[0] += R[0] * x[0] + R[3] * x[1] + R[6] * x[2];
    Xa_rigid[1] += R[1] * x[0] + R[4] * x[1] + R[7] * x[2];
    Xa_rigid[2] += R[2] * x[0] + R[5] * x[1] + R[8] * x[2];
    u_global[3 * j + 0] = Xad_global[3 * j + 0] - Xa_rigid[0];
    u_global[3 * j + 1] = Xad_global[3 * j + 1] - Xa_rigid[1];
    u_global[3 * j + 2] = Xad_global[3 * j + 2] - Xa_rigid[2];
  }

  // Copy rotation matrix and centroid to global variables for use in
  // computing derivatives
  memcpy(Raero, R, 9 * sizeof(F2FScalar));
  memcpy(xa0bar, x0_bar, 3 * sizeof(F2FScalar));
  memcpy(xabar, x_bar, 3 * sizeof(F2FScalar));

  // Scatter elastic deformations to aerodynamic processors
  aeroScatter(3 * na_global, u_global, na, u);

  // Free memory
  delete[] Xa_global;
  delete[] Ua_global;
  delete[] Xad_global;
  delete[] u_global;
}

/*
  Apply the action of the rigid transformation w.r.t aerodynamic surface node
  displacements Jacobian to the input vector

  Arguments
  ---------
  vecs       : input vector

  Returns
  -------
  prods      : output vectors
*/
void LDTransferScheme::applydRduATrans(const F2FScalar *vecs,
                                       F2FScalar *prods) {
  // Compute the rotation sensitivities
  F2FScalar dR[9 * 9];

  F2FScalar M1[15 * 15];
  assembleM1(Raero, Saero, M1);
  int ipiv[15], m = 15, info = 0;
  LAPACKgetrf(&m, &m, M1, &m, ipiv, &info);

  for (int k = 0; k < 9; k++) {
    // Solve system for each component of R
    F2FScalar x[15];
    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[k] = -1.0;
    int nrhs = 1;
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    memcpy(&dR[9 * k], x, 9 * sizeof(F2FScalar));
  }

  // Decompose input adjoint vector into rotation and translation parts
  F2FScalar psi_R[9];
  memcpy(psi_R, vecs, 9 * sizeof(F2FScalar));
  F2FScalar psi_t[3];
  memcpy(psi_t, &vecs[9], 3 * sizeof(F2FScalar));

  // Compute X^{T} = d(psi_{t}^{T}*R*xA0bar)/dH
  F2FScalar X[9];
  memset(X, 0.0, 9 * sizeof(F2FScalar));
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          X[i + 3 * j] +=
              psi_t[m] * dR[(i + 3 * j) + 9 * (m + 3 * n)] * xa0bar[n];
        }
      }
    }
  }

  // Compute the weight (simply 1/na)
  F2FScalar w = 1.0 / na_global;

  // For each aero node, the product consists of rotation and translation
  // contributions
  for (int i = 0; i < na; i++) {
    F2FScalar *prod = &prods[3 * i];

    // Compute the translation contribution
    // tcont = -w*(psi_{t} - X^{T}*q)
    F2FScalar tcont[3];

    const F2FScalar *xa0 = &Xa[3 * i];
    F2FScalar q[3];
    vec_diff(xa0bar, xa0, q);

    tcont[0] = -w * (psi_t[0] - X[0] * q[0] - X[1] * q[1] - X[2] * q[2]);
    tcont[1] = -w * (psi_t[1] - X[3] * q[0] - X[4] * q[1] - X[5] * q[2]);
    tcont[2] = -w * (psi_t[2] - X[6] * q[0] - X[7] * q[1] - X[8] * q[2]);

    // Compute the rotation contribution
    F2FScalar rcont[3];
    memset(rcont, 0.0, 3 * sizeof(F2FScalar));

    for (int m = 0; m < 3; m++) {
      for (int n = 0; n < 3; n++) {
        for (int k = 0; k < 3; k++) {
          for (int j = 0; j < 3; j++) {
            rcont[k] -=
                w * dR[(j + 3 * k) + 9 * (m + 3 * n)] * q[j] * psi_R[m + 3 * n];
          }
        }
      }
    }

    // Add the translation and rotation contributions into prod
    prod[0] = rcont[0] + tcont[0];
    prod[1] = rcont[1] + tcont[1];
    prod[2] = rcont[2] + tcont[2];
  }
}

/*
  Apply the action of the rigid transformation w.r.t initial aerodynamic
  surface node locations Jacobian to the right of the transposed input vector

  psi_{R} is 9 x 1
  psi_{t} is 3 x 1
  dR/dxA0 is 9 x (3 x na)
  dt/dxA0 is 3 x (3 x na)

  [psi_{R}^{T} psi_{t}^{T}][[dR/dxA0]] is 1 x(3 x na)
                           [[dt/dxA0]]

  Arguments
  ----------
  aero_disps : aerodynamic surface node displacements
  vecs       : input vector

  Returns
  --------
  prods      : output vector

*/
void LDTransferScheme::applydRdxA0Trans(const F2FScalar *aero_disps,
                                        const F2FScalar *vecs,
                                        F2FScalar *prods) {
  // Compute the rotation sensitivities
  F2FScalar dR[9 * 9];

  F2FScalar M1[15 * 15];
  assembleM1(Raero, Saero, M1);
  int ipiv[15], m = 15, info = 0;
  LAPACKgetrf(&m, &m, M1, &m, ipiv, &info);

  for (int k = 0; k < 9; k++) {
    // Solve system for each component of R
    F2FScalar x[15];
    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[k] = -1.0;
    int nrhs = 1;
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    memcpy(&dR[9 * k], x, 9 * sizeof(F2FScalar));
  }

  // Decompose input adjoint vector into rotation and translation parts
  F2FScalar psi_R[9];
  memcpy(psi_R, vecs, 9 * sizeof(F2FScalar));
  F2FScalar psi_t[3];
  memcpy(psi_t, &vecs[9], 3 * sizeof(F2FScalar));

  // Compute X^{T} = d(psi_{t}^{T}*R*xA0bar)/dH
  F2FScalar X[9];
  memset(X, 0.0, 9 * sizeof(F2FScalar));
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          X[i + 3 * j] +=
              psi_t[m] * dR[(i + 3 * j) + 9 * (m + 3 * n)] * xa0bar[n];
        }
      }
    }
  }

  // Compute the weight (simply 1/na)
  F2FScalar w = 1.0 / na_global;

  // For each aero node, the product consists of rotation and translation
  // contributions
  for (int i = 0; i < na; i++) {
    F2FScalar *prod = &prods[3 * i];

    // Compute the translation contribution
    // tcont = w*(X^{T}*q + X*p + psi_{t}^{T}*(R - I))
    F2FScalar tcont[3];
    memset(tcont, 0.0, 3 * sizeof(F2FScalar));

    const F2FScalar *xa0 = &Xa[3 * i];
    F2FScalar q[3];
    vec_diff(xa0bar, xa0, q);

    tcont[0] += w * (X[0] * q[0] + X[1] * q[1] + X[2] * q[2]);
    tcont[1] += w * (X[3] * q[0] + X[4] * q[1] + X[5] * q[2]);
    tcont[2] += w * (X[6] * q[0] + X[7] * q[1] + X[8] * q[2]);

    const F2FScalar *ua = &aero_disps[3 * i];
    F2FScalar p[3];
    p[0] = xa0[0] + ua[0] - xabar[0];
    p[1] = xa0[1] + ua[1] - xabar[1];
    p[2] = xa0[2] + ua[2] - xabar[2];

    tcont[0] += w * (X[0] * p[0] + X[3] * p[1] + X[6] * p[2]);
    tcont[1] += w * (X[1] * p[0] + X[4] * p[1] + X[7] * p[2]);
    tcont[2] += w * (X[2] * p[0] + X[5] * p[1] + X[8] * p[2]);

    tcont[0] += w * (psi_t[0] * (Raero[0] - 1.0) + psi_t[1] * Raero[1] +
                     psi_t[2] * Raero[2]);
    tcont[1] += w * (psi_t[0] * Raero[3] + psi_t[1] * (Raero[4] - 1.0) +
                     psi_t[2] * Raero[5]);
    tcont[2] += w * (psi_t[0] * Raero[6] + psi_t[1] * Raero[7] +
                     psi_t[2] * (Raero[8] - 1.0));

    // Compute the rotation contribution
    F2FScalar rcont[3];
    memset(rcont, 0.0, 3 * sizeof(F2FScalar));

    for (int m = 0; m < 3; m++) {
      for (int n = 0; n < 3; n++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            rcont[k] -= w * psi_R[m + 3 * n] *
                        (dR[(j + 3 * k) + 9 * (m + 3 * n)] * q[j] +
                         dR[(k + 3 * j) + 9 * (m + 3 * n)] * p[j]);
          }
        }
      }
    }

    // Add the translation and rotation contributions into prod
    prod[0] = rcont[0] + tcont[0];
    prod[1] = rcont[1] + tcont[1];
    prod[2] = rcont[2] + tcont[2];
  }
}

/*
  Run all of the tests for the transfer scheme.
*/
int LDTransferScheme::testAllDerivatives(const F2FScalar *struct_disps,
                                         const F2FScalar *aero_loads,
                                         const F2FScalar h, const double rtol,
                                         const double atol) {
  const int aero_nnodes = getNumLocalAeroNodes();
  const int struct_nnodes = getNumLocalStructNodes();
  const int dof_per_node = getStructNodeDof();

  // Create random testing data
  F2FScalar *test_vec_a1 = new F2FScalar[3 * aero_nnodes];
  F2FScalar *test_vec_a2 = new F2FScalar[3 * aero_nnodes];
  for (int i = 0; i < 3 * aero_nnodes; i++) {
    test_vec_a1[i] = (1.0 * rand()) / RAND_MAX;
    test_vec_a2[i] = (1.0 * rand()) / RAND_MAX;
  }

  F2FScalar *uS_pert = new F2FScalar[dof_per_node * struct_nnodes];
  F2FScalar *test_vec_s1 = new F2FScalar[dof_per_node * struct_nnodes];
  F2FScalar *test_vec_s2 = new F2FScalar[dof_per_node * struct_nnodes];
  for (int j = 0; j < dof_per_node * struct_nnodes; j++) {
    uS_pert[j] = (1.0 * rand()) / RAND_MAX;
    test_vec_s1[j] = (1.0 * rand()) / RAND_MAX;
    test_vec_s2[j] = (1.0 * rand()) / RAND_MAX;
  }

  int fail1 =
      testLoadTransfer(struct_disps, aero_loads, uS_pert, h, rtol, atol);
  int fail2 = testDispJacVecProducts(struct_disps, test_vec_a1, test_vec_s1, h,
                                     rtol, atol);
  int fail3 = testLoadJacVecProducts(struct_disps, aero_loads, test_vec_s1,
                                     test_vec_s2, h, rtol, atol);
  int fail4 =
      testdDdxA0Products(struct_disps, test_vec_a1, test_vec_a2, h, rtol, atol);
  int fail5 =
      testdDdxS0Products(struct_disps, test_vec_a1, test_vec_s1, h, rtol, atol);
  int fail6 = testdLdxA0Products(struct_disps, aero_loads, test_vec_a1,
                                 test_vec_s1, h, rtol, atol);
  int fail7 = testdLdxS0Products(struct_disps, aero_loads, test_vec_s1,
                                 test_vec_s2, h, rtol, atol);

  int fail = (fail1 || fail2 || fail3 || fail4 || fail5 || fail6 || fail7);

  delete[] uS_pert;
  delete[] test_vec_a1;
  delete[] test_vec_a2;
  delete[] test_vec_s1;
  delete[] test_vec_s2;

  return fail;
}

/*
  Tests load transfer by computing derivative of product of loads on and
  displacements of aerodynamic surface nodes with respect to structural node
  displacements and comparing with results from finite difference and complex
  step approximation

  Arguments
  ---------
  struct_disps : structural node displacements
  aero_loads   : loads on aerodynamic surface nodes
  pert         : direction of perturbation of structural node displacements
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int LDTransferScheme::testLoadTransfer(const F2FScalar *struct_disps,
                                       const F2FScalar *aero_loads,
                                       const F2FScalar *pert, const F2FScalar h,
                                       const double rtol, const double atol) {
  const int dof_per_node = getStructNodeDof();

  // Transfer the structural displacements
  F2FScalar *aero_disps = new F2FScalar[3 * na];
  transferDisps(struct_disps, aero_disps);

  // Transfer the aerodynamic loads
  F2FScalar *struct_loads = new F2FScalar[dof_per_node * ns_local];
  transferLoads(aero_loads, struct_loads);

  // Compute directional derivative (structural loads times perturbation)
  F2FScalar deriv = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    deriv += struct_loads[j] * pert[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *Us_cs = new F2FScalar[dof_per_node * ns_local];
  F2FScalar *Ua_cs = new F2FScalar[3 * na];

  for (int j = 0; j < dof_per_node * ns_local; j++) {
    Us_cs[j] =
        struct_disps[j] + F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(pert[j]));
  }
  transferDisps(Us_cs, Ua_cs);

  F2FScalar work = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    work += Fa[i] * Ua_cs[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &work, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = F2FImagPart(work) / h;

  delete[] Us_cs;
  delete[] Ua_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *Us_pos = new F2FScalar[dof_per_node * ns_local];
  F2FScalar *Us_neg = new F2FScalar[dof_per_node * ns_local];
  F2FScalar *Ua_pos = new F2FScalar[3 * na];
  F2FScalar *Ua_neg = new F2FScalar[3 * na];
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    Us_pos[j] = struct_disps[j] + h * pert[j];
    Us_neg[j] = struct_disps[j] - h * pert[j];
  }

  transferDisps(Us_pos, Ua_pos);
  F2FScalar work_pos = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    work_pos += Fa[i] * Ua_pos[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &work_pos, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  transferDisps(Us_neg, Ua_neg);
  F2FScalar work_neg = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    work_neg += Fa[i] * Ua_neg[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &work_neg, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  F2FScalar deriv_approx = 0.5 * (work_pos - work_neg) / h;

  delete[] Us_pos;
  delete[] Us_neg;
  delete[] Ua_pos;
  delete[] Ua_neg;
#endif  // FUNTOFEM_USE_COMPLEX
  // Compute relative error
  double rel_error = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error = F2FRealPart((deriv - deriv_approx) / deriv_approx);
  }
  double abs_error = F2FRealPart(deriv - deriv_approx);

  // Print results
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    printf("\n");
    printf("Load transfer test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error);
    printf("\n");
  }

  // Free allocated memory
  delete[] aero_disps;
  delete[] struct_loads;

  // Check if the test failed
  int fail = 0;
  if (fabs(rel_error) >= rtol && fabs(abs_error) >= atol) {
    fail = 1;
  }

  return fail;
}

/*
  Tests output of dDduSProducts and dDduSTransProducts by computing a product
  test_vec_a*J*test_vec_s (where J is the Jacobian) and comparing with results
  from finite difference and complex step

  Arguments
  ---------
  struct_disps : structural node displacements
  test_vec_a   : test vector the length of the aero nodes
  test_vec_s   : test vector the length of the struct nodes
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int LDTransferScheme::testDispJacVecProducts(const F2FScalar *struct_disps,
                                             const F2FScalar *test_vec_a,
                                             const F2FScalar *test_vec_s,
                                             const F2FScalar h,
                                             const double rtol,
                                             const double atol) {
  const int dof_per_node = getStructNodeDof();

  // Transfer the structural displacements to get rotations
  F2FScalar *aero_disps = new F2FScalar[3 * na];
  transferDisps(struct_disps, aero_disps);

  // Compute the Jacobian-vector products using the function
  F2FScalar *grad1 = new F2FScalar[3 * na];
  applydDduS(test_vec_s, grad1);

  // Compute product of test_vec_a with the Jacobian-vector products
  F2FScalar deriv1 = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    deriv1 += test_vec_a[i] * grad1[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv1, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Compute the transpose Jacobian-vector products using the function
  F2FScalar *grad2 = new F2FScalar[dof_per_node * ns_local];
  applydDduSTrans(test_vec_a, grad2);

  // Compute product of V1 with the transpose Jacobian-vector products
  F2FScalar deriv2 = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    deriv2 += test_vec_s[j] * grad2[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv2, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Compute complex step approximation
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *Us_cs = new F2FScalar[dof_per_node * ns_local];
  memset(Us_cs, 0.0, dof_per_node * ns_local * sizeof(F2FScalar));
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    Us_cs[j] +=
        struct_disps[j] + F2FScalar(0.0, F2FRealPart(h * test_vec_s[j]));
  }
  transferDisps(Us_cs, aero_disps);

  F2FScalar VPsi = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    F2FScalar Psi = Xa[i] + aero_disps[i];
    VPsi += test_vec_a[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPsi, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = -1.0 * F2FImagPart(VPsi) / h;

  delete[] Us_cs;

  // Compute finite difference approximation (central)
#else
  F2FScalar *Us_pos = new F2FScalar[dof_per_node * ns_local];
  F2FScalar *Us_neg = new F2FScalar[dof_per_node * ns_local];
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    Us_pos[j] = struct_disps[j] + h * test_vec_s[j];
    Us_neg[j] = struct_disps[j] - h * test_vec_s[j];
  }

  transferDisps(Us_pos, aero_disps);
  F2FScalar VPsi_pos = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    F2FScalar Psi = Xa[i] + aero_disps[i];
    VPsi_pos += test_vec_a[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPsi_pos, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  transferDisps(Us_neg, aero_disps);
  F2FScalar VPsi_neg = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    F2FScalar Psi = Xa[i] + aero_disps[i];
    VPsi_neg += test_vec_a[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPsi_neg, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  F2FScalar deriv_approx = -0.5 * (VPsi_pos - VPsi_neg) / h;

  delete[] Us_pos;
  delete[] Us_neg;
#endif
  // Compute relative error
  double rel_error1 = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error1 = F2FRealPart((deriv1 - deriv_approx) / deriv_approx);
  }
  double rel_error2 = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error2 = F2FRealPart((deriv2 - deriv_approx) / deriv_approx);
  }
  double abs_error1 = F2FRealPart(deriv1 - deriv_approx);
  double abs_error2 = F2FRealPart(deriv2 - deriv_approx);

  // Print out results of test
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    printf("V2^{T}*dD/du_{S}*V1 test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv1));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error1);
    printf("\n");

    printf("V1^{T}*(dD/du_{S})^{T}*V2 test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv2));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error2);
    printf("\n");
  }

  // Free allocated memory
  delete[] aero_disps;
  delete[] grad1;
  delete[] grad2;

  int fail = 0;
  if (fabs(rel_error1) >= rtol && fabs(abs_error1) >= rtol) {
    fail = 1;
  }
  if (fabs(rel_error2) >= rtol && fabs(abs_error2) >= rtol) {
    fail = 1;
  }

  if (rank == 0 && fail) {
    printf("testDispJacVecProducts failed\n");
  }

  return fail;
}

/*
  Tests output of dLduSProducts and dLduSTransProducts by computing a product
  test_vec_s2*J*test_vec_s1 (where J is the Jacobian) and comparing with
  results from finite difference and complex step

  Arguments
  ---------
  struct_disps : structural node displacements
  aero_loads   : aerodynamic loads
  test_vec_s1  : test vector the size of struct nodes
  test_vec_s2  : test vector the size of struct nodes
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int LDTransferScheme::testLoadJacVecProducts(
    const F2FScalar *struct_disps, const F2FScalar *aero_loads,
    const F2FScalar *test_vec_s1, const F2FScalar *test_vec_s2,
    const F2FScalar h, const double rtol, const double atol) {
  const int dof_per_node = getStructNodeDof();

  // Transfer the structural displacements
  F2FScalar *aero_disps = new F2FScalar[3 * na];
  transferDisps(struct_disps, aero_disps);

  // Transfer the aerodynamic loads to get MM, IPIV
  F2FScalar *struct_loads = new F2FScalar[dof_per_node * ns_local];
  transferLoads(aero_loads, struct_loads);

  // Compute the Jacobian-vector products using the function
  F2FScalar *grad1 = new F2FScalar[dof_per_node * ns_local];
  applydLduS(test_vec_s1, grad1);

  // Compute directional derivative
  F2FScalar deriv1 = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    deriv1 += grad1[j] * test_vec_s2[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv1, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Compute transpose Jacobian-vector products using the function
  F2FScalar *grad2 = new F2FScalar[dof_per_node * ns_local];
  applydLduSTrans(test_vec_s2, grad2);

  // Compute directional derivative
  F2FScalar deriv2 = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    deriv2 += grad2[j] * test_vec_s1[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv2, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *Us_cs = new F2FScalar[dof_per_node * ns_local];

  for (int j = 0; j < dof_per_node * ns_local; j++) {
    Us_cs[j] = struct_disps[j] +
               F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(test_vec_s1[j]));
  }
  transferDisps(Us_cs, aero_disps);
  transferLoads(aero_loads, struct_loads);

  F2FScalar VPhi = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    F2FScalar Phi = struct_loads[j];
    VPhi += test_vec_s2[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPhi, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = -1.0 * F2FImagPart(VPhi) / h;

  delete[] Us_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *Us_pos = new F2FScalar[dof_per_node * ns_local];
  F2FScalar *Us_neg = new F2FScalar[dof_per_node * ns_local];
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    Us_pos[j] = struct_disps[j] + h * test_vec_s1[j];
    Us_neg[j] = struct_disps[j] - h * test_vec_s1[j];
  }

  transferDisps(Us_pos, aero_disps);
  transferLoads(aero_loads, struct_loads);
  F2FScalar VPhi_pos = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    F2FScalar Phi = struct_loads[j];
    VPhi_pos += test_vec_s2[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPhi_pos, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  transferDisps(Us_neg, aero_disps);
  transferLoads(aero_loads, struct_loads);
  F2FScalar VPhi_neg = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    F2FScalar Phi = struct_loads[j];
    VPhi_neg += test_vec_s2[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPhi_neg, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  F2FScalar deriv_approx = -0.5 * (VPhi_pos - VPhi_neg) / h;

  delete[] Us_pos;
  delete[] Us_neg;
#endif
  // Compute relative error
  double rel_error1 = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error1 = F2FRealPart((deriv1 - deriv_approx) / deriv_approx);
  }
  double rel_error2 = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error2 = F2FRealPart((deriv2 - deriv_approx) / deriv_approx);
  }
  double abs_error1 = F2FRealPart(deriv1 - deriv_approx);
  double abs_error2 = F2FRealPart(deriv2 - deriv_approx);

  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    // Print out results of test
    printf("V2^{T}*dL/du_{S}*V1 test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv1));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error1);
    printf("\n");

    // Print out results of test
    printf("V1^{T}*(dL/du_{S})^{T}*V2 test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv2));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error2);
    printf("\n");
  }

  // Free allocated memory
  delete[] aero_disps;
  delete[] struct_loads;
  delete[] grad1;
  delete[] grad2;

  int fail = 0;
  if (fabs(rel_error1) >= rtol && fabs(abs_error1) >= rtol) {
    fail = 1;
  }
  if (fabs(rel_error2) >= rtol && fabs(abs_error2) >= rtol) {
    fail = 1;
  }

  if (rank == 0 && fail) {
    printf("testLoadJacVecProducts failed\n");
  }

  return fail;
}

/*
  Test output of dDdxA0Products function by computing directional derivative
  using function and comparing to finite difference and complex step
  approximations

  Arguments
  ---------
  struct_disps : structural node displacments
  test_vec_a1  : test vector of size aero nodes
  test_vec_a2  : test vector of size aero nodes
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int LDTransferScheme::testdDdxA0Products(const F2FScalar *struct_disps,
                                         const F2FScalar *test_vec_a1,
                                         const F2FScalar *test_vec_a2,
                                         const F2FScalar h, const double rtol,
                                         const double atol) {
  // Copy original unperturbed node locations
  F2FScalar *Xa_copy = new F2FScalar[3 * na];
  memcpy(Xa_copy, Xa, 3 * na * sizeof(F2FScalar));

  // Transfer the structural displacements to get rotations
  F2FScalar *aero_disps = new F2FScalar[3 * na];
  transferDisps(struct_disps, aero_disps);

  // Compute the adjoint-residual products using the function
  F2FScalar *grad = new F2FScalar[3 * na];
  applydDdxA0(test_vec_a1, grad);

  // Compute directional derivative
  F2FScalar deriv = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    deriv += grad[i] * test_vec_a2[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *XA0_cs = new F2FScalar[3 * na];

  for (int i = 0; i < 3 * na; i++) {
    XA0_cs[i] =
        Xa[i] + F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(test_vec_a2[i]));
  }
  setAeroNodes(XA0_cs, na);
  transferDisps(struct_disps, aero_disps);

  F2FScalar lamPsi = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    F2FScalar Psi = aero_disps[i];
    lamPsi += test_vec_a1[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPsi, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = -1.0 * F2FImagPart(lamPsi) / h;

  delete[] XA0_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *XA0_pos = new F2FScalar[3 * na];
  F2FScalar *XA0_neg = new F2FScalar[3 * na];
  for (int i = 0; i < 3 * na; i++) {
    XA0_pos[i] = Xa[i] + h * test_vec_a2[i];
    XA0_neg[i] = Xa[i] - h * test_vec_a2[i];
  }

  setAeroNodes(XA0_pos, na);
  transferDisps(struct_disps, aero_disps);
  F2FScalar lamPsi_pos = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    F2FScalar Psi = aero_disps[i];
    lamPsi_pos += test_vec_a1[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPsi_pos, 1, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  setAeroNodes(XA0_neg, na);
  transferDisps(struct_disps, aero_disps);
  F2FScalar lamPsi_neg = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    F2FScalar Psi = aero_disps[i];
    lamPsi_neg += test_vec_a1[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPsi_neg, 1, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  F2FScalar deriv_approx = -0.5 * (lamPsi_pos - lamPsi_neg) / h;

  delete[] XA0_pos;
  delete[] XA0_neg;
#endif
  // Compute relative error
  double abs_error = F2FRealPart(deriv - deriv_approx);
  double rel_error = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error = F2FRealPart((deriv - deriv_approx) / deriv_approx);
  }

  // Print out results of test
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    printf("lambda^{T}*dD/dx_{A0} test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error);
    printf("\n");
  }

  // Reset to unperturbed, original node locations
  setAeroNodes(Xa_copy, na);

  // Free allocated memory
  delete[] Xa_copy;
  delete[] aero_disps;
  delete[] grad;

  // Check if the test failed
  int fail = 0;
  if (fabs(rel_error) >= rtol && fabs(abs_error) >= atol) {
    fail = 1;
  }

  if (rank == 0 && fail) {
    printf("testdDdxA0Products failed\n");
  }

  return fail;
}

/*
  Test output of dDdxS0Products function by computing directional derivative
  using function and comparing to finite difference and complex step
  approximations

  Arguments
  ---------
  struct_disps : structural node displacements
  test_vec_a   : test vector of size aero nodes
  test_vec_s   : test vector of size struct nodes
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int LDTransferScheme::testdDdxS0Products(const F2FScalar *struct_disps,
                                         const F2FScalar *test_vec_a,
                                         const F2FScalar *test_vec_s,
                                         const F2FScalar h, const double rtol,
                                         const double atol) {
  // Copy the original, unperturbed node locations
  F2FScalar *Xs_copy = new F2FScalar[3 * ns_local];
  memcpy(Xs_copy, Xs_local, 3 * ns_local * sizeof(F2FScalar));

  // Transfer the structural displacements to get rotations
  F2FScalar *aero_disps = new F2FScalar[3 * na];
  transferDisps(struct_disps, aero_disps);

  // Compute the adjoint-residual products using the function
  F2FScalar *grad = new F2FScalar[3 * ns_local];
  applydDdxS0(test_vec_a, grad);

  // Compute directional derivative
  F2FScalar deriv = 0.0;
  for (int j = 0; j < 3 * ns_local; j++) {
    deriv += grad[j] * test_vec_s[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *XS0_cs = new F2FScalar[3 * ns_local];

  for (int j = 0; j < 3 * ns_local; j++) {
    XS0_cs[j] = Xs_copy[j] +
                F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(test_vec_s[j]));
  }
  setStructNodes(XS0_cs, ns_local);
  transferDisps(struct_disps, aero_disps);

  F2FScalar lamPsi = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    F2FScalar Psi = Xa[i] + aero_disps[i];
    lamPsi += test_vec_a[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPsi, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = -1.0 * F2FImagPart(lamPsi) / h;

  delete[] XS0_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *XS0_pos = new F2FScalar[3 * ns_local];
  F2FScalar *XS0_neg = new F2FScalar[3 * ns_local];
  for (int j = 0; j < 3 * ns_local; j++) {
    XS0_pos[j] = Xs_copy[j] + h * test_vec_s[j];
    XS0_neg[j] = Xs_copy[j] - h * test_vec_s[j];
  }

  setStructNodes(XS0_pos, ns_local);
  transferDisps(struct_disps, aero_disps);
  F2FScalar lamPsi_pos = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    F2FScalar Psi = Xa[i] + aero_disps[i];
    lamPsi_pos += test_vec_a[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPsi_pos, 1, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  setStructNodes(XS0_neg, ns_local);
  transferDisps(struct_disps, aero_disps);
  F2FScalar lamPsi_neg = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    F2FScalar Psi = Xa[i] + aero_disps[i];
    lamPsi_neg += test_vec_a[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPsi_neg, 1, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  F2FScalar deriv_approx = -0.5 * (lamPsi_pos - lamPsi_neg) / h;

  delete[] XS0_pos;
  delete[] XS0_neg;
#endif
  // Compute relative error
  double abs_error = F2FRealPart(deriv - deriv_approx);
  double rel_error = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error = F2FRealPart((deriv - deriv_approx) / deriv_approx);
  }

  // Print out results of test
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    printf("lambda^{T}*dD/dx_{S0} test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error);
    printf("\n");
  }

  // Reset the original, unperturbed nodes
  setStructNodes(Xs_copy, ns_local);

  // Free allocated memory
  delete[] Xs_copy;
  delete[] aero_disps;
  delete[] grad;

  // Check if the test failed
  int fail = 0;
  if (fabs(rel_error) >= rtol && fabs(abs_error) >= atol) {
    fail = 1;
  }

  if (rank == 0 && fail) {
    printf("testdDdxS0Products failed\n");
  }

  return fail;
}

/*
  Test output of dLdxA0Products function by computing directional derivative
  using function and comparing to finite difference and complex step
  approximations

  Arguments
  ---------
  struct_disps : structural node displacements
  aero_loads   : aerodynamic loads
  test_vec_a   : test vector of size aero nodes
  test_vec_s   : test vector of size struct nodes
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int LDTransferScheme::testdLdxA0Products(const F2FScalar *struct_disps,
                                         const F2FScalar *aero_loads,
                                         const F2FScalar *test_vec_a,
                                         const F2FScalar *test_vec_s,
                                         const F2FScalar h, const double rtol,
                                         const double atol) {
  const int dof_per_node = getStructNodeDof();

  // Copy original, unperturbed nodes
  F2FScalar *Xa_copy = new F2FScalar[3 * na];
  memcpy(Xa_copy, Xa, 3 * na * sizeof(F2FScalar));

  // Transfer the structural displacements
  F2FScalar *aero_disps = new F2FScalar[3 * na];
  transferDisps(struct_disps, aero_disps);

  // Transfer the aerodynamic loads
  F2FScalar *struct_loads = new F2FScalar[dof_per_node * ns_local];
  transferLoads(aero_loads, struct_loads);

  // Compute the derivatives of the adjoint-residual products using the
  // function
  F2FScalar *grad = new F2FScalar[3 * na];
  applydLdxA0(test_vec_s, grad);

  // Compute directional derivative
  F2FScalar deriv = 0.0;
  for (int i = 0; i < 3 * na; i++) {
    deriv += grad[i] * test_vec_a[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *XA0_cs = new F2FScalar[3 * na];

  for (int i = 0; i < 3 * na; i++) {
    XA0_cs[i] =
        Xa[i] + F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(test_vec_a[i]));
  }
  setAeroNodes(XA0_cs, na);
  transferDisps(struct_disps, aero_disps);
  transferLoads(aero_loads, struct_loads);

  F2FScalar lamPhi = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    F2FScalar Phi = struct_loads[j];
    lamPhi += test_vec_s[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPhi, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = -1.0 * F2FImagPart(lamPhi) / h;

  delete[] XA0_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *XA0_pos = new F2FScalar[3 * na];
  F2FScalar *XA0_neg = new F2FScalar[3 * na];
  for (int i = 0; i < 3 * na; i++) {
    XA0_pos[i] = Xa[i] + h * test_vec_a[i];
    XA0_neg[i] = Xa[i] - h * test_vec_a[i];
  }

  setAeroNodes(XA0_pos, na);
  transferDisps(struct_disps, aero_disps);
  transferLoads(aero_loads, struct_loads);

  F2FScalar lamPhi_pos = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    F2FScalar Phi = struct_loads[j];
    lamPhi_pos += test_vec_s[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPhi_pos, 1, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  setAeroNodes(XA0_neg, na);
  transferDisps(struct_disps, aero_disps);
  transferLoads(aero_loads, struct_loads);

  F2FScalar lamPhi_neg = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    F2FScalar Phi = struct_loads[j];
    lamPhi_neg += test_vec_s[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPhi_neg, 1, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  F2FScalar deriv_approx = -0.5 * (lamPhi_pos - lamPhi_neg) / h;

  delete[] XA0_pos;
  delete[] XA0_neg;
#endif
  // Compute relative error
  double rel_error = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error = F2FRealPart((deriv - deriv_approx) / deriv_approx);
  }
  double abs_error = F2FRealPart(deriv - deriv_approx);

  // Print out results of test
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    printf("lambda^{T}*dL/dx_{A0} test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", F2FRealPart(rel_error));
    printf("\n");
  }

  // Reset to original, unperturbed nodes
  setAeroNodes(Xa_copy, na);

  // Free allocated memory
  delete[] Xa_copy;
  delete[] aero_disps;
  delete[] struct_loads;
  delete[] grad;

  int fail = 0;
  if (fabs(rel_error) >= rtol && fabs(abs_error) >= rtol) {
    fail = 1;
  }

  if (rank == 0 && fail) {
    printf("testdLdxA0Products failed\n");
  }

  return fail;
}

/*
  Test output of dLdxS0Products function by computing directional derivative
  using function and comparing to finite difference and complex step
  approximations

  Arguments
  ---------
  struct_disps : structural node displacements
  aero_loads   : aerodynamic loads
  test_vec_s1  : test vector of size struct nodes
  test_vec_s2  : test vector of size struct nodes
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int LDTransferScheme::testdLdxS0Products(const F2FScalar *struct_disps,
                                         const F2FScalar *aero_loads,
                                         const F2FScalar *test_vec_s1,
                                         const F2FScalar *test_vec_s2,
                                         const F2FScalar h, const double rtol,
                                         const double atol) {
  const int dof_per_node = getStructNodeDof();

  // Copy original, unperturbed node locations
  F2FScalar *Xs_copy = new F2FScalar[3 * ns_local];
  memcpy(Xs_copy, Xs_local, 3 * ns_local * sizeof(F2FScalar));

  // Transfer the structural displacements
  F2FScalar *aero_disps = new F2FScalar[3 * na];
  transferDisps(struct_disps, aero_disps);

  // Transfer the aerodynamic loads
  F2FScalar *struct_loads = new F2FScalar[dof_per_node * ns_local];
  transferLoads(aero_loads, struct_loads);

  // Compute the derivatives of the adjoint-residual products using the
  // function
  F2FScalar *grad = new F2FScalar[3 * ns_local];
  applydLdxS0(test_vec_s1, grad);

  // Compute directional derivative
  F2FScalar deriv = 0.0;
  for (int j = 0; j < 3 * ns_local; j++) {
    deriv += grad[j] * test_vec_s2[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *XS0_cs = new F2FScalar[3 * ns_local];

  for (int j = 0; j < 3 * ns_local; j++) {
    XS0_cs[j] = Xs_copy[j] +
                F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(test_vec_s2[j]));
  }
  setStructNodes(XS0_cs, ns_local);
  transferDisps(struct_disps, aero_disps);
  transferLoads(aero_loads, struct_loads);

  F2FScalar lamPhi = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    F2FScalar Phi = struct_loads[j];
    lamPhi += test_vec_s1[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPhi, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = -1.0 * F2FImagPart(lamPhi) / h;

  delete[] XS0_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *XS0_pos = new F2FScalar[3 * ns_local];
  F2FScalar *XS0_neg = new F2FScalar[3 * ns_local];
  for (int j = 0; j < 3 * ns_local; j++) {
    XS0_pos[j] = Xs_copy[j] + h * test_vec_s2[j];
    XS0_neg[j] = Xs_copy[j] - h * test_vec_s2[j];
  }

  setStructNodes(XS0_pos, ns_local);
  transferDisps(struct_disps, aero_disps);
  transferLoads(aero_loads, struct_loads);
  F2FScalar lamPhi_pos = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    F2FScalar Phi = struct_loads[j];
    lamPhi_pos += test_vec_s1[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPhi_pos, 1, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  setStructNodes(XS0_neg, ns_local);
  transferDisps(struct_disps, aero_disps);
  transferLoads(aero_loads, struct_loads);
  F2FScalar lamPhi_neg = 0.0;
  for (int j = 0; j < dof_per_node * ns_local; j++) {
    F2FScalar Phi = struct_loads[j];
    lamPhi_neg += test_vec_s1[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &lamPhi_neg, 1, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  F2FScalar deriv_approx = -0.5 * (lamPhi_pos - lamPhi_neg) / h;

  delete[] XS0_pos;
  delete[] XS0_neg;
#endif
  // Compute relative error
  double rel_error = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error = F2FRealPart((deriv - deriv_approx) / deriv_approx);
  }
  double abs_error = F2FRealPart(deriv - deriv_approx);

  // Print out results of test
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    printf("lambda^{T}*dL/dx_{S0} test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", F2FRealPart(rel_error));
    printf("\n");
  }

  // Free allocated memory
  delete[] Xs_copy;
  delete[] aero_disps;
  delete[] struct_loads;
  delete[] grad;

  // Check if the test failed
  int fail = 0;
  if (fabs(rel_error) >= rtol && fabs(abs_error) >= atol) {
    fail = 1;
  }

  if (rank == 0 && fail) {
    printf("testdLdxS0Products failed\n");
  }

  return fail;
}

/*
  Computes decomposition H = RS using the Singular Value Decomposition (SVD)

  Arguments
  ----------
  H : covariance matrix

  Returns
  --------
  R : rotation matrix
  S : symmetric matrix from polar decomposition of H

*/
void LDTransferScheme::computeRotation(const F2FScalar *H, F2FScalar *R,
                                       F2FScalar *S) {
  // Allocate memory for local variables
  int m = 3, n = 3, lda = 3, ldu = 3, ldvt = 3, info, lwork = 50;  // for SVD
  F2FReal work[50];     // work matrix for SVD
  F2FReal U[9], VT[9];  // output matrices for SVD
  F2FReal detR;         // determinant of rotation matrix
  F2FReal s[3];
  F2FReal Hcopy[9];

  // Copy over the values of H - LAPACK conveniently destroys
  // the entries of H
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar Hreal[9], Himag[9];
  for (int i = 0; i < 9; i++) {
    Hcopy[i] = F2FRealPart(H[i]);

    // Keep the real and the imaginary parts of the input for later
    // use...
    Hreal[i] = F2FRealPart(H[i]);
    Himag[i] = F2FImagPart(H[i]);
  }
#else
  memcpy(Hcopy, H, 9 * sizeof(F2FScalar));
#endif

  // Perform SVD of the covariance matrix
  LAPACKdgesvd("All", "All", &m, &n, Hcopy, &lda, s, U, &ldu, VT, &ldvt, work,
               &lwork, &info);  // compute SVD

  // R = U * V^T
  // [ R[0] R[3] R[6] ] = [ U[0] U[3] U[6] ][ VT[0] VT[3] VT[6] ]
  // [ R[1] R[4] R[7] ]   [ U[1] U[4] U[7] ][ VT[1] VT[4] VT[7] ]
  // [ R[2] R[5] R[8] ]   [ U[2] U[5] U[8] ][ VT[2] VT[5] VT[8] ]

  R[0] = U[0] * VT[0] + U[3] * VT[1] + U[6] * VT[2];
  R[1] = U[1] * VT[0] + U[4] * VT[1] + U[7] * VT[2];
  R[2] = U[2] * VT[0] + U[5] * VT[1] + U[8] * VT[2];

  R[3] = U[0] * VT[3] + U[3] * VT[4] + U[6] * VT[5];
  R[4] = U[1] * VT[3] + U[4] * VT[4] + U[7] * VT[5];
  R[5] = U[2] * VT[3] + U[5] * VT[4] + U[8] * VT[5];

  R[6] = U[0] * VT[6] + U[3] * VT[7] + U[6] * VT[8];
  R[7] = U[1] * VT[6] + U[4] * VT[7] + U[7] * VT[8];
  R[8] = U[2] * VT[6] + U[5] * VT[7] + U[8] * VT[8];

  // Take determinant of rotation matrix
  detR = F2FRealPart(det(R));

  // If negative determinant, matrix computed is a reflection
  // Can calculate rotation matrix from reflection found
  if (F2FRealPart(detR) < 0.0) {
    // Rotation matrix given is a reflection
    // Can calculate rotation matrix from reflection found
    R[0] = U[0] * VT[0] + U[3] * VT[1] - U[6] * VT[2];
    R[1] = U[1] * VT[0] + U[4] * VT[1] - U[7] * VT[2];
    R[2] = U[2] * VT[0] + U[5] * VT[1] - U[8] * VT[2];

    R[3] = U[0] * VT[3] + U[3] * VT[4] - U[6] * VT[5];
    R[4] = U[1] * VT[3] + U[4] * VT[4] - U[7] * VT[5];
    R[5] = U[2] * VT[3] + U[5] * VT[4] - U[8] * VT[5];

    R[6] = U[0] * VT[6] + U[3] * VT[7] - U[6] * VT[8];
    R[7] = U[1] * VT[6] + U[4] * VT[7] - U[7] * VT[8];
    R[8] = U[2] * VT[6] + U[5] * VT[7] - U[8] * VT[8];
  }

#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar Sreal[9];
  Sreal[0] = R[0] * Hreal[0] + R[1] * Hreal[1] + R[2] * Hreal[2];
  Sreal[1] = R[3] * Hreal[0] + R[4] * Hreal[1] + R[5] * Hreal[2];
  Sreal[2] = R[6] * Hreal[0] + R[7] * Hreal[1] + R[8] * Hreal[2];
  Sreal[3] = R[0] * Hreal[3] + R[1] * Hreal[4] + R[2] * Hreal[5];
  Sreal[4] = R[3] * Hreal[3] + R[4] * Hreal[4] + R[5] * Hreal[5];
  Sreal[5] = R[6] * Hreal[3] + R[7] * Hreal[4] + R[8] * Hreal[5];
  Sreal[6] = R[0] * Hreal[6] + R[1] * Hreal[7] + R[2] * Hreal[8];
  Sreal[7] = R[3] * Hreal[6] + R[4] * Hreal[7] + R[5] * Hreal[8];
  Sreal[8] = R[6] * Hreal[6] + R[7] * Hreal[7] + R[8] * Hreal[8];

  // Assemble matrix system and factor
  F2FScalar M1[15 * 15];
  assembleM1(R, Sreal, M1);
  int ipiv[15];
  m = 15;
  info = 0;
  LAPACKgetrf(&m, &m, M1, &m, ipiv, &info);

  for (int k = 0; k < 9; k++) {
    // Solve system for each component of R
    F2FScalar x[15];
    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[k] = -1.0;
    int nrhs = 1;
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar X[] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]};

    F2FScalar Rcomplex = X[0] * Himag[0] + X[3] * Himag[1] + X[6] * Himag[2] +
                         X[1] * Himag[3] + X[4] * Himag[4] + X[7] * Himag[5] +
                         X[2] * Himag[6] + X[5] * Himag[7] + X[8] * Himag[8];

    R[k] += F2FScalar(0.0, F2FRealPart(Rcomplex));
  }

#endif  // FUNTOFEM_USE_COMPLEX

  // Compute the positive-semidefinite polar decomposition matrix S
  // S = R^{T}*H
  // [ S[0] S[3] S[6] ] = [ R[0] R[1] R[2] ][ H[0] H[3] H[6] ]
  // [ S[1] S[4] S[7] ] = [ R[3] R[4] R[5] ][ H[1] H[4] H[7] ]
  // [ S[2] S[5] S[8] ] = [ R[6] R[7] R[8] ][ H[2] H[5] H[8] ]
  S[0] = R[0] * H[0] + R[1] * H[1] + R[2] * H[2];
  S[1] = R[3] * H[0] + R[4] * H[1] + R[5] * H[2];
  S[2] = R[6] * H[0] + R[7] * H[1] + R[8] * H[2];
  S[3] = R[0] * H[3] + R[1] * H[4] + R[2] * H[5];
  S[4] = R[3] * H[3] + R[4] * H[4] + R[5] * H[5];
  S[5] = R[6] * H[3] + R[7] * H[4] + R[8] * H[5];
  S[6] = R[0] * H[6] + R[1] * H[7] + R[2] * H[8];
  S[7] = R[3] * H[6] + R[4] * H[7] + R[5] * H[8];
  S[8] = R[6] * H[6] + R[7] * H[7] + R[8] * H[8];
}

/*
  compute and check the determinant of M1 15x15 matrix used in adjoint computations
  product of the diagonal elements after LU factorization
*/
F2FScalar LDTransferScheme::printDetM1(const F2FScalar *A) {
    F2FScalar detM1 = 1.0;
    for (int i = 0; i < 15; i++) {
      detM1 *= A[i + 15 * i];
    }
    return detM1;
}


/*
  Builds the matrix of the linear system to be solved in the process of
  computing the SVD derivative

  Arguments
  ---------
  R  : rotation matrix
  S  : symmetric matrix

  Returns
  -------
  M1 : matrix system

*/
void LDTransferScheme::assembleM1(const F2FScalar *R, const F2FScalar *S,
                                  F2FScalar *A) {
  // Set the entries to zero
  memset(A, 0, 15 * 15 * sizeof(F2FScalar));

  /*
  M1 = [ A1 C1 ]
       [ B1 0  ]

  A1 = -kron(S, eye(3))*T                               9x9
  B1 = D*(kron(I, R^T)*T + kron(R^T, I))                6x9
  C1 = kron(eye(3), R)*Dstar                            9x6
  */

  // Fill in the elements of A corresponding to A1
  // Rows 0-2
  A[0 + 15 * 0] = -S[0];
  A[0 + 15 * 1] = -S[3];
  A[0 + 15 * 2] = -S[6];
  A[1 + 15 * 3] = -S[0];
  A[1 + 15 * 4] = -S[3];
  A[1 + 15 * 5] = -S[6];
  A[2 + 15 * 6] = -S[0];
  A[2 + 15 * 7] = -S[3];
  A[2 + 15 * 8] = -S[6];

  // Rows 3-5
  A[3 + 15 * 0] = -S[1];
  A[3 + 15 * 1] = -S[4];
  A[3 + 15 * 2] = -S[7];
  A[4 + 15 * 3] = -S[1];
  A[4 + 15 * 4] = -S[4];
  A[4 + 15 * 5] = -S[7];
  A[5 + 15 * 6] = -S[1];
  A[5 + 15 * 7] = -S[4];
  A[5 + 15 * 8] = -S[7];

  // Rows 6-8
  A[6 + 15 * 0] = -S[2];
  A[6 + 15 * 1] = -S[5];
  A[6 + 15 * 2] = -S[8];
  A[7 + 15 * 3] = -S[2];
  A[7 + 15 * 4] = -S[5];
  A[7 + 15 * 5] = -S[8];
  A[8 + 15 * 6] = -S[2];
  A[8 + 15 * 7] = -S[5];
  A[8 + 15 * 8] = -S[8];

  // Fill in the elements of A corresponding to B1
  // Columns 0-2
  A[9 + 15 * 0] = 2.0 * R[0];
  A[10 + 15 * 1] = R[0];
  A[11 + 15 * 2] = R[0];
  A[10 + 15 * 0] = R[3];
  A[12 + 15 * 1] = 2.0 * R[3];
  A[13 + 15 * 2] = R[3];
  A[11 + 15 * 0] = R[6];
  A[13 + 15 * 1] = R[6];
  A[14 + 15 * 2] = 2.0 * R[6];

  // Columns 3-5
  A[9 + 15 * 3] = 2.0 * R[1];
  A[10 + 15 * 4] = R[1];
  A[11 + 15 * 5] = R[1];
  A[10 + 15 * 3] = R[4];
  A[12 + 15 * 4] = 2.0 * R[4];
  A[13 + 15 * 5] = R[4];
  A[11 + 15 * 3] = R[7];
  A[13 + 15 * 4] = R[7];
  A[14 + 15 * 5] = 2.0 * R[7];

  // Columns 6-8
  A[9 + 15 * 6] = 2.0 * R[2];
  A[10 + 15 * 7] = R[2];
  A[11 + 15 * 8] = R[2];
  A[10 + 15 * 6] = R[5];
  A[12 + 15 * 7] = 2.0 * R[5];
  A[13 + 15 * 8] = R[5];
  A[11 + 15 * 6] = R[8];
  A[13 + 15 * 7] = R[8];
  A[14 + 15 * 8] = 2.0 * R[8];

  // Fill in the elements of A corresponding to C1
  // Rows 0-2
  A[0 + 15 * 9] = R[0];
  A[0 + 15 * 10] = R[3];
  A[0 + 15 * 11] = R[6];
  A[1 + 15 * 9] = R[1];
  A[1 + 15 * 10] = R[4];
  A[1 + 15 * 11] = R[7];
  A[2 + 15 * 9] = R[2];
  A[2 + 15 * 10] = R[5];
  A[2 + 15 * 11] = R[8];

  // Rows 3-5
  A[3 + 15 * 10] = R[0];
  A[3 + 15 * 12] = R[3];
  A[3 + 15 * 13] = R[6];
  A[4 + 15 * 10] = R[1];
  A[4 + 15 * 12] = R[4];
  A[4 + 15 * 13] = R[7];
  A[5 + 15 * 10] = R[2];
  A[5 + 15 * 12] = R[5];
  A[5 + 15 * 13] = R[8];

  // Rows 6-8
  A[6 + 15 * 11] = R[0];
  A[6 + 15 * 13] = R[3];
  A[6 + 15 * 14] = R[6];
  A[7 + 15 * 11] = R[1];
  A[7 + 15 * 13] = R[4];
  A[7 + 15 * 14] = R[7];
  A[8 + 15 * 11] = R[2];
  A[8 + 15 * 13] = R[5];
  A[8 + 15 * 14] = R[8];
}

// Test Functions
int ThermalTransfer::testAllDerivatives(const F2FScalar *struct_temps,
                                        const F2FScalar *aero_flux,
                                        const F2FScalar h, const double rtol,
                                        const double atol) {
  const int aero_nnodes = getNumLocalAeroNodes();
  const int struct_nnodes = getNumLocalStructNodes();

  // Create random testing data
  F2FScalar *test_vec_a1 = new F2FScalar[aero_nnodes];
  for (int i = 0; i < aero_nnodes; i++) {
    test_vec_a1[i] = (1.0 * rand()) / RAND_MAX;
  }

  F2FScalar *test_vec_s1 = new F2FScalar[struct_nnodes];
  F2FScalar *test_vec_s2 = new F2FScalar[struct_nnodes];
  for (int j = 0; j < struct_nnodes; j++) {
    test_vec_s1[j] = (1.0 * rand()) / RAND_MAX;
    test_vec_s2[j] = (1.0 * rand()) / RAND_MAX;
  }

  int fail1 =
      testFluxTransfer(struct_temps, aero_flux, test_vec_s1, h, rtol, atol);
  int fail2 = testTempJacVecProducts(struct_temps, test_vec_a1, test_vec_s1, h,
                                     rtol, atol);
  int fail3 = testFluxJacVecProducts(struct_temps, aero_flux, test_vec_a1,
                                     test_vec_s1, h, rtol, atol);
  int fail = (fail1 || fail2 || fail3);

  delete[] test_vec_a1;
  delete[] test_vec_s1;
  delete[] test_vec_s2;

  return fail;
}

/*
  Tests flux transfer by computing derivative of product of heat flux on and
  temperatures of aerodynamic surface nodes with respect to structural node
  temperatures and comparing with results from finite difference and complex
  step approximation

  Arguments
  ---------
  struct_temps : structural node temperatures
  aero_flux    : heat flux on aerodynamic surface nodes
  pert         : direction of perturbation of structural node heat flux
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int ThermalTransfer::testFluxTransfer(const F2FScalar *struct_temps,
                                      const F2FScalar *aero_flux,
                                      const F2FScalar *pert, const F2FScalar h,
                                      const double rtol, const double atol) {
  // Transfer the structural temperatures
  F2FScalar *aero_temps = new F2FScalar[na];
  transferTemp(struct_temps, aero_temps);

  // Transfer the aerodynamic heat flux
  F2FScalar *struct_flux = new F2FScalar[ns_local];
  transferFlux(aero_flux, struct_flux);

  // Compute directional derivative (structural heat flux times perturbation)
  F2FScalar deriv = 0.0;
  for (int j = 0; j < ns_local; j++) {
    deriv += struct_flux[j] * pert[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *Ts_cs = new F2FScalar[ns_local];
  F2FScalar *Ta_cs = new F2FScalar[na];

  for (int j = 0; j < ns_local; j++) {
    Ts_cs[j] =
        struct_temps[j] + F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(pert[j]));
  }
  transferTemp(Ts_cs, Ta_cs);

  F2FScalar work = 0.0;
  for (int i = 0; i < na; i++) {
    work += Ha[i] * Ta_cs[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &work, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = F2FImagPart(work) / h;

  delete[] Ts_cs;
  delete[] Ta_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *Ts_pos = new F2FScalar[ns_local];
  F2FScalar *Ts_neg = new F2FScalar[ns_local];
  F2FScalar *Ta_pos = new F2FScalar[na];
  F2FScalar *Ta_neg = new F2FScalar[na];
  for (int j = 0; j < ns_local; j++) {
    Ts_pos[j] = struct_temps[j] + h * pert[j];
    Ts_neg[j] = struct_temps[j] - h * pert[j];
  }

  transferTemp(Ts_pos, Ta_pos);
  F2FScalar work_pos = 0.0;
  for (int i = 0; i < na; i++) {
    work_pos += Ha[i] * Ta_pos[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &work_pos, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  transferTemp(Ts_neg, Ta_neg);
  F2FScalar work_neg = 0.0;
  for (int i = 0; i < na; i++) {
    work_neg += Ha[i] * Ta_neg[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &work_neg, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  F2FScalar deriv_approx = 0.5 * (work_pos - work_neg) / h;

  delete[] Ts_pos;
  delete[] Ts_neg;
  delete[] Ta_pos;
  delete[] Ta_neg;
#endif  // FUNTOFEM_USE_COMPLEX
  // Compute relative error
  double rel_error = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error = F2FRealPart((deriv - deriv_approx) / deriv_approx);
  }
  double abs_error = F2FRealPart(deriv - deriv_approx);

  // Print results
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    printf("\n");
    printf("Heat Flux transfer test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error);
    printf("\n");
  }

  // Free allocated memory
  delete[] aero_temps;
  delete[] struct_flux;

  // Check if the test failed
  int fail = 0;
  if (fabs(rel_error) >= rtol && fabs(abs_error) >= atol) {
    fail = 1;
  }

  return fail;
}

/*
  Tests output of dTdtSProducts and dTdtSTransProducts by computing a product
  test_vec_a*J*test_vec_s (where J is the Jacobian) and comparing with results
  from finite difference and complex step

  Arguments
  ---------
  struct_temps : structural node temperatures
  test_vec_a   : test vector the length of the aero nodes
  test_vec_s   : test vector the length of the struct nodes
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int ThermalTransfer::testTempJacVecProducts(const F2FScalar *struct_temps,
                                            const F2FScalar *test_vec_a,
                                            const F2FScalar *test_vec_s,
                                            const F2FScalar h,
                                            const double rtol,
                                            const double atol) {
  // Transfer the structural temperatures
  F2FScalar *aero_temps = new F2FScalar[na];
  transferTemp(struct_temps, aero_temps);

  // Compute the Jacobian-vector products using the function
  F2FScalar *grad1 = new F2FScalar[na];
  applydTdtS(test_vec_s, grad1);

  // Compute product of test_vec_a with the Jacobian-vector products
  F2FScalar deriv1 = 0.0;
  for (int i = 0; i < na; i++) {
    deriv1 += test_vec_a[i] * grad1[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv1, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Compute the transpose Jacobian-vector products using the function
  F2FScalar *grad2 = new F2FScalar[ns_local];
  applydTdtSTrans(test_vec_a, grad2);

  // Compute product of V1 with the transpose Jacobian-vector products
  F2FScalar deriv2 = 0.0;
  for (int j = 0; j < ns_local; j++) {
    deriv2 += test_vec_s[j] * grad2[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv2, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Compute complex step approximation
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *Ts_cs = new F2FScalar[ns_local];
  for (int j = 0; j < ns_local; j++) {
    Ts_cs[j] +=
        struct_temps[j] + F2FScalar(0.0, F2FRealPart(h * test_vec_s[j]));
  }
  transferTemp(Ts_cs, aero_temps);

  F2FScalar VPsi = 0.0;
  for (int i = 0; i < na; i++) {
    F2FScalar Psi = Xa[i] + aero_temps[i];
    VPsi += test_vec_a[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPsi, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = -1.0 * F2FImagPart(VPsi) / h;

  delete[] Ts_cs;

  // Compute finite difference approximation (central)
#else
  F2FScalar *Ts_pos = new F2FScalar[ns_local];
  F2FScalar *Ts_neg = new F2FScalar[ns_local];
  for (int j = 0; j < ns_local; j++) {
    Ts_pos[j] = struct_temps[j] + h * test_vec_s[j];
    Ts_neg[j] = struct_temps[j] - h * test_vec_s[j];
  }

  transferTemp(Ts_pos, aero_temps);
  F2FScalar VPsi_pos = 0.0;
  for (int i = 0; i < na; i++) {
    F2FScalar Psi = Xa[i] + aero_temps[i];
    VPsi_pos += test_vec_a[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPsi_pos, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  transferTemp(Ts_neg, aero_temps);
  F2FScalar VPsi_neg = 0.0;
  for (int i = 0; i < na; i++) {
    F2FScalar Psi = Xa[i] + aero_temps[i];
    VPsi_neg += test_vec_a[i] * Psi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPsi_neg, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  F2FScalar deriv_approx = -0.5 * (VPsi_pos - VPsi_neg) / h;

  delete[] Ts_pos;
  delete[] Ts_neg;
#endif
  // Compute relative error
  double rel_error1 = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error1 = F2FRealPart((deriv1 - deriv_approx) / deriv_approx);
  }
  double rel_error2 = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error2 = F2FRealPart((deriv2 - deriv_approx) / deriv_approx);
  }
  double abs_error1 = F2FRealPart(deriv1 - deriv_approx);
  double abs_error2 = F2FRealPart(deriv2 - deriv_approx);

  // Print out results of test
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    printf("V2^{T}*dT/dt_{S}*V1 test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv1));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error1);
    printf("\n");

    printf("V1^{T}*(dT/dt_{S})^{T}*V2 test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv2));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error2);
    printf("\n");
  }

  // Free allocated memory
  delete[] aero_temps;
  delete[] grad1;
  delete[] grad2;

  int fail = 0;
  if (fabs(rel_error1) >= rtol && fabs(abs_error1) >= rtol) {
    fail = 1;
  }
  if (fabs(rel_error2) >= rtol && fabs(abs_error2) >= rtol) {
    fail = 1;
  }

  if (rank == 0 && fail) {
    printf("testTempJacVecProducts failed\n");
  }

  return fail;
}

/*
  Tests output of dQdqAProducts and dQdqATransProducts by computing a product
  test_vec_a*J*test_vec_s (where J is the Jacobian) and comparing with
  results from finite difference and complex step

  Arguments
  ---------
  struct_temps : structural node temperatures
  aero_flux   : aerodynamic heat flux
  test_vec_a  : test vector the size of aero nodes
  test_vec_s  : test vector the size of struct nodes
  h            : step size
  rtol         : relative error tolerance for the test to pass
  atol         : absolute error tolerance for the test to pass
*/
int ThermalTransfer::testFluxJacVecProducts(
    const F2FScalar *struct_temps, const F2FScalar *aero_flux,
    const F2FScalar *test_vec_a, const F2FScalar *test_vec_s, const F2FScalar h,
    const double rtol, const double atol) {
  // Transfer the structural displacements
  F2FScalar *aero_temps = new F2FScalar[na];
  transferTemp(struct_temps, aero_temps);

  // Transfer the aerodynamic loads to get MM, IPIV
  F2FScalar *struct_flux = new F2FScalar[ns_local];
  transferFlux(aero_flux, struct_flux);

  // Compute the Jacobian-vector products using the function
  F2FScalar *grad1 = new F2FScalar[ns_local];
  applydQdqA(test_vec_a, grad1);

  // Compute directional derivative
  F2FScalar deriv1 = 0.0;
  for (int j = 0; j < ns_local; j++) {
    deriv1 += grad1[j] * test_vec_s[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv1, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Compute transpose Jacobian-vector products using the function
  F2FScalar *grad2 = new F2FScalar[na];
  applydQdqATrans(test_vec_s, grad2);

  // Compute directional derivative
  F2FScalar deriv2 = 0.0;
  for (int j = 0; j < na; j++) {
    deriv2 += grad2[j] * test_vec_a[j];
  }
  MPI_Allreduce(MPI_IN_PLACE, &deriv2, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  // Approximate using complex step
#ifdef FUNTOFEM_USE_COMPLEX
  F2FScalar *Ts_cs = new F2FScalar[ns_local];
  for (int j = 0; j < ns_local; j++) {
    Ts_cs[j] += struct_temps[j] +
                F2FScalar(0.0, F2FRealPart(h) * F2FRealPart(test_vec_s[j]));
    // F2FScalar(0.0, F2FRealPart(h*test_vec_s1[j]));
  }
  transferTemp(Ts_cs, aero_temps);

  F2FScalar VPhi = 0.0;
  for (int j = 0; j < na; j++) {
    F2FScalar Phi = Xa[j] + aero_temps[j];
    VPhi += test_vec_a[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPhi, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);
  F2FScalar deriv_approx = -1.0 * F2FImagPart(VPhi) / h;
  delete[] Ts_cs;

  // Approximate using finite difference (central)
#else
  F2FScalar *Ts_pos = new F2FScalar[ns_local];
  F2FScalar *Ts_neg = new F2FScalar[ns_local];
  for (int j = 0; j < ns_local; j++) {
    Ts_pos[j] = struct_temps[j] + h * test_vec_s[j];
    Ts_neg[j] = struct_temps[j] - h * test_vec_s[j];
  }

  transferTemp(Ts_pos, aero_temps);
  F2FScalar VPhi_pos = 0.0;
  for (int j = 0; j < na; j++) {
    F2FScalar Phi = Xa[j] + aero_temps[j];
    VPhi_pos += test_vec_a[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPhi_pos, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  transferTemp(Ts_neg, aero_temps);
  F2FScalar VPhi_neg = 0.0;
  for (int j = 0; j < na; j++) {
    F2FScalar Phi = Xa[j] + aero_temps[j];
    VPhi_neg += test_vec_a[j] * Phi;
  }
  MPI_Allreduce(MPI_IN_PLACE, &VPhi_neg, 1, F2F_MPI_TYPE, MPI_SUM, global_comm);

  F2FScalar deriv_approx = -0.5 * (VPhi_pos - VPhi_neg) / h;

  delete[] Ts_pos;
  delete[] Ts_neg;
#endif
  // Compute relative error
  double rel_error1 = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error1 = F2FRealPart((deriv1 - deriv_approx) / deriv_approx);
  }
  double rel_error2 = 0.0;
  if (F2FRealPart(deriv_approx) != 0.0) {
    rel_error2 = F2FRealPart((deriv2 - deriv_approx) / deriv_approx);
  }
  double abs_error1 = F2FRealPart(deriv1 - deriv_approx);
  double abs_error2 = F2FRealPart(deriv2 - deriv_approx);

  // Print out results of test
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == 0) {
    printf("V2^{T}*dQ/dq_{A}*V1 test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv1));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error1);
    printf("\n");

    // Print out results of test
    printf("V1^{T}*(dQ/dq_{A})^{T}*V2 test with step: %e\n", F2FRealPart(h));
    printf("deriv          = %22.15e\n", F2FRealPart(deriv2));
    printf("deriv, approx  = %22.15e\n", F2FRealPart(deriv_approx));
    printf("relative error = %22.15e\n", rel_error2);
    printf("\n");
  }

  // Free allocated memory
  delete[] aero_temps;
  delete[] struct_flux;
  delete[] grad1;
  delete[] grad2;

  int fail = 0;
  if (fabs(rel_error1) >= rtol && fabs(abs_error1) >= rtol) {
    fail = 1;
  }
  if (fabs(rel_error2) >= rtol && fabs(abs_error2) >= rtol) {
    fail = 1;
  }

  if (rank == 0 && fail) {
    printf("testFluxJacVecProducts failed\n");
  }

  return fail;
}

/*
  Add two R^{3} vectors
*/
void vec_add(const F2FScalar *x, const F2FScalar *y, F2FScalar *xy) {
  xy[0] = x[0] + y[0];
  xy[1] = x[1] + y[1];
  xy[2] = x[2] + y[2];
}

/*
  Subtract one R^{3} vector from another
*/
void vec_diff(const F2FScalar *x, const F2FScalar *y, F2FScalar *xy) {
  xy[0] = y[0] - x[0];
  xy[1] = y[1] - x[1];
  xy[2] = y[2] - x[2];
}

/*
  Perform scalar-vector multiplication for an R^{3} vector
*/
void vec_scal_mult(const F2FScalar a, const F2FScalar *x, F2FScalar *ax) {
  ax[0] = a * x[0];
  ax[1] = a * x[1];
  ax[2] = a * x[2];
}

/*
  Take cross product of two R^{3} vectors
*/
void vec_cross(const F2FScalar *x, const F2FScalar *y, F2FScalar *xy) {
  xy[0] = x[1] * y[2] - x[2] * y[1];
  xy[1] = x[2] * y[0] - x[0] * y[2];
  xy[2] = x[0] * y[1] - x[1] * y[0];
}

/*
  Compute magnituge (L2-norm) of an R^{3} vector
*/
F2FScalar vec_mag(const F2FScalar *x) {
  return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

/*
  Take dot product of two R^{3} vectors
*/
F2FScalar vec_dot(const F2FScalar *x, const F2FScalar *y) {
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

/*
  Take the determinant of a 3x3 matrix (given in column major order)
*/
F2FScalar det(const F2FScalar *A) {
  F2FScalar detA = A[0] * (A[4] * A[8] - A[5] * A[7]) -
                   A[3] * (A[1] * A[8] - A[2] * A[7]) +
                   A[6] * (A[1] * A[5] - A[2] * A[4]);

  return detA;
}