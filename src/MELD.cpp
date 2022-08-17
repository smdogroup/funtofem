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
#include "MELD.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>

#include "LocatePoint.h"
#include "funtofemlapack.h"

MELD::MELD(MPI_Comm all, MPI_Comm structure, int _struct_root, MPI_Comm aero,
           int _aero_root, int symmetry, int num_nearest, F2FScalar beta) {
  // Initialize communicators
  global_comm = all;
  struct_comm = structure;
  aero_comm = aero;
  struct_root = _struct_root;
  aero_root = _aero_root;

  // Initialize aerodynamic data member variables
  Xa = NULL;
  Fa = NULL;
  na = 0;

  // Initialize structural data member variables
  Xs = NULL;
  Us = NULL;
  ns = 0;

  Xs_local = NULL;
  ns_local = 0;

  // Set the symmetry parameter
  isymm = symmetry;

  // Set the number of structural nodes linked to each aerodynamic node
  nn = num_nearest;

  // Set the global weighting decay parameter
  global_beta = beta;

  // Initialize the aerostuctural connectivity
  global_conn = NULL;
  global_W = NULL;

  // Initialize the load transfer data
  global_xs0bar = NULL;
  global_R = NULL;
  global_S = NULL;

  // Initialize the Jacobian-vector product data
  global_M1 = NULL;
  global_ipiv = NULL;

  // Initialize object id
  object_id = TransferScheme::object_count++;

  // Inintialize the indicator of a mesh update
  mesh_update = 0;

  // Notify user of the type of transfer scheme they are using
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: Creating scheme of type MELD...\n",
           object_id);
  }
}

MELD::~MELD() {
  // Free the aerostructural connectivity data
  if (global_conn) {
    delete[] global_conn;
  }

  // Free the load transfer data
  if (global_W) {
    delete[] global_W;
  }
  if (global_xs0bar) {
    delete[] global_xs0bar;
  }
  if (global_R) {
    delete[] global_R;
  }
  if (global_S) {
    delete[] global_S;
  }

  // Free the Jacobian-vector product data
  if (global_M1) {
    delete[] global_M1;
  }
  if (global_ipiv) {
    delete[] global_ipiv;
  }

  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: freeing MELD data...\n", object_id);
  }
}

void MELD::setStructNodes(const F2FScalar *struct_X, int struct_nnodes) {
  // Free the structural data if any is allocated
  if (Xs_local) {
    delete[] Xs_local;
    Xs_local = NULL;
  }

  ns_local = struct_nnodes;
  Xs_local = new F2FScalar[3 * ns_local];
  memcpy(Xs_local, struct_X, 3 * ns_local * sizeof(F2FScalar));

  mesh_update = 1;
}

/*
  Collect a structural vector to create a global image then distribute to the
  aerodynamic processors
*/
void MELD::collectStructuralVector(const F2FScalar *local, F2FScalar *global) {
  // Collect how many structural nodes every processor has
  if (struct_comm != MPI_COMM_NULL) {
    int struct_nprocs;
    int struct_rank;
    MPI_Comm_size(struct_comm, &struct_nprocs);
    MPI_Comm_rank(struct_comm, &struct_rank);

    int *ns_list = new int[struct_nprocs];
    memset(ns_list, 0, struct_nprocs * sizeof(int));

    MPI_Gather(&ns_local, 1, MPI_INT, ns_list, 1, MPI_INT, struct_root,
               struct_comm);

    // Collect the structural nodes on the master
    int send_size = ns_local * 3;
    int *disps = new int[struct_nprocs];
    memset(disps, 0, struct_nprocs * sizeof(int));

    if (struct_rank == struct_root) {
      for (int proc = 0; proc < struct_nprocs; proc++) {
        ns_list[proc] *= 3;
        if (proc > 0) {
          disps[proc] = disps[proc - 1] + ns_list[proc - 1];
        }
      }
    }

    MPI_Gatherv(local, send_size, F2F_MPI_TYPE, global, ns_list, disps,
                F2F_MPI_TYPE, 0, struct_comm);

    delete[] ns_list;
    delete[] disps;
  }

  // Pass the global list to all the processors
  MPI_Bcast(global, 3 * ns, F2F_MPI_TYPE, struct_root, global_comm);
}

/*
  Reduce vector to get the total across all aero procs then distribute to the
  structural processors
*/
void MELD::distributeStructuralVector(F2FScalar *global, F2FScalar *local) {
  // Get the contributions from each aero processor
  MPI_Allreduce(MPI_IN_PLACE, global, ns * 3, F2F_MPI_TYPE, MPI_SUM,
                global_comm);

  // Collect how many nodes each structural processor has
  if (struct_comm != MPI_COMM_NULL) {
    int struct_nprocs;
    int struct_rank;
    MPI_Comm_size(struct_comm, &struct_nprocs);
    MPI_Comm_rank(struct_comm, &struct_rank);

    int *ns_list = new int[struct_nprocs];

    MPI_Gather(&ns_local, 1, MPI_INT, ns_list, 1, MPI_INT, 0, struct_comm);

    // Distribute to the structural processors
    int *disps = new int[struct_nprocs];
    memset(disps, 0, struct_nprocs * sizeof(int));

    if (struct_rank == 0) {
      for (int proc = 0; proc < struct_nprocs; proc++) {
        ns_list[proc] *= 3;
        if (proc > 0) {
          disps[proc] = disps[proc - 1] + ns_list[proc - 1];
        }
      }
    }

    MPI_Scatterv(global, ns_list, disps, F2F_MPI_TYPE, local, ns_local * 3,
                 F2F_MPI_TYPE, 0, struct_comm);

    delete[] ns_list;
    delete[] disps;
  }
}

void MELD::distributeStructuralMesh() {
  MPI_Allreduce(MPI_IN_PLACE, &mesh_update, 1, MPI_INT, MPI_SUM, global_comm);
  if (mesh_update > 0) {
    ns = 0;
    if (struct_comm != MPI_COMM_NULL) {
      MPI_Reduce(&ns_local, &ns, 1, MPI_INT, MPI_SUM, 0, struct_comm);
    }

    MPI_Bcast(&ns, 1, MPI_INT, struct_root, global_comm);

    // Allocate memory for structural data, initialize displacement array
    if (Xs) {
      delete[] Xs;
    }
    if (Us) {
      delete[] Us;
    }

    Xs = new F2FScalar[3 * ns];
    memset(Xs, 0, 3 * ns * sizeof(F2FScalar));

    Us = new F2FScalar[3 * ns];
    memset(Us, 0, 3 * ns * sizeof(F2FScalar));

    collectStructuralVector(Xs_local, Xs);
    if (Xs_local) {
      delete[] Xs_local;
      Xs_local = NULL;
    }
    mesh_update = 0;
  }
}

/*
  Set aerostructural connectivity, compute weights, and allocate memory needed
  for transfers and products
*/
void MELD::initialize() {
  // global number of structural nodes
  distributeStructuralMesh();

  // Check that user doesn't set more nearest nodes than exist in total
  if (nn > ns) {
    nn = ns;
  }

  // Create aerostructural connectivity
  global_conn = new int[nn * na];
  setAeroStructConn(global_conn);

  // Allocate and compute the weights
  global_W = new F2FScalar[nn * na];
  computeWeights(global_W);

  // Allocate and initialize load transfer variables
  global_xs0bar = new F2FScalar[3 * na];
  global_R = new F2FScalar[9 * na];
  global_S = new F2FScalar[9 * na];

  // Allocate and initialize Jacobian-vector product variables
  global_M1 = new F2FScalar[15 * 15 * na];
  global_ipiv = new int[15 * na];
}

/*
  Builds aerostructural connectivity through LocatePoint search, linking each
  aerodynamic node with a specified number of nearest structural nodes

  Returns
  --------
  conn : aerostructural connectivity

*/
void MELD::setAeroStructConn(int *conn) {
  // Copy or duplicate and reflect the unique structural nodes
  F2FScalar *Xs_dup = NULL;
  int num_locate_nodes = 0;
  int *locate_to_reflected_index = NULL;

  if (isymm > -1) {
    Xs_dup = new F2FScalar[6 * ns];
    memcpy(Xs_dup, Xs, 3 * ns * sizeof(F2FScalar));

    double tol = 1e-7;
    for (int k = 0; k < ns; k++) {
      if (fabs(F2FRealPart(Xs_dup[3 * k + isymm])) > tol) {
        // Node is not on the plane of symmetry, so copy and...
        memcpy(&Xs_dup[3 * (ns + k)], &Xs_dup[3 * k], 3 * sizeof(F2FScalar));
        // reflect
        Xs_dup[3 * (ns + k) + isymm] *= -1.0;
      } else {
        Xs_dup[3 * (ns + k) + 0] = -(k + 1) * 9.0e9;
        Xs_dup[3 * (ns + k) + 1] = -(k + 1) * 9.0e9 + 1;
        Xs_dup[3 * (ns + k) + 2] = -(k + 1) * 9.0e9 + 2;
      }
    }
    num_locate_nodes = 2 * ns;
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
    memcpy(&conn[nn * i], indx, nn * sizeof(int));
  }

  // Free the duplicate array
  delete[] Xs_dup;

  // Delete the LocatePoint object and release memory
  delete[] indx;
  delete[] dist;
  delete locator;
}

/*
  Computes weights of structural nodes

  Returns
  --------
  W : weights

*/
void MELD::computeWeights(F2FScalar *W) {
  for (int i = 0; i < na; i++) {
    const F2FScalar *xa0 = &Xa[3 * i];
    const int *local_conn = &global_conn[i * nn];
    F2FScalar *w = &W[i * nn];

    // Compute the weights based on the difference between the aerodynamic point
    // location and the local undeformed structural point
    F2FScalar wtotal = 0.0;
    F2FScalar v[3];

    // Find the average squared distance for normalization
    F2FScalar v2_avg = 0.0;
    F2FScalar dist;

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
    if (F2FRealPart(v2_avg) < 1.0e-7) v2_avg = 1.0e-7;

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
      w[j] = exp(-global_beta * vec_dot(v, v) / v2_avg);
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
  Computes the displacements of aerodynamic surface nodes by fitting an
  optimal rigid rotation and translation to the displacement of the set of
  structural nodes nearest each aerodynamic surface node

  Arguments
  ---------
  struct_disps : structural node displacements

  Returns
  -------
  aero_disps   : aerodynamic node displacements
*/
void MELD::transferDisps(const F2FScalar *struct_disps, F2FScalar *aero_disps) {
  // Check if struct nodes locations need to be redistributed
  distributeStructuralMesh();

  // Copy prescribed displacements into displacement vector
  collectStructuralVector(struct_disps, Us);

  // Zero the outputs
  memset(global_xs0bar, 0.0, 3 * na * sizeof(F2FScalar));
  memset(global_R, 0.0, 9 * na * sizeof(F2FScalar));
  memset(global_S, 0.0, 9 * na * sizeof(F2FScalar));
  memset(aero_disps, 0.0, 3 * na * sizeof(F2FScalar));

  // Add structural displacments to structural node locations
  F2FScalar *Xsd = new F2FScalar[3 * ns];
  for (int j = 0; j < 3 * ns; j++) {
    Xsd[j] = Xs[j] + Us[j];
  }

  for (int i = 0; i < na; i++) {
    const F2FScalar *xa0 = &Xa[3 * i];

    // Compute the centroids of the original and displaced sets of nodes
    F2FScalar *xs0bar = &global_xs0bar[3 * i];
    const int *local_conn = &global_conn[i * nn];
    const F2FScalar *W = &global_W[i * nn];
    computeCentroid(local_conn, W, Xs, xs0bar);

    F2FScalar xsbar[3];
    computeCentroid(local_conn, W, Xsd, xsbar);

    // Compute the covariance matrix
    F2FScalar H[9];
    computeCovariance(Xs, Xsd, local_conn, W, xs0bar, xsbar, H);

    // Compute the optimal rotation
    computeRotation(H, &global_R[9 * i], &global_S[9 * i]);

    // Form the vector r from the initial centroid to the aerodynamic surface
    // node
    F2FScalar r[3];
    vec_diff(xs0bar, xa0, r);

    // Rotate r vector using rotation matrix
    const F2FScalar *R = &global_R[9 * i];
    F2FScalar rho[3];
    rho[0] = R[0] * r[0] + R[3] * r[1] + R[6] * r[2];
    rho[1] = R[1] * r[0] + R[4] * r[1] + R[7] * r[2];
    rho[2] = R[2] * r[0] + R[5] * r[1] + R[8] * r[2];

    // Add rotated vector to centroid of second set to obtain final location
    F2FScalar xa[3];  // location of displaced aerodynamic node
    F2FScalar *ua = &aero_disps[3 * i];  // displacement of aerodynamic node
    vec_add(xsbar, rho, xa);
    vec_diff(xa0, xa, ua);
  }

  // Free memory
  delete[] Xsd;
}

/*
  Computes centroids of set of structural nodes

  Arguments
  ----------
  local_conn : IDs of structural nodes in set
  W : array of local weights
  X : set of all structural nodes

  Returns
  --------
  xsbar : centroid
*/
void MELD::computeCentroid(const int *local_conn, const F2FScalar *W,
                           const F2FScalar *X, F2FScalar *xsbar) {
  memset(xsbar, 0, 3 * sizeof(F2FScalar));
  for (int j = 0; j < nn; j++) {
    if (local_conn[j] < ns) {
      const F2FScalar *xs = &X[3 * local_conn[j]];

      for (int k = 0; k < 3; k++) {
        xsbar[k] += W[j] * xs[k];
      }
    } else {
      F2FScalar rxs[3];
      memcpy(rxs, &X[3 * (local_conn[j] - ns)], 3 * sizeof(F2FScalar));
      rxs[isymm] *= -1.0;

      for (int k = 0; k < 3; k++) {
        xsbar[k] += W[j] * rxs[k];
      }
    }
  }
}

/*
  Computes covariance matrix from initial and displaced sets of structural nodes

  Arguments
  ---------
  X : initial structural node locations
  Xd : displaced structural node locations
  local_conn : IDs of structural nodes in set
  W : array of local weights
  xs0bar : initial centroid
  xsbar : displaced centroid

  Returns
  -------
  H : covariance matrix
*/
void MELD::computeCovariance(const F2FScalar *X, const F2FScalar *Xd,
                             const int *local_conn, const F2FScalar *W,
                             const F2FScalar *xs0bar, const F2FScalar *xsbar,
                             F2FScalar *H) {
  // Form the covariance matrix of the two point sets
  memset(H, 0, 9 * sizeof(F2FScalar));

  for (int j = 0; j < nn; j++) {
    F2FScalar q[3];  // vector from centroid to node
    F2FScalar p[3];  // vector from diplaced centroid to displaced node

    if (local_conn[j] < ns) {
      const F2FScalar *xs0 = &X[3 * local_conn[j]];
      const F2FScalar *xs = &Xd[3 * local_conn[j]];

      vec_diff(xs0bar, xs0, q);
      vec_diff(xsbar, xs, p);
    } else {
      // We're dealing with a reflected node/displacement
      F2FScalar rxs0[3];  // Reflected xs0
      F2FScalar rxs[3];   // Reflected xs

      // Copy the node locations
      memcpy(rxs0, &X[3 * (local_conn[j] - ns)], 3 * sizeof(F2FScalar));
      memcpy(rxs, &Xd[3 * (local_conn[j] - ns)], 3 * sizeof(F2FScalar));

      // Reflect the node locations about the axis of symmetry
      rxs0[isymm] *= -1.0;
      rxs[isymm] *= -1.0;

      vec_diff(xs0bar, rxs0, q);
      vec_diff(xsbar, rxs, p);
    }

    // H_{mn} = sum_{j}^{N} w^{(j)} p_{m}^{(j)} q_{n}^{(j)}

    H[0] += W[j] * p[0] * q[0];
    H[1] += W[j] * p[1] * q[0];
    H[2] += W[j] * p[2] * q[0];

    H[3] += W[j] * p[0] * q[1];
    H[4] += W[j] * p[1] * q[1];
    H[5] += W[j] * p[2] * q[1];

    H[6] += W[j] * p[0] * q[2];
    H[7] += W[j] * p[1] * q[2];
    H[8] += W[j] * p[2] * q[2];
  }
}

/*
  Computes the loads on all structural nodes consistently and conservatively
  from loads on aerodynamic surface nodes

  Arguments
  ---------
  aero_loads   : loads on aerodynamic surface nodes

  Returns
  -------
  struct_loads : loads on structural nodes
*/
void MELD::transferLoads(const F2FScalar *aero_loads, F2FScalar *struct_loads) {
  // Copy prescribed aero loads into member variable
  memcpy(Fa, aero_loads, 3 * na * sizeof(F2FScalar));

  // Zero struct loads
  F2FScalar *struct_loads_global = new F2FScalar[3 * ns];
  memset(struct_loads_global, 0, 3 * ns * sizeof(F2FScalar));

  // Loop over all aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Compute vector d from centroid to aero node
    const F2FScalar *xa0 = &Xa[3 * i];
    const F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, xa0, r);

    // Compute X
    const F2FScalar *R = &global_R[9 * i];
    const F2FScalar *S = &global_S[9 * i];
    F2FScalar *M1 = &global_M1[15 * 15 * i];
    assembleM1(R, S, M1);

    int *ipiv = &global_ipiv[15 * i];
    int m = 15, info = 0;
    LAPACKgetrf(&m, &m, M1, &m, ipiv, &info);

    const F2FScalar *fa = &Fa[3 * i];
    F2FScalar x[] = {-fa[0] * r[0], -fa[1] * r[0], -fa[2] * r[0],
                     -fa[0] * r[1], -fa[1] * r[1], -fa[2] * r[1],
                     -fa[0] * r[2], -fa[1] * r[2], -fa[2] * r[2],
                     0.0,           0.0,           0.0,
                     0.0,           0.0,           0.0};
    int one = 1;
    info = 0;
    LAPACKgetrs("N", &m, &one, M1, &m, ipiv, x, &m, &info);
    F2FScalar X[] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]};

    // Compute load contribution of aerodynamic surface node to structural node
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[i * nn + j];

      if (indx < ns) {
        // Compute vector q from centroid to structural node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar q[3];
        vec_diff(xs0bar, xs0, q);

        const F2FScalar w = global_W[nn * i + j];
        F2FScalar *fs = &struct_loads_global[3 * indx];

        // fs = w*(X^{T}*q + w*fa)
        fs[0] += w * (X[0] * q[0] + X[1] * q[1] + X[2] * q[2] + fa[0]);
        fs[1] += w * (X[3] * q[0] + X[4] * q[1] + X[5] * q[2] + fa[1]);
        fs[2] += w * (X[6] * q[0] + X[7] * q[1] + X[8] * q[2] + fa[2]);
      } else {
        indx -= ns;

        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        const F2FScalar w = global_W[nn * i + j];
        F2FScalar *fs = &struct_loads_global[3 * indx];

        F2FScalar rfs[3];
        rfs[0] = w * (X[0] * q[0] + X[1] * q[1] + X[2] * q[2] + fa[0]);
        rfs[1] = w * (X[3] * q[0] + X[4] * q[1] + X[5] * q[2] + fa[1]);
        rfs[2] = w * (X[6] * q[0] + X[7] * q[1] + X[8] * q[2] + fa[2]);
        rfs[isymm] *= -1.0;

        // fs = w*(X^{T}*q + w*fa)
        fs[0] += rfs[0];
        fs[1] += rfs[1];
        fs[2] += rfs[2];
      }
    }
  }

  // distribute the structural loads
  distributeStructuralVector(struct_loads_global, struct_loads);
  delete[] struct_loads_global;
}

/*
  Apply the action of the displacement transfer w.r.t structural displacments
  Jacobian to the input vector

  Arguments
  ---------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void MELD::applydDduS(const F2FScalar *vecs, F2FScalar *prods) {
  // Make a global image of the input vector
  F2FScalar *vecs_global = new F2FScalar[3 * ns];
  collectStructuralVector(vecs, vecs_global);

  // Zero array of Jacobian-vector products every call
  memset(prods, 0, 3 * na * sizeof(F2FScalar));

  // Loop over all aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    F2FScalar *prod = &prods[3 * i];

    // Compute vector r from centroid to aero node
    const F2FScalar *xa0 = &Xa[3 * i];
    const F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, xa0, r);

    // Compute XX
    const F2FScalar *R = &global_R[9 * i];
    const F2FScalar *S = &global_S[9 * i];
    F2FScalar M1[15 * 15];
    assembleM1(R, S, M1);
    int ipiv[15];
    int m = 15, info = 0;
    LAPACKgetrf(&m, &m, M1, &m, ipiv, &info);
    F2FScalar x[15];
    F2FScalar XX[9 * 3];

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[0] -= r[0];
    x[3] -= r[1];
    x[6] -= r[2];
    int nrhs = 1;
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    memcpy(&XX[0], x, 9 * sizeof(F2FScalar));

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[1] -= r[0];
    x[4] -= r[1];
    x[7] -= r[2];
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    memcpy(&XX[9], x, 9 * sizeof(F2FScalar));

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[2] -= r[0];
    x[5] -= r[1];
    x[8] -= r[2];
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    memcpy(&XX[18], x, 9 * sizeof(F2FScalar));

    // Loop over linked structural nodes and add up nonzero contributions to
    // Jacobian-vector product
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      // Compute vector q from centroid to structural node and get components
      // of vector input corresponding to current structural node
      F2FScalar q[3];
      F2FScalar v[3];
      if (indx < ns) {
        const F2FScalar *xs0 = &Xs[3 * indx];
        vec_diff(xs0bar, xs0, q);
        memcpy(v, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
      } else {
        indx -= ns;
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(xs0bar, rxs0, q);
        memcpy(v, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
        v[isymm] *= -1.0;
      }

      // Compute each component of the Jacobian vector product as follows:
      // Jv[k] = w * [ q[0] q[1] q[2] ][ X[0] X[3] X[6] ][ v[0] ] + w*v[k]
      //                               [ X[1] X[4] X[7] ][ v[1] ]
      //                               [ X[2] X[5] X[8] ][ v[2] ]
      F2FScalar w = global_W[nn * i + j];

      for (int k = 0; k < 3; k++) {
        F2FScalar *X = &XX[9 * k];
        prod[k] -= w * (q[0] * (X[0] * v[0] + X[3] * v[1] + X[6] * v[2]) +
                        q[1] * (X[1] * v[0] + X[4] * v[1] + X[7] * v[2]) +
                        q[2] * (X[2] * v[0] + X[5] * v[1] + X[8] * v[2])) +
                   w * v[k];
      }
    }
  }

  // Clean up the allocated memory
  delete[] vecs_global;
}

/*
  Apply the action of the displacement transfer w.r.t structural displacements
  transpose Jacobian to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void MELD::applydDduSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  // Zero array of transpose Jacobian-vector products every call
  F2FScalar *prods_global = new F2FScalar[3 * ns];
  memset(prods_global, 0, 3 * ns * sizeof(F2FScalar));

  // Loop over aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    const F2FScalar *v = &vecs[3 * i];

    // Compute vector r from centroid to aero node
    const F2FScalar *xa0 = &Xa[3 * i];
    const F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, xa0, r);

    // Compute XX
    const F2FScalar *R = &global_R[9 * i];
    const F2FScalar *S = &global_S[9 * i];
    F2FScalar M1[15 * 15];
    assembleM1(R, S, M1);
    int ipiv[15];
    int m = 15, info = 0;
    LAPACKgetrf(&m, &m, M1, &m, ipiv, &info);
    F2FScalar x[15];
    F2FScalar XX[9 * 3];

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[0] -= r[0];
    x[3] -= r[1];
    x[6] -= r[2];
    int nrhs = 1;
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    memcpy(&XX[0], x, 9 * sizeof(F2FScalar));

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[1] -= r[0];
    x[4] -= r[1];
    x[7] -= r[2];
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    memcpy(&XX[9], x, 9 * sizeof(F2FScalar));

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[2] -= r[0];
    x[5] -= r[1];
    x[8] -= r[2];
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    memcpy(&XX[18], x, 9 * sizeof(F2FScalar));

    // Loop over linked structural nodes and add up nonzero contributions to
    // Jacobian-vector product
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        // Compute vector q from centroid to structural node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar q[3];
        vec_diff(xs0bar, xs0, q);

        // Compute each component of the transpose Jacobian-vector product as
        // follows:
        // J^{T}*v = w[X_{1}^{T}*q X_{2}^{T}*q X_{3}^{T}*q]*v + w*v
        F2FScalar w = global_W[nn * i + j];
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= w * v[0];
        prod[1] -= w * v[1];
        prod[2] -= w * v[2];

        for (int k = 0; k < 3; k++) {
          F2FScalar *X = &XX[9 * k];
          prod[0] -= w * (X[0] * q[0] + X[1] * q[1] + X[2] * q[2]) * v[k];
          prod[1] -= w * (X[3] * q[0] + X[4] * q[1] + X[5] * q[2]) * v[k];
          prod[2] -= w * (X[6] * q[0] + X[7] * q[1] + X[8] * q[2]) * v[k];
        }
      } else {
        indx -= ns;

        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        F2FScalar w = global_W[nn * i + j];
        F2FScalar rprod[] = {0.0, 0.0, 0.0};
        rprod[0] += w * v[0];
        rprod[1] += w * v[1];
        rprod[2] += w * v[2];

        for (int k = 0; k < 3; k++) {
          F2FScalar *X = &XX[9 * k];
          rprod[0] += w * (X[0] * q[0] + X[1] * q[1] + X[2] * q[2]) * v[k];
          rprod[1] += w * (X[3] * q[0] + X[4] * q[1] + X[5] * q[2]) * v[k];
          rprod[2] += w * (X[6] * q[0] + X[7] * q[1] + X[8] * q[2]) * v[k];
        }
        rprod[isymm] *= -1.0;

        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= rprod[0];
        prod[1] -= rprod[1];
        prod[2] -= rprod[2];
      }
    }
  }
  // distribute the results to the structural processors
  distributeStructuralVector(prods_global, prods);

  // clean up allocated memory
  delete[] prods_global;
}

/*
  Apply the action of the load transfer w.r.t structural displacements Jacobian
  to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector

*/
void MELD::applydLduS(const F2FScalar *vecs, F2FScalar *prods) {
  F2FScalar *vecs_global = new F2FScalar[3 * ns];
  collectStructuralVector(vecs, vecs_global);

  // Zero products
  F2FScalar *prods_global = new F2FScalar[3 * ns];
  memset(prods_global, 0, 3 * ns * sizeof(F2FScalar));

  // Loop over aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Compute vector r from centroid to aero node
    const F2FScalar *xa0 = &Xa[3 * i];
    const F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, xa0, r);

    // Get the load on the aerodynamic surface node
    const F2FScalar *fa = &Fa[3 * i];

    // Recompute X and Y
    const F2FScalar *M1 = &global_M1[15 * 15 * i];
    const int *ipiv = &global_ipiv[15 * i];
    F2FScalar x[] = {-fa[0] * r[0], -fa[1] * r[0], -fa[2] * r[0],
                     -fa[0] * r[1], -fa[1] * r[1], -fa[2] * r[1],
                     -fa[0] * r[2], -fa[1] * r[2], -fa[2] * r[2],
                     0.0,           0.0,           0.0,
                     0.0,           0.0,           0.0};
    int m = 15, nrhs = 1, info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar XT[] = {x[0], x[3], x[6], x[1], x[4], x[7], x[2], x[5], x[8]};
    F2FScalar nY[] = {-x[9],  -x[10], -x[11], -x[10], -x[12],
                      -x[13], -x[11], -x[13], -x[14]};

    // Assemble X and Y into matrix M2
    F2FScalar M2[15 * 15];
    assembleM1(XT, nY, M2);

    // Assemble matrix M3 from R and S
    F2FScalar M3[15 * 15];
    const F2FScalar *R = &global_R[9 * i];
    const F2FScalar *S = &global_S[9 * i];
    assembleM3(R, S, M3);

    // Build right-hand side of first system to be solved
    F2FScalar z2[15];
    memset(z2, 0.0, 15 * sizeof(F2FScalar));
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      // Get vector q and subset of input vector
      F2FScalar q[3];
      F2FScalar v[3];
      if (indx < ns) {
        const F2FScalar *xs0 = &Xs[3 * indx];
        vec_diff(xs0bar, xs0, q);
        memcpy(v, &vecs[3 * indx], 3 * sizeof(F2FScalar));
      } else {
        indx -= ns;
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(xs0bar, rxs0, q);
        memcpy(v, &vecs[3 * indx], 3 * sizeof(F2FScalar));
        v[isymm] *= -1.0;
      }

      F2FScalar w = global_W[nn * i + j];

      z2[0] -= w * q[0] * v[0];
      z2[1] -= w * q[1] * v[0];
      z2[2] -= w * q[2] * v[0];
      z2[3] -= w * q[0] * v[1];
      z2[4] -= w * q[1] * v[1];
      z2[5] -= w * q[2] * v[1];
      z2[6] -= w * q[0] * v[2];
      z2[7] -= w * q[1] * v[2];
      z2[8] -= w * q[2] * v[2];
    }

    // Solve the first linear system
    int ipiv3[15];
    info = 0;
    LAPACKgetrf(&m, &m, M3, &m, ipiv3, &info);
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M3, &m, ipiv3, z2, &m, &info);

    // Compute right-hand side of second system
    F2FScalar z1[15];
    F2FScalar alpha = -1.0, beta = 0.0;
    int inc = 1;
    BLASgemv("N", &m, &m, &alpha, M2, &m, z2, &inc, &beta, z1, &inc);

    // Solve the second system
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, z1, &m, &info);

    // Extract ZH
    F2FScalar ZH[9] = {z1[0], z1[1], z1[2], z1[3], z1[4],
                       z1[5], z1[6], z1[7], z1[8]};

    // Loop over linked structural nodes and add contributions from aerodynamic
    // surface node to global structural loads
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        // Compute vector q from centroid to structural node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar q[3];
        vec_diff(xs0bar, xs0, q);

        // Compute load contribution of aerodynamic surface node to structural
        // node
        const F2FScalar w = global_W[nn * i + j];
        F2FScalar *prod = &prods_global[3 * indx];

        // prod = w * [ ZH[0] ZH[1] ZH[2] ][ q[0] ]
        //            [ ZH[3] ZH[4] ZH[5] ][ q[1] ]
        //            [ ZH[6] ZH[7] ZH[8] ][ q[2] ]
        prod[0] -= w * (ZH[0] * q[0] + ZH[1] * q[1] + ZH[2] * q[2]);
        prod[1] -= w * (ZH[3] * q[0] + ZH[4] * q[1] + ZH[5] * q[2]);
        prod[2] -= w * (ZH[6] * q[0] + ZH[7] * q[1] + ZH[8] * q[2]);
      } else {
        indx -= ns;

        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        const F2FScalar w = global_W[nn * i + j];
        F2FScalar *prod = &prods_global[3 * indx];

        F2FScalar rprod[3];
        rprod[0] = w * (ZH[0] * q[0] + ZH[1] * q[1] + ZH[2] * q[2]);
        rprod[1] = w * (ZH[3] * q[0] + ZH[4] * q[1] + ZH[5] * q[2]);
        rprod[2] = w * (ZH[6] * q[0] + ZH[7] * q[1] + ZH[8] * q[2]);
        rprod[isymm] *= -1.0;

        prod[0] -= rprod[0];
        prod[1] -= rprod[1];
        prod[2] -= rprod[2];
      }
    }
  }

  // distribute the results to the structural processors
  distributeStructuralVector(prods_global, prods);

  // clean up allocated memory
  delete[] vecs_global;
  delete[] prods_global;
}

/*
  Apply the action of the load transfer w.r.t structural displacements
  transpose Jacobian to the input vector

  Arguments
  ---------
  vecs  : input vector

  Returns
  --------
  prods : output vector

*/
void MELD::applydLduSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  F2FScalar *vecs_global = new F2FScalar[3 * ns];
  collectStructuralVector(vecs, vecs_global);

  // Zero products every call
  F2FScalar *prods_global = new F2FScalar[3 * ns];
  memset(prods_global, 0, 3 * ns * sizeof(F2FScalar));

  // Loop over all aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Compute vector r from centroid to aero node
    const F2FScalar *xa0 = &Xa[3 * i];
    const F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, xa0, r);

    // Get the load on the aerodynamic surface node
    const F2FScalar *fa = &Fa[3 * i];

    // Recompute X and Y
    const F2FScalar *M1 = &global_M1[15 * 15 * i];
    const int *ipiv = &global_ipiv[15 * i];
    F2FScalar x[] = {-fa[0] * r[0], -fa[1] * r[0], -fa[2] * r[0],
                     -fa[0] * r[1], -fa[1] * r[1], -fa[2] * r[1],
                     -fa[0] * r[2], -fa[1] * r[2], -fa[2] * r[2],
                     0.0,           0.0,           0.0,
                     0.0,           0.0,           0.0};
    int m = 15, nrhs = 1, info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar XT[] = {x[0], x[3], x[6], x[1], x[4], x[7], x[2], x[5], x[8]};
    F2FScalar nY[] = {-x[9],  -x[10], -x[11], -x[10], -x[12],
                      -x[13], -x[11], -x[13], -x[14]};

    // Assemble X and Y into matrix M2
    F2FScalar M2[15 * 15];
    assembleM1(XT, nY, M2);

    // Assemble matrix M3 from R and S
    F2FScalar M3[15 * 15];
    const F2FScalar *R = &global_R[9 * i];
    const F2FScalar *S = &global_S[9 * i];
    assembleM3(R, S, M3);

    // Build right-hand side of first system to be solved
    F2FScalar y2[15];
    memset(y2, 0.0, 15 * sizeof(F2FScalar));
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      // Get vector q and subset of input vector
      F2FScalar q[3];
      F2FScalar v[3];
      if (indx < ns) {
        const F2FScalar *xs0 = &Xs[3 * indx];
        vec_diff(xs0bar, xs0, q);
        memcpy(v, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
      } else {
        indx -= ns;
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(xs0bar, rxs0, q);
        memcpy(v, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
        v[isymm] *= -1.0;
      }

      F2FScalar w = global_W[nn * i + j];

      y2[0] -= w * q[0] * v[0];
      y2[1] -= w * q[1] * v[0];
      y2[2] -= w * q[2] * v[0];
      y2[3] -= w * q[0] * v[1];
      y2[4] -= w * q[1] * v[1];
      y2[5] -= w * q[2] * v[1];
      y2[6] -= w * q[0] * v[2];
      y2[7] -= w * q[1] * v[2];
      y2[8] -= w * q[2] * v[2];
    }

    // Solve the first linear system
    const char *t = "T";
    int ipiv3[15];
    info = 0;
    LAPACKgetrf(&m, &m, M3, &m, ipiv3, &info);
    info = 0;
    LAPACKgetrs(t, &m, &nrhs, M3, &m, ipiv, y2, &m, &info);

    // Compute right-hand side of second system
    F2FScalar y1[15];
    F2FScalar alpha = -1.0, beta = 0.0;
    int inc = 1;
    BLASgemv(t, &m, &m, &alpha, M2, &m, y2, &inc, &beta, y1, &inc);

    // Solve the second system
    info = 0;
    LAPACKgetrs(t, &m, &nrhs, M1, &m, ipiv, y1, &m, &info);

    // Extract YF
    F2FScalar YF[] = {y1[0], y1[1], y1[2], y1[3], y1[4],
                      y1[5], y1[6], y1[7], y1[8]};

    // Loop over linked structural nodes and add contributions from aerodynamic
    // surface node to global structural loads
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        // Compute vector q from centroid to structural node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar q[3];
        vec_diff(xs0bar, xs0, q);

        // Compute load contribution of aerodynamic surface node to structural
        // node
        F2FScalar w = global_W[nn * i + j];
        F2FScalar *prod = &prods_global[3 * indx];

        // prod  = w * [ YF[0] YF[3] YF[6] ][ q[0] ]
        //             [ YF[1] YF[4] YF[7] ][ q[1] ]
        //             [ YF[2] YF[5] YF[8] ][ q[2] ]
        prod[0] -= w * (YF[0] * q[0] + YF[3] * q[1] + YF[6] * q[2]);
        prod[1] -= w * (YF[1] * q[0] + YF[4] * q[1] + YF[7] * q[2]);
        prod[2] -= w * (YF[2] * q[0] + YF[5] * q[1] + YF[8] * q[2]);
      } else {
        indx -= ns;

        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        const F2FScalar w = global_W[nn * i + j];
        F2FScalar *prod = &prods_global[3 * indx];

        F2FScalar rprod[3];
        rprod[0] = w * (YF[0] * q[0] + YF[3] * q[1] + YF[6] * q[2]);
        rprod[1] = w * (YF[1] * q[0] + YF[4] * q[1] + YF[7] * q[2]);
        rprod[2] = w * (YF[2] * q[0] + YF[5] * q[1] + YF[8] * q[2]);
        rprod[isymm] *= -1.0;

        prod[0] -= rprod[0];
        prod[1] -= rprod[1];
        prod[2] -= rprod[2];
      }
    }
  }

  // distribute the results to the structural processors
  distributeStructuralVector(prods_global, prods);

  // clean up allocated memory
  delete[] vecs_global;
  delete[] prods_global;
}

/*
  Builds the matrix of the linear system to be solved in the process of
  computing the second-order adjoints

  Arguments
  ----------
  R  : rotation matrix
  S  : symmetric matrix

  Returns
  --------
  M3 : matrix system

*/
void MELD::assembleM3(const F2FScalar *R, const F2FScalar *S, F2FScalar *M3) {
  // Set the entries to zero
  memset(M3, 0, 15 * 15 * sizeof(F2FScalar));

  /*
  M3 = [ A3 C3 ]
       [ B3 0  ]

  A3 = -kron(I, S)                                      9x9
  B3 = D*(kron(I, R^T)*T + kron(R^T, I))                6x9
  C3 = kron(R, I)*Dstar                                 9x6
  */

  // Fill in the elements of M3 corresponding to A3
  // Rows 0-2
  M3[0 + 15 * 0] = -S[0];
  M3[0 + 15 * 1] = -S[3];
  M3[0 + 15 * 2] = -S[6];
  M3[1 + 15 * 0] = -S[1];
  M3[1 + 15 * 1] = -S[4];
  M3[1 + 15 * 2] = -S[7];
  M3[2 + 15 * 0] = -S[2];
  M3[2 + 15 * 1] = -S[5];
  M3[2 + 15 * 2] = -S[8];

  // Rows 3-5
  M3[3 + 15 * 3] = -S[0];
  M3[3 + 15 * 4] = -S[3];
  M3[3 + 15 * 5] = -S[6];
  M3[4 + 15 * 3] = -S[1];
  M3[4 + 15 * 4] = -S[4];
  M3[4 + 15 * 5] = -S[7];
  M3[5 + 15 * 3] = -S[2];
  M3[5 + 15 * 4] = -S[5];
  M3[5 + 15 * 5] = -S[8];

  // Rows 6-8
  M3[6 + 15 * 6] = -S[0];
  M3[6 + 15 * 7] = -S[3];
  M3[6 + 15 * 8] = -S[6];
  M3[7 + 15 * 6] = -S[1];
  M3[7 + 15 * 7] = -S[4];
  M3[7 + 15 * 8] = -S[7];
  M3[8 + 15 * 6] = -S[2];
  M3[8 + 15 * 7] = -S[5];
  M3[8 + 15 * 8] = -S[8];

  // Fill in the elements of M3 corresponding to B3
  // Columns 0-2
  M3[9 + 15 * 0] = 2.0 * R[0];
  M3[10 + 15 * 1] = R[0];
  M3[11 + 15 * 2] = R[0];
  M3[10 + 15 * 0] = R[3];
  M3[12 + 15 * 1] = 2.0 * R[3];
  M3[13 + 15 * 2] = R[3];
  M3[11 + 15 * 0] = R[6];
  M3[13 + 15 * 1] = R[6];
  M3[14 + 15 * 2] = 2.0 * R[6];

  // Columns 3-5
  M3[9 + 15 * 3] = 2.0 * R[1];
  M3[10 + 15 * 4] = R[1];
  M3[11 + 15 * 5] = R[1];
  M3[10 + 15 * 3] = R[4];
  M3[12 + 15 * 4] = 2.0 * R[4];
  M3[13 + 15 * 5] = R[4];
  M3[11 + 15 * 3] = R[7];
  M3[13 + 15 * 4] = R[7];
  M3[14 + 15 * 5] = 2.0 * R[7];

  // Columns 6-8
  M3[9 + 15 * 6] = 2.0 * R[2];
  M3[10 + 15 * 7] = R[2];
  M3[11 + 15 * 8] = R[2];
  M3[10 + 15 * 6] = R[5];
  M3[12 + 15 * 7] = 2.0 * R[5];
  M3[13 + 15 * 8] = R[5];
  M3[11 + 15 * 6] = R[8];
  M3[13 + 15 * 7] = R[8];
  M3[14 + 15 * 8] = 2.0 * R[8];

  // Fill in the elements of M3 corresponding to C3
  // Rows 0-2
  M3[0 + 15 * 9] = R[0];
  M3[0 + 15 * 10] = R[3];
  M3[0 + 15 * 11] = R[6];
  M3[1 + 15 * 10] = R[0];
  M3[1 + 15 * 12] = R[3];
  M3[1 + 15 * 13] = R[6];
  M3[2 + 15 * 11] = R[0];
  M3[2 + 15 * 13] = R[3];
  M3[2 + 15 * 14] = R[6];

  // Rows 3-5
  M3[3 + 15 * 9] = R[1];
  M3[3 + 15 * 10] = R[4];
  M3[3 + 15 * 11] = R[7];
  M3[4 + 15 * 10] = R[1];
  M3[4 + 15 * 12] = R[4];
  M3[4 + 15 * 13] = R[7];
  M3[5 + 15 * 11] = R[1];
  M3[5 + 15 * 13] = R[4];
  M3[5 + 15 * 14] = R[7];

  // Rows 6-8
  M3[6 + 15 * 9] = R[2];
  M3[6 + 15 * 10] = R[5];
  M3[6 + 15 * 11] = R[8];
  M3[7 + 15 * 10] = R[2];
  M3[7 + 15 * 12] = R[5];
  M3[7 + 15 * 13] = R[8];
  M3[8 + 15 * 11] = R[2];
  M3[8 + 15 * 13] = R[5];
  M3[8 + 15 * 14] = R[8];
}

/*
  Apply the action of the displacement transfer w.r.t initial aerodynamic
  surface node locations Jacobian to the right of the transposed input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector

*/
void MELD::applydDdxA0(const F2FScalar *vecs, F2FScalar *prods) {
  for (int i = 0; i < na; i++) {
    // Get vector of adjoint variables and rotation matrix for each aerodynamic
    // node
    const F2FScalar *lam = &vecs[3 * i];
    const F2FScalar *R = &global_R[9 * i];

    // Compute vector-matrix product
    F2FScalar *prod = &prods[3 * i];
    prod[0] = -(lam[0] * (R[0] - 1.0) + lam[1] * R[1] + lam[2] * R[2]);
    prod[1] = -(lam[0] * R[3] + lam[1] * (R[4] - 1.0) + lam[2] * R[5]);
    prod[2] = -(lam[0] * R[6] + lam[1] * R[7] + lam[2] * (R[8] - 1.0));
  }
}

/*
  Apply the action of the displacement transfer w.r.t initial structural node
  locations Jacobian to the right of the transposed input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector

*/
void MELD::applydDdxS0(const F2FScalar *vecs, F2FScalar *prods) {
  // Set products to zero
  F2FScalar *prods_global = new F2FScalar[3 * ns];
  memset(prods_global, 0.0, 3 * ns * sizeof(F2FScalar));

  // Add structural displacments to structural node locations
  F2FScalar *Xsd = new F2FScalar[3 * ns];
  for (int j = 0; j < 3 * ns; j++) {
    Xsd[j] = Xs[j] + Us[j];
  }

  for (int i = 0; i < na; i++) {
    const F2FScalar *lam = &vecs[3 * i];

    // Compute vector r from centroid to aero node
    const F2FScalar *xa0 = &Xa[3 * i];
    const F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, xa0, r);

    // Compute displaced centroid xsbar
    F2FScalar xsbar[3];
    computeCentroid(&global_conn[i * nn], &global_W[i * nn], Xsd, xsbar);

    // Compute X
    const F2FScalar *R = &global_R[9 * i];
    const F2FScalar *S = &global_S[9 * i];
    F2FScalar M1[15 * 15];
    assembleM1(R, S, M1);
    int ipiv[15];
    int m = 15, info = 0;
    LAPACKgetrf(&m, &m, M1, &m, ipiv, &info);
    F2FScalar x[] = {-lam[0] * r[0],
                     -lam[1] * r[0],
                     -lam[2] * r[0],
                     -lam[0] * r[1],
                     -lam[1] * r[1],
                     -lam[2] * r[1],
                     -lam[0] * r[2],
                     -lam[1] * r[2],
                     -lam[2] * r[2],
                     0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0,
                     0.0};
    int nrhs = 1;
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar X[] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]};

    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        // Compute vector q from centroid to structural node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar q[3];
        vec_diff(xs0bar, xs0, q);

        // Compute vector p from displaced centroid to displaced structural node
        F2FScalar *xs = &Xsd[3 * indx];
        F2FScalar p[3];
        vec_diff(xsbar, xs, p);

        // Compute the contribution to the products
        F2FScalar w = global_W[nn * i + j];
        F2FScalar *prod = &prods_global[3 * indx];

        // prod = -w*q^{T}*X - w*p^{T}*X^{T} + w*lam^{T}*(R - I)

        prod[0] += -w * (q[0] * X[0] + q[1] * X[1] + q[2] * X[2]) -
                   w * (X[0] * p[0] + X[3] * p[1] + X[6] * p[2]) +
                   w * (lam[0] * (R[0] - 1.0) + lam[1] * R[1] + lam[2] * R[2]);
        prod[1] += -w * (q[0] * X[3] + q[1] * X[4] + q[2] * X[5]) -
                   w * (X[1] * p[0] + X[4] * p[1] + X[7] * p[2]) +
                   w * (lam[0] * R[3] + lam[1] * (R[4] - 1.0) + lam[2] * R[5]);
        prod[2] += -w * (q[0] * X[6] + q[1] * X[7] + q[2] * X[8]) -
                   w * (X[2] * p[0] + X[5] * p[1] + X[8] * p[2]) +
                   w * (lam[0] * R[6] + lam[1] * R[7] + lam[2] * (R[8] - 1.0));
      } else {
        indx -= ns;

        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        F2FScalar *xs = &Xsd[3 * indx];
        F2FScalar rxs[3];
        memcpy(rxs, xs, 3 * sizeof(F2FScalar));
        rxs[isymm] *= -1.0;

        F2FScalar p[3];
        vec_diff(xsbar, rxs, p);

        F2FScalar w = global_W[nn * i + j];
        F2FScalar *prod = &prods_global[3 * indx];

        F2FScalar rprod[3];
        rprod[0] = -w * (q[0] * X[0] + q[1] * X[1] + q[2] * X[2]) -
                   w * (X[0] * p[0] + X[3] * p[1] + X[6] * p[2]) +
                   w * (lam[0] * (R[0] - 1.0) + lam[1] * R[1] + lam[2] * R[2]);
        rprod[1] = -w * (q[0] * X[3] + q[1] * X[4] + q[2] * X[5]) -
                   w * (X[1] * p[0] + X[4] * p[1] + X[7] * p[2]) +
                   w * (lam[0] * R[3] + lam[1] * (R[4] - 1.0) + lam[2] * R[5]);
        rprod[2] = -w * (q[0] * X[6] + q[1] * X[7] + q[2] * X[8]) -
                   w * (X[2] * p[0] + X[5] * p[1] + X[8] * p[2]) +
                   w * (lam[0] * R[6] + lam[1] * R[7] + lam[2] * (R[8] - 1.0));
        rprod[isymm] *= -1.0;

        prod[0] += rprod[0];
        prod[1] += rprod[1];
        prod[2] += rprod[2];
      }
    }
  }
  // distribute the results to the structural processors
  distributeStructuralVector(prods_global, prods);

  // clean up allocated memory
  delete[] Xsd;
  delete[] prods_global;
}

/*
  Apply the action of the load transfer w.r.t initial aerodynamic surface node
  locations Jacobian to the right of the transposed input vector

  Arguments
  ---------
  vecs  : input vector

  Returns
  --------
  prods : output vector

*/
void MELD::applydLdxA0(const F2FScalar *vecs, F2FScalar *prods) {
  F2FScalar *vecs_global = new F2FScalar[3 * ns];
  collectStructuralVector(vecs, vecs_global);

  // Zero products
  memset(prods, 0, 3 * na * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    const F2FScalar *fa = &Fa[3 * i];
    const F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar *prod = &prods[3 * i];

    // Compute X1, X2, X3
    const F2FScalar *M1 = &global_M1[15 * 15 * i];
    const int *ipiv = &global_ipiv[15 * i];
    int m = 15, nrhs = 1, info = 0;
    F2FScalar x[15];

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[0] -= fa[0];
    x[1] -= fa[1];
    x[2] -= fa[2];
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar X1[] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]};

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[3] -= fa[0];
    x[4] -= fa[1];
    x[5] -= fa[2];
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar X2[] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]};

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[6] -= fa[0];
    x[7] -= fa[1];
    x[8] -= fa[2];
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar X3[] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]};

    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      // Compute vector q from centroid to structural node and get components
      // of vector input corresponding to current structural node
      F2FScalar q[3];
      F2FScalar lam[3];
      if (indx < ns) {
        const F2FScalar *xs0 = &Xs[3 * indx];
        vec_diff(xs0bar, xs0, q);
        memcpy(lam, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
      } else {
        indx -= ns;
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(xs0bar, rxs0, q);
        memcpy(lam, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
        lam[isymm] *= -1.0;
      }

      // Compute the contribution to the products
      const F2FScalar w = global_W[nn * i + j];

      prod[0] -= w * q[0] * (X1[0] * lam[0] + X1[3] * lam[1] + X1[6] * lam[2]) +
                 w * q[1] * (X1[1] * lam[0] + X1[4] * lam[1] + X1[7] * lam[2]) +
                 w * q[2] * (X1[2] * lam[0] + X1[5] * lam[1] + X1[8] * lam[2]);

      prod[1] -= w * q[0] * (X2[0] * lam[0] + X2[3] * lam[1] + X2[6] * lam[2]) +
                 w * q[1] * (X2[1] * lam[0] + X2[4] * lam[1] + X2[7] * lam[2]) +
                 w * q[2] * (X2[2] * lam[0] + X2[5] * lam[1] + X2[8] * lam[2]);

      prod[2] -= w * q[0] * (X3[0] * lam[0] + X3[3] * lam[1] + X3[6] * lam[2]) +
                 w * q[1] * (X3[1] * lam[0] + X3[4] * lam[1] + X3[7] * lam[2]) +
                 w * q[2] * (X3[2] * lam[0] + X3[5] * lam[1] + X3[8] * lam[2]);
    }
  }

  // Clean up the allocated memory
  delete[] vecs_global;
}

/*
  Apply the action of the load transfer w.r.t initial structural node locations
  Jacobian to the right of the transposed input vector

  Arguments
  ---------
  vecs  : input vector

  Returns
  -------
  prods : output vector

*/
void MELD::applydLdxS0(const F2FScalar *vecs, F2FScalar *prods) {
  F2FScalar *vecs_global = new F2FScalar[3 * ns];
  collectStructuralVector(vecs, vecs_global);

  // Zero products
  F2FScalar *prods_global = new F2FScalar[3 * ns];
  memset(prods_global, 0, 3 * ns * sizeof(F2FScalar));

  // Add structural displacments to structural node locations
  F2FScalar *Xsd = new F2FScalar[3 * ns];
  for (int j = 0; j < 3 * ns; j++) {
    Xsd[j] = Xs[j] + Us[j];
  }

  // Loop over aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Compute vector r from centroid to aero node
    const F2FScalar *xa0 = &Xa[3 * i];
    const F2FScalar *xs0bar = &global_xs0bar[3 * i];
    F2FScalar r[3];
    vec_diff(xs0bar, xa0, r);

    // Compute displaced centroid xsbar
    F2FScalar xsbar[3];
    computeCentroid(&global_conn[i * nn], &global_W[i * nn], Xsd, xsbar);

    // Get the load on the aerodynamic surface node
    const F2FScalar *fa = &Fa[3 * i];

    // Recompute X and Y
    const F2FScalar *M1 = &global_M1[15 * 15 * i];
    const int *ipiv = &global_ipiv[15 * i];
    F2FScalar x[] = {-fa[0] * r[0], -fa[1] * r[0], -fa[2] * r[0],
                     -fa[0] * r[1], -fa[1] * r[1], -fa[2] * r[1],
                     -fa[0] * r[2], -fa[1] * r[2], -fa[2] * r[2],
                     0.0,           0.0,           0.0,
                     0.0,           0.0,           0.0};
    int m = 15, nrhs = 1, info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar XT[] = {x[0], x[3], x[6], x[1], x[4], x[7], x[2], x[5], x[8]};
    F2FScalar nY[] = {-x[9],  -x[10], -x[11], -x[10], -x[12],
                      -x[13], -x[11], -x[13], -x[14]};

    // Assemble X and Y into matrix M2
    F2FScalar M2[15 * 15];
    assembleM1(XT, nY, M2);

    // Assemble matrix M3 from R and S
    F2FScalar M3[15 * 15];
    const F2FScalar *R = &global_R[9 * i];
    const F2FScalar *S = &global_S[9 * i];
    assembleM3(R, S, M3);

    // Build right-hand side of first system to be solved
    F2FScalar z2[15];
    memset(z2, 0.0, 15 * sizeof(F2FScalar));
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      // Get vector q, and subset of input vector
      F2FScalar q[3];
      F2FScalar lam[3];
      if (indx < ns) {
        const F2FScalar *xs0 = &Xs[3 * indx];
        vec_diff(xs0bar, xs0, q);
        memcpy(lam, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
      } else {
        indx -= ns;
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(xs0bar, rxs0, q);
        memcpy(lam, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
        lam[isymm] *= -1.0;
      }

      F2FScalar w = global_W[nn * i + j];

      z2[0] -= w * q[0] * lam[0];
      z2[1] -= w * q[1] * lam[0];
      z2[2] -= w * q[2] * lam[0];
      z2[3] -= w * q[0] * lam[1];
      z2[4] -= w * q[1] * lam[1];
      z2[5] -= w * q[2] * lam[1];
      z2[6] -= w * q[0] * lam[2];
      z2[7] -= w * q[1] * lam[2];
      z2[8] -= w * q[2] * lam[2];
    }

    // Solve the first linear system
    int ipiv3[15];
    info = 0;
    LAPACKgetrf(&m, &m, M3, &m, ipiv3, &info);
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M3, &m, ipiv3, z2, &m, &info);

    // Compute right-hand side of second system
    F2FScalar z1[15];
    F2FScalar alpha = -1.0, beta = 0.0;
    int inc = 1;
    BLASgemv("N", &m, &m, &alpha, M2, &m, z2, &inc, &beta, z1, &inc);

    // Solve the second system
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, z1, &m, &info);

    // Extract ZH
    F2FScalar ZH[9] = {z1[0], z1[1], z1[2], z1[3], z1[4],
                       z1[5], z1[6], z1[7], z1[8]};

    // Compute centroid of adjoint variables
    F2FScalar lambar[3];
    computeCentroid(&global_conn[i * nn], &global_W[i * nn], vecs_global,
                    lambar);

    // Compute X1, X2, X3
    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[0] -= fa[0];
    x[1] -= fa[1];
    x[2] -= fa[2];
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar X1[] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]};

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[3] -= fa[0];
    x[4] -= fa[1];
    x[5] -= fa[2];
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar X2[] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]};

    memset(x, 0.0, 15 * sizeof(F2FScalar));
    x[6] -= fa[0];
    x[7] -= fa[1];
    x[8] -= fa[2];
    info = 0;
    LAPACKgetrs("N", &m, &nrhs, M1, &m, ipiv, x, &m, &info);
    F2FScalar X3[] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]};

    // Compute vector for third term
    F2FScalar qXlam[] = {0.0, 0.0, 0.0};
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      // Get vector q and subset of input vector
      F2FScalar q[3];
      F2FScalar lam[3];
      if (indx < ns) {
        const F2FScalar *xs0 = &Xs[3 * indx];
        vec_diff(xs0bar, xs0, q);
        memcpy(lam, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
      } else {
        indx -= ns;
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;
        vec_diff(xs0bar, rxs0, q);
        memcpy(lam, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
        lam[isymm] *= -1.0;
      }

      F2FScalar w = global_W[nn * i + j];
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          qXlam[0] += w * q[m] * X1[m + 3 * n] * lam[n];
          qXlam[1] += w * q[m] * X2[m + 3 * n] * lam[n];
          qXlam[2] += w * q[m] * X3[m + 3 * n] * lam[n];
        }
      }
    }

    // Loop over linked structural nodes and add contributions from aerodynamic
    // surface node to global structural loads
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      if (indx < ns) {
        // Compute vector q from centroid to structural node
        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar q[3];
        vec_diff(xs0bar, xs0, q);

        // Compute vector p from displaced centroid to displaced structural node
        F2FScalar *xs = &Xsd[3 * indx];
        F2FScalar p[3];
        vec_diff(xsbar, xs, p);

        // Compute vector lamp from centroid of adjoint variables to the
        // components of the adjoint variable corresponding to this node
        const F2FScalar *lam = &vecs_global[3 * indx];
        F2FScalar lamp[3];
        vec_diff(lambar, lam, lamp);

        // Take contribution of first term
        F2FScalar w = global_W[nn * i + j];
        F2FScalar *prod = &prods_global[3 * indx];
        prod[0] -= w * (q[0] * ZH[0] + q[1] * ZH[1] + q[2] * ZH[2]) +
                   w * (ZH[0] * p[0] + ZH[3] * p[1] + ZH[6] * p[2]);
        prod[1] -= w * (q[0] * ZH[3] + q[1] * ZH[4] + q[2] * ZH[5]) +
                   w * (ZH[1] * p[0] + ZH[4] * p[1] + ZH[7] * p[2]);
        prod[2] -= w * (q[0] * ZH[6] + q[1] * ZH[7] + q[2] * ZH[8]) +
                   w * (ZH[2] * p[0] + ZH[5] * p[1] + ZH[8] * p[2]);

        // Take contribution of second term
        prod[0] -= w * (lamp[0] * XT[0] + lamp[1] * XT[1] + lamp[2] * XT[2]);
        prod[1] -= w * (lamp[0] * XT[3] + lamp[1] * XT[4] + lamp[2] * XT[5]);
        prod[2] -= w * (lamp[0] * XT[6] + lamp[1] * XT[7] + lamp[2] * XT[8]);

        // Take contribution of third term
        prod[0] += w * qXlam[0];
        prod[1] += w * qXlam[1];
        prod[2] += w * qXlam[2];
      } else {
        indx -= ns;

        const F2FScalar *xs0 = &Xs[3 * indx];
        F2FScalar rxs0[3];
        memcpy(rxs0, xs0, 3 * sizeof(F2FScalar));
        rxs0[isymm] *= -1.0;

        F2FScalar q[3];
        vec_diff(xs0bar, rxs0, q);

        F2FScalar *xs = &Xsd[3 * indx];
        F2FScalar rxs[3];
        memcpy(rxs, xs, 3 * sizeof(F2FScalar));
        rxs[isymm] *= -1.0;

        F2FScalar p[3];
        vec_diff(xsbar, rxs, p);

        F2FScalar lam[3];
        memcpy(lam, &vecs_global[3 * indx], 3 * sizeof(F2FScalar));
        lam[isymm] *= -1.0;
        F2FScalar lamp[3];
        vec_diff(lambar, lam, lamp);

        F2FScalar w = global_W[nn * i + j];
        F2FScalar *prod = &prods_global[3 * indx];

        F2FScalar rprod[3];

        rprod[0] = w * (q[0] * ZH[0] + q[1] * ZH[1] + q[2] * ZH[2]) +
                   w * (ZH[0] * p[0] + ZH[3] * p[1] + ZH[6] * p[2]);
        rprod[1] = w * (q[0] * ZH[3] + q[1] * ZH[4] + q[2] * ZH[5]) +
                   w * (ZH[1] * p[0] + ZH[4] * p[1] + ZH[7] * p[2]);
        rprod[2] = w * (q[0] * ZH[6] + q[1] * ZH[7] + q[2] * ZH[8]) +
                   w * (ZH[2] * p[0] + ZH[5] * p[1] + ZH[8] * p[2]);

        rprod[0] += w * (lamp[0] * XT[0] + lamp[1] * XT[1] + lamp[2] * XT[2]);
        rprod[1] += w * (lamp[0] * XT[3] + lamp[1] * XT[4] + lamp[2] * XT[5]);
        rprod[2] += w * (lamp[0] * XT[6] + lamp[1] * XT[7] + lamp[2] * XT[8]);

        rprod[0] -= w * qXlam[0];
        rprod[1] -= w * qXlam[1];
        rprod[2] -= w * qXlam[2];

        rprod[isymm] *= -1.0;

        prod[0] -= rprod[0];
        prod[1] -= rprod[1];
        prod[2] -= rprod[2];
      }
    }
  }
  // distribute the results to the structural processors
  distributeStructuralVector(prods_global, prods);

  // Free allocated memory
  delete[] Xsd;
  delete[] vecs_global;
  delete[] prods_global;
}
