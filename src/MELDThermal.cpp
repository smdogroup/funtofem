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

#include "MELDThermal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>

#include "LocatePoint.h"
#include "funtofemlapack.h"

MELDThermal::MELDThermal(MPI_Comm global_comm, MPI_Comm struct_comm,
                         int struct_root, MPI_Comm aero_comm, int aero_root,
                         int isymm, int num_nearest, F2FScalar beta)
    : ThermalTransfer(global_comm, struct_comm, struct_root, aero_comm,
                      aero_root),
      isymm(isymm),
      nn(num_nearest),
      global_beta(beta) {
  global_conn = NULL;
  global_W = NULL;

  // Space to be allocated for the structural temperatures and aero
  // normal component of the heat flux
  Ts = NULL;
  Ha = NULL;

  // Notify user of the type of transfer scheme they are using
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: Creating scheme of type MELDThermal...\n",
           object_id);
  }
}

MELDThermal::~MELDThermal() {
  // Free the aerostructural connectivity data
  if (global_conn) {
    delete[] global_conn;
  }

  // Free the load transfer data
  if (global_W) {
    delete[] global_W;
  }

  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: freeing MELDThermal data...\n", object_id);
  }
}

/*
  Set aerostructural connectivity, compute weights, and allocate memory needed
  for transfers and products
*/
void MELDThermal::initialize() {
  // global number of structural nodes
  distributeStructuralMesh();

  if (Ts) {
    delete[] Ts;
  }
  Ts = new F2FScalar[ns];

  if (Ha) {
    delete[] Ha;
  }
  Ha = new F2FScalar[na];

  // Check that user doesn't set more nearest nodes than exist in total
  if (nn > ns) {
    nn = ns;
  }

  // Create aerostructural connectivity
  global_conn = new int[nn * na];
  computeAeroStructConn(isymm, nn, global_conn);

  // Allocate and compute the weights
  global_W = new F2FScalar[nn * na];
  computeWeights(F2FRealPart(global_beta), isymm, nn, global_conn, global_W);
}

/*
  Computes the displacements of aerodynamic surface nodes by fitting an
  optimal rigid rotation and translation to the displacement of the set of
  structural nodes nearest each aerodynamic surface node

  Arguments
  ---------
  struct_temps : structural node temperatures

  Returns
  -------
  aero_temps   : aerodynamic node temperatures
*/
void MELDThermal::transferTemp(const F2FScalar *struct_temps,
                               F2FScalar *aero_temps) {
  // Distribute the mesh components if needed
  distributeStructuralMesh();

  // Copy the temperature into the global temperature vector
  structGatherBcast(ns_local, struct_temps, ns, Ts);

  // Zero the outputs
  memset(aero_temps, 0.0, na * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    const int *local_conn = &global_conn[i * nn];
    const F2FScalar *w = &global_W[i * nn];

    F2FScalar Taero = 0.0;
    for (int j = 0; j < nn; j++) {
      if (local_conn[j] < ns) {
        Taero += w[j] * Ts[local_conn[j]];
      } else {
        Taero += w[j] * Ts[local_conn[j] - ns];
      }
    }

    aero_temps[i] = Taero;
  }
}

/*
  Computes the loads on all structural nodes consistently and conservatively
  from loads on aerodynamic surface nodes

  Arguments
  ---------
  aero_flux   : normal flux through surface on aerodynamic surface nodes

  Returns
  -------
  struct_loads : loads on structural nodes
*/
void MELDThermal::transferFlux(const F2FScalar *aero_flux,
                               F2FScalar *struct_flux) {
  // Copy prescribed aero loads into member variable
  memcpy(Ha, aero_flux, na * sizeof(F2FScalar));

  // Zero struct flux
  F2FScalar *struct_flux_global = new F2FScalar[ns];
  memset(struct_flux_global, 0, ns * sizeof(F2FScalar));

  for (int i = 0; i < na; i++) {
    const int *local_conn = &global_conn[i * nn];
    const F2FScalar *w = &global_W[i * nn];
    const F2FScalar *ha = &Ha[i];

    for (int j = 0; j < nn; j++) {
      int index = 0;
      if (local_conn[j] < ns) {
        index = local_conn[j];
      } else {
        index = local_conn[j] - ns;
      }
      struct_flux_global[index] += w[j] * ha[0];
    }
  }

  structAddScatter(ns, struct_flux_global, ns_local, struct_flux);

  delete[] struct_flux_global;
}

/*
  Apply the action of the temperature transfer w.r.t structural temperature
  Jacobian to the input vector

  Arguments
  ---------
  vecs  : input structural vector

  Returns
  --------
  prods : output aerodynamic vector
*/
void MELDThermal::applydTdtS(const F2FScalar *vecs, F2FScalar *prods) {
  // Make a global image of the input vector
  F2FScalar *vecs_global = new F2FScalar[ns];
  structGatherBcast(ns_local, vecs, ns, vecs_global);

  // Zero array of Jacobian-vector products every call
  memset(prods, 0, na * sizeof(F2FScalar));

  // Loop over all aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Loop over linked structural nodes and add up nonzero contributions to
    // Jacobian-vector product
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];

      F2FScalar v;
      if (indx < ns) {
        v = vecs_global[indx];
      } else {
        indx -= ns;
        v = vecs_global[indx];
      }

      // Compute each component of the Jacobian vector product as follows:
      // Jv[k] = w*v[k]
      F2FScalar w = global_W[nn * i + j];
      prods[i] -= w * v;
    }
  }

  // Clean up the allocated memory
  delete[] vecs_global;
}

/*
  Apply the action of the temperature transfer w.r.t structural temperature
  transpose Jacobian to the input vector

  Arguments
  ----------
  vecs  : input aerodynamic vector

  Returns
  --------
  prods : output structural vector
*/
void MELDThermal::applydTdtSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  // Zero array of transpose Jacobian-vector products every call
  F2FScalar *prods_global = new F2FScalar[ns];
  memset(prods_global, 0, ns * sizeof(F2FScalar));

  // Loop over aerodynamic surface nodes
  for (int i = 0; i < na; i++) {
    // Loop over linked structural nodes and add up nonzero contributions to
    // Jacobian-vector product
    for (int j = 0; j < nn; j++) {
      int indx = global_conn[nn * i + j];
      F2FScalar w = global_W[nn * i + j];

      if (indx < ns) {
        prods_global[indx] -= w * vecs[i];
      } else {
        indx -= ns;
        prods_global[indx] -= w * vecs[i];
      }
    }
  }

  // distribute the results to the structural processors
  structAddScatter(ns, prods_global, ns_local, prods);

  // clean up allocated memory
  delete[] prods_global;
}

/*
  Apply the action of the flux transfer w.r.t structural temperature
  Jacobian to the input vector

  Arguments
  ---------
  vecs  : input aerodynamic vector

  Returns
  --------
  prods : output structural vector
*/
void MELDThermal::applydQdqA(const F2FScalar *vecs, F2FScalar *prods) {
  applydTdtSTrans(vecs, prods);
}

/*
  Apply the action of the flux transfer w.r.t structural temperature
  transpose Jacobian to the input vector

  Arguments
  ----------
  vecs  : input structural vector

  Returns
  --------
  prods : output aerodynamic vector
*/
void MELDThermal::applydQdqATrans(const F2FScalar *vecs, F2FScalar *prods) {
  applydTdtS(vecs, prods);
}
