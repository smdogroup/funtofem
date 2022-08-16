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

#include "RBF.h"

#include <math.h>
#include <stdio.h>

#include <cstdlib>
#include <cstring>

#include "Octree.h"
#include "funtofemlapack.h"

RBF::RBF(MPI_Comm all, MPI_Comm structure, int _struct_root, MPI_Comm aero,
         int _aero_root, enum RbfType rbf_type, int sampling_ratio) {
  // TODO: figure out parallelism for RBFs
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

  // Point to the selected type of RBF
  switch (rbf_type) {
    case GAUSSIAN:
      phi = &gaussian;
      break;
    case MULTIQUADRIC:
      phi = &multiquadric;
      break;
    case INVERSE_MULTIQUADRIC:
      phi = &invMultiquadric;
      break;
    case THIN_PLATE_SPLINE:
      phi = &thinPlateSpline;
      break;
  }

  // Initialize sampling data
  denominator = sampling_ratio;
  sample_ids = NULL;

  // Initialize object id
  object_id = TransferScheme::object_count++;

  // Notify user of the type of transfer scheme they are using
  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: Creating scheme of type RBF...\n", object_id);
  }
}

RBF::~RBF() {
  // Free the sample ids matrix
  if (sample_ids) {
    delete[] sample_ids;
  }

  int rank;
  MPI_Comm_rank(global_comm, &rank);
  if (rank == struct_root) {
    printf("Transfer scheme [%i]: freeing RBF data...\n", object_id);
  }
}

/*
  Sample the structural nodes and build the interpolation matrix
*/
void RBF::initialize() {
  // Sample the structural nodes
  if (denominator > 1) {
    printf("Transfer scheme [%i]: attempting to sample nodes using octree...\n",
           object_id);

    // Generate an octree
    int min_point_count = denominator;
    double min_edge_length = 1.0e-6;
    int max_tree_depth = 50;
    Octree *octree =
        new Octree(Xs, ns, min_point_count, min_edge_length, max_tree_depth);
    octree->generate();
    nsub = octree->nleaf;
    sample_ids = new int[nsub];

    // Randomly sample one point from each leaf bin of the octree
    for (int i = 0; i < nsub; i++) {
      int bin_id = octree->leaf_bins[i];
      int num_bin_pts = 0;
      int *bin_points_ids = new int[1];  // must allocate array to start

      // First find the ids of the points in each leaf bin
      for (int j = 0; j < ns; j++) {
        if (octree->points_bins[j] == bin_id) {
          if (num_bin_pts > 0) {
            int *new_bin_points_ids = new int[num_bin_pts + 1];
            memcpy(new_bin_points_ids, bin_points_ids,
                   num_bin_pts * sizeof(int));
            delete[] bin_points_ids;
            bin_points_ids = new_bin_points_ids;
          }
          bin_points_ids[num_bin_pts] = j;
          num_bin_pts++;
        }
      }

      // Then randomly pick one id
      double h = 1.0 / num_bin_pts;
      double random_num = (1.0 * rand()) / RAND_MAX;
      int k = 0;
      while (k < num_bin_pts) {
        if (random_num < (k + 1) * h) break;
        k++;
      }

      // Add id to array of sampled ids
      sample_ids[i] = bin_points_ids[k];

      // Free allocated memory
      delete[] bin_points_ids;
    }

    // Delete octree
    delete octree;

    // Report actual sampling
    double percent_sampled = (100.0 * nsub) / ns;
    printf("Transfer scheme [%i]: sampled %4.1f%% of nodes.\n", object_id,
           percent_sampled);

    // Write full and sampled point clouds to ASCII file for Tecplot
    writeCloudsToTecplot();

  } else {
    nsub = ns;
    sample_ids = new int[nsub];
    for (int i = 0; i < nsub; i++) {
      sample_ids[i] = i;
    }
  }

  // Allocate memory for interpolation matrix
  interp_mat = new F2FScalar[na * nsub];

  // Build the interpolation matrix
  buildInterpolationMatrix();
}

/*
  Auxiliary function for building the interpolation matrix
*/
void RBF::buildInterpolationMatrix() {
  // Check how many first order polynomial terms to include
  int npoly = 4;
  double xsum = 0.0;
  double ysum = 0.0;
  double zsum = 0.0;

  for (int i = 0; i < nsub; i++) {
    xsum += F2FRealPart(abs(Xs[3 * i + 0]));
    ysum += F2FRealPart(abs(Xs[3 * i + 1]));
    zsum += F2FRealPart(abs(Xs[3 * i + 2]));
  }

  bool x_all_zero = xsum < 1.0e-15;
  if (x_all_zero) npoly--;
  bool y_all_zero = ysum < 1.0e-15;
  if (y_all_zero) npoly--;
  bool z_all_zero = zsum < 1.0e-15;
  if (z_all_zero) npoly--;

  // Build the P matrix
  F2FScalar *P = new F2FScalar[npoly * nsub];
  for (int j = 0; j < nsub; j++) {
    int indx = sample_ids[j];
    P[0 + npoly * j] = 1.0;
    if (npoly > 1) {
      for (int k = 1; k < npoly; k++) {
        P[k + npoly * j] = Xs[3 * indx + k - 1];
      }
    }
  }

  // Build the M matrix
  F2FScalar *M = new F2FScalar[nsub * nsub];
  for (int i = 0; i < nsub; i++) {
    for (int j = 0; j < nsub; j++) {
      int indx1 = sample_ids[i];
      int indx2 = sample_ids[j];
      F2FScalar *x = &Xs[3 * indx1];
      F2FScalar *y = &Xs[3 * indx2];
      M[i + nsub * j] = phi(x, y);
    }
  }

  /*
    Need to build the C_{ss}^{-1} matrix, which is composed of a top half
    (corresponding to the polynomial coefficients) and a bottom half
    (corresponding to and the radial basis function coefficients)

    The top half is M_{p}*P*M^{-1}

    The bottom half is M^{-1} - M^{-1}*P^{T}*M_{p}*P*M^{-1}

    It is a bit difficult to understand the procedure for assembling these
    matrices from the calls to BLAS, so I have tried to add clarifying
    comments. If something remains unclear, it is best to refer back to the
    paper cited in the header file.
  */

  // Invert the M matrix
  F2FScalar *invM = new F2FScalar[nsub * nsub];
  memset(invM, 0.0, nsub * nsub * sizeof(F2FScalar));
  for (int i = 0; i < nsub; i++) invM[i + nsub * i] = 1.0;
  int *ipiv = new int[nsub];
  int info = 0;
  LAPACKgetrf(&nsub, &nsub, M, &nsub, ipiv, &info);
  LAPACKgetrs("N", &nsub, &nsub, M, &nsub, ipiv, invM, &nsub, &info);
  delete[] ipiv;

  // M_{p}^{-1} = P*M^{-1}*P^{T}
  F2FScalar *invMp = new F2FScalar[npoly * npoly];
  F2FScalar alpha = 1.0, beta = 0.0;
  F2FScalar *Psized = new F2FScalar[npoly * nsub];  // work matrix
  BLASgemm("N", "N", &npoly, &nsub, &nsub, &alpha, P, &npoly, invM, &nsub,
           &beta, Psized, &npoly);
#ifdef FUNTOFEM_USE_COMPLEX
  const char *t = "C";
#else
  const char *t = "T";
#endif
  BLASgemm("N", t, &npoly, &npoly, &nsub, &alpha, Psized, &npoly, P, &npoly,
           &beta, invMp, &npoly);

  // Invert M_{p}^{-1} to obtain M_{p}
  F2FScalar *Mp = new F2FScalar[npoly * npoly];
  memset(Mp, 0.0, npoly * npoly * sizeof(F2FScalar));
  for (int i = 0; i < npoly; i++) Mp[i + npoly * i] = 1.0;
  ipiv = new int[npoly];
  LAPACKgetrf(&npoly, &npoly, invMp, &npoly, ipiv, &info);
  LAPACKgetrs("N", &npoly, &npoly, invMp, &npoly, ipiv, Mp, &npoly, &info);
  delete[] ipiv;

  // Build top half
  F2FScalar *top_half = new F2FScalar[npoly * nsub];  // more work matrices
  BLASgemm("N", "N", &npoly, &nsub, &npoly, &alpha, Mp, &npoly, Psized, &npoly,
           &beta, top_half, &npoly);

  // Use the top half to build the bottom half
  BLASgemm(t, "N", &nsub, &nsub, &npoly, &alpha, P, &npoly, top_half, &npoly,
           &beta, M, &nsub);
  F2FScalar *bot_half = new F2FScalar[nsub * nsub];
  memcpy(bot_half, invM, nsub * nsub * sizeof(F2FScalar));
  alpha = -1.0;
  beta = 1.0;
  BLASgemm("N", "N", &nsub, &nsub, &nsub, &alpha, invM, &nsub, M, &nsub, &beta,
           bot_half, &nsub);

  // Copy the top and bottom halves into one matrix C_{ss}^{-1}
  F2FScalar *invCss = new F2FScalar[(nsub + npoly) * nsub];
  for (int i = 0; i < npoly; i++) {
    for (int j = 0; j < nsub; j++) {
      invCss[i + (nsub + npoly) * j] = top_half[i + npoly * j];
    }
  }
  for (int i = 0; i < nsub; i++) {
    for (int j = 0; j < nsub; j++) {
      invCss[npoly + i + (nsub + npoly) * j] = bot_half[i + nsub * j];
    }
  }

  // Build the A_{as} matrix, the entries of which are the evaluation of the
  // polynomial and radial basis functions at the aerodynamic nodes
  F2FScalar *Aas = new F2FScalar[na * (nsub + npoly)];
  for (int i = 0; i < na; i++) {
    Aas[i + na * 0] = 1.0;
    Aas[i + na * 1] = Xa[3 * i + 0];
    Aas[i + na * 2] = Xa[3 * i + 1];
    Aas[i + na * 3] = Xa[3 * i + 2];
    for (int j = 0; j < nsub; j++) {
      int indx = sample_ids[j];
      F2FScalar *x = &Xa[3 * i];
      F2FScalar *y = &Xs[3 * indx];
      Aas[i + na * (j + npoly)] = phi(x, y);
    }
  }

  // Multiply A_{as} and C_{ss}^{-1} to get the interpolation matrix
  int k = nsub + npoly;
  alpha = 1.0;
  beta = 0.0;
  BLASgemm("N", "N", &na, &nsub, &k, &alpha, Aas, &na, invCss, &k, &beta,
           interp_mat, &na);

  // Free allocated memory
  delete[] P;
  delete[] M;
  delete[] invM;
  delete[] invMp;
  delete[] Psized;
  delete[] Mp;
  delete[] top_half;
  delete[] bot_half;
  delete[] invCss;
  delete[] Aas;
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
void RBF::transferDisps(const F2FScalar *struct_disps, F2FScalar *aero_disps) {
  // Copy prescribed displacements into displacement vector
  memcpy(Us, struct_disps, 3 * ns * sizeof(F2FScalar));

  // Zero the outputs
  memset(aero_disps, 0.0, 3 * na * sizeof(F2FScalar));

  // Rearrange structural displacements
  F2FScalar *US = new F2FScalar[nsub * 3];
  for (int i = 0; i < nsub; i++) {
    int indx = sample_ids[i];
    US[i + nsub * 0] = Us[3 * indx + 0];
    US[i + nsub * 1] = Us[3 * indx + 1];
    US[i + nsub * 2] = Us[3 * indx + 2];
  }

  // Apply action of interpolation matrix
  F2FScalar *UA = new F2FScalar[na * 3];
  int n = 3;
  F2FScalar alpha = 1.0, beta = 0.0;
  BLASgemm("N", "N", &na, &n, &nsub, &alpha, interp_mat, &na, US, &nsub, &beta,
           UA, &na);

  // Copy aerodynamic displacements to output
  for (int i = 0; i < na; i++) {
    aero_disps[3 * i + 0] = UA[i + na * 0];
    aero_disps[3 * i + 1] = UA[i + na * 1];
    aero_disps[3 * i + 2] = UA[i + na * 2];
  }

  // Free allocated memory
  delete[] US;
  delete[] UA;
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
void RBF::transferLoads(const F2FScalar *aero_loads, F2FScalar *struct_loads) {
  // Copy prescribed aero loads into member variable
  memcpy(Fa, aero_loads, 3 * na * sizeof(F2FScalar));

  // Zero struct loads
  memset(struct_loads, 0, 3 * ns * sizeof(F2FScalar));

  // Copy Fa into matrix
  F2FScalar *Fxyz = new F2FScalar[na * 3];
  for (int i = 0; i < na; i++) {
    Fxyz[i + na * 0] = Fa[3 * i + 0];
    Fxyz[i + na * 1] = Fa[3 * i + 1];
    Fxyz[i + na * 2] = Fa[3 * i + 2];
  }

  // Apply action of transpose of the interpolation matrix
  F2FScalar *Fsub = new F2FScalar[nsub * 3];
  int n = 3;
  F2FScalar alpha = 1.0, beta = 0.0;
#ifdef FUNTOFEM_USE_COMPLEX
  const char *t = "C";
#else
  const char *t = "T";
#endif
  BLASgemm(t, "N", &nsub, &n, &na, &alpha, interp_mat, &na, Fxyz, &na, &beta,
           Fsub, &nsub);

  // Copy the structural forces to struct loads
  for (int i = 0; i < nsub; i++) {
    int indx = sample_ids[i];
    struct_loads[3 * indx + 0] = Fsub[i + nsub * 0];
    struct_loads[3 * indx + 1] = Fsub[i + nsub * 1];
    struct_loads[3 * indx + 2] = Fsub[i + nsub * 2];
  }

  // Free allocated memory
  delete[] Fxyz;
  delete[] Fsub;
}

/*
  Apply the action of the displacement transfer w.r.t structural displacments
  Jacobian to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void RBF::applydDduS(const F2FScalar *vecs, F2FScalar *prods) {
  transferDisps(vecs, prods);

  // Reverse sign due to definition of diplacement transfer residual
  for (int i = 0; i < 3 * na; i++) {
    prods[i] *= -1.0;
  }
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
void RBF::applydDduSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  transferLoads(vecs, prods);

  // Reverse sign due to definition of diplacement transfer residual
  for (int j = 0; j < 3 * ns; j++) {
    prods[j] *= -1.0;
  }
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
void RBF::applydLduS(const F2FScalar *vecs, F2FScalar *prods) {
  memset(prods, 0, 3 * ns * sizeof(F2FScalar));
}

/*
  Apply the action of the load transfer w.r.t structural displacements
  transpose Jacobian to the input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void RBF::applydLduSTrans(const F2FScalar *vecs, F2FScalar *prods) {
  memset(prods, 0, 3 * ns * sizeof(F2FScalar));
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
void RBF::applydDdxA0(const F2FScalar *vecs, F2FScalar *prods) {
  memset(prods, 0, 3 * na * sizeof(F2FScalar));
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
void RBF::applydDdxS0(const F2FScalar *vecs, F2FScalar *prods) {
  memset(prods, 0, 3 * ns * sizeof(F2FScalar));
}

/*
  Apply the action of the load transfer w.r.t initial aerodynamic surface node
  locations Jacobian to the right of the transposed input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void RBF::applydLdxA0(const F2FScalar *vecs, F2FScalar *prods) {
  memset(prods, 0, 3 * na * sizeof(F2FScalar));
}

/*
  Apply the action of the load transfer w.r.t initial structural node locations
  Jacobian to the right of the transposed input vector

  Arguments
  ----------
  vecs  : input vector

  Returns
  --------
  prods : output vector
*/
void RBF::applydLdxS0(const F2FScalar *vecs, F2FScalar *prods) {
  memset(prods, 0, 3 * ns * sizeof(F2FScalar));
}

/*
  Defines the Gaussian radial basis function

  phi(x, y) = exp(-0.5*(r/sigma)**2)/(sigma*sqrt(2*pi)),
  where r = ||x - y||_{2}

  Arguments
  ---------
  x        : target point
  y        : source point

  Returns
  -------
  phi : evaluation of radial basis function
*/
F2FScalar RBF::gaussian(F2FScalar *x, F2FScalar *y) {
  F2FScalar sigma = 0.5;
  F2FScalar r =
      sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) +
           (x[2] - y[2]) * (x[2] - y[2]));
  F2FScalar eval =
      exp(-0.5 * (r / sigma) * (r / sigma)) / (sigma * sqrt(2.0 * M_PI));
  return eval;
}

/*
  Defines the multiquadric radial basis function

  phi(x, y) = sqrt(c**2 + r**2),
  where r = ||x - y||_{2}

  Arguments
  ---------
  x        : target point
  y        : source point

  Returns
  -------
  phi : evaluation of radial basis function
*/
F2FScalar RBF::multiquadric(F2FScalar *x, F2FScalar *y) {
  F2FScalar r =
      sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) +
           (x[2] - y[2]) * (x[2] - y[2]));
  F2FScalar eval = sqrt(1.0 + r * r);
  return eval;
}

/*
  Defines the inverse multiquadric radial basis function

  phi(x, y) = 1/sqrt(c**2 + r**2),
  where r = ||x - y||_{2}

  Arguments
  ---------
  x        : target point
  y        : source point

  Returns
  -------
  phi : evaluation of radial basis function
*/
F2FScalar RBF::invMultiquadric(F2FScalar *x, F2FScalar *y) {
  F2FScalar r =
      sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) +
           (x[2] - y[2]) * (x[2] - y[2]));
  F2FScalar eval = 1.0 / sqrt(1.0 + r * r);
  return eval;
}

/*
  Defines the thin plate spline radial basis function

  phi(x, y) = r^{2}*log(r),
  where r = ||x - y||_{2}

  Arguments
  ---------
  x        : target point
  y        : source point

  Returns
  -------
  phi : evaluation of radial basis function
*/
F2FScalar RBF::thinPlateSpline(F2FScalar *x, F2FScalar *y) {
  F2FScalar r =
      sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) +
           (x[2] - y[2]) * (x[2] - y[2]));
  F2FScalar eps = 1.0e-7;
  F2FScalar eval = r * r * log(r + eps);
  return eval;
}

/*
  Write full and sampled structural point clouds to ASCII file that can be read
  into Tecplot
*/
void RBF::writeCloudsToTecplot() {
  FILE *file = fopen("point_clouds.dat", "w");
  // Write Tecplot compatible header
  fprintf(file, "TITLE=\"Structural point clouds\"\n");
  fprintf(file, "VARIABLES=\"X\", \"Y\", \"Z\"\n");

  // Write full point cloud data
  fprintf(file, "ZONE T=\"Full\"\n");
  fprintf(file, "I=%i \n", ns);
  for (int i = 0; i < ns; i++) {
    fprintf(file, "%22.15e %22.15e %22.15e\n", F2FRealPart(Xs[3 * i + 0]),
            F2FRealPart(Xs[3 * i + 1]), F2FRealPart(Xs[3 * i + 2]));
  }

  // Write sampled point cloud data
  fprintf(file, "ZONE T=\"Sampled\"\n");
  fprintf(file, "I=%i \n", nsub);
  for (int i = 0; i < nsub; i++) {
    int indx = sample_ids[i];
    fprintf(file, "%22.15e %22.15e %22.15e\n", F2FRealPart(Xs[3 * indx + 0]),
            F2FRealPart(Xs[3 * indx + 1]), F2FRealPart(Xs[3 * indx + 2]));
  }

  fclose(file);
}
