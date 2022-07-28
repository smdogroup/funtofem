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

#include "LocatePoint.h"

#include <math.h>
#include <stdio.h>

#include "funtofemlapack.h"

/*
  Implementation of the locate point code

  Copyright (c) 2010 Graeme Kennedy. All rights reserved.
  Not for commercial purposes.
*/

/*
  Create an object that can rapidly locate the closest point
  within a cloud of points to the specified input point.
  This works in O(log(n)) time roughly, rather than O(n) time.
*/
LocatePoint::LocatePoint(const F2FScalar *_Xpts, int _npts,
                         int _max_num_points) {
  Xpts = _Xpts;
  npts = _npts;
  max_num_points = _max_num_points;

  // Calculate approximately how many nodes there should be
  max_nodes = (2 * npts) / max_num_points;
  if (max_nodes < 1) {
    max_nodes = 1;
  }
  num_nodes = 0;

  // The point indicies
  indices = new int[npts];
  for (int i = 0; i < npts; i++) {
    indices[i] = i;
  }

  // Set up the data structure that represents the
  // splitting planes
  nodes = new int[2 * max_nodes];
  indices_ptr = new int[max_nodes];
  num_indices = new int[max_nodes];

  // The base point and normal direction for the splitting
  // planes
  node_xav = new F2FScalar[3 * max_nodes];
  node_normal = new F2FScalar[3 * max_nodes];

  for (int i = 0; i < max_nodes; i++) {
    nodes[2 * i] = nodes[2 * i + 1] = -1;
    indices_ptr[i] = -1;
    num_indices[i] = -1;

    for (int k = 0; k < 3; k++) {
      node_xav[3 * i + k] = 0.0;
      node_normal[3 * i + k] = 0.0;
    }
  }

  // Recursively split the points
  split(0, npts);
}

/*
  Deallocate the memory for this object
*/
LocatePoint::~LocatePoint() {
  delete[] indices;
  delete[] nodes;
  delete[] indices_ptr;
  delete[] num_indices;
  delete[] node_xav;
  delete[] node_normal;
}

/*
  Locate the closest point using an exhaustive search
  of all points
*/
int LocatePoint::locateExhaustive(const F2FScalar xpt[]) {
  // Find the closest point via exhaustive search
  F2FReal dist = 1e40;
  int index = -1;

  for (int n = 0; n < npts; n++) {
    F2FReal t =
        F2FRealPart((Xpts[3 * n] - xpt[0]) * (Xpts[3 * n] - xpt[0]) +
                    (Xpts[3 * n + 1] - xpt[1]) * (Xpts[3 * n + 1] - xpt[1]) +
                    (Xpts[3 * n + 2] - xpt[2]) * (Xpts[3 * n + 2] - xpt[2]));
    if (t < dist) {
      dist = t;
      index = n;
    }
  }

  return index;
}

/*
  Locate the K closest points using an exhaustive search of all
  points in the domain
*/
void LocatePoint::locateKExhaustive(int K, int indx[], F2FScalar dist[],
                                    const F2FScalar xpt[]) {
  dist[0] = 1e40;
  indx[0] = -1;

  int nk = 1;

  for (int n = 0; n < npts; n++) {
    F2FScalar t = ((Xpts[3 * n] - xpt[0]) * (Xpts[3 * n] - xpt[0]) +
                   (Xpts[3 * n + 1] - xpt[1]) * (Xpts[3 * n + 1] - xpt[1]) +
                   (Xpts[3 * n + 2] - xpt[2]) * (Xpts[3 * n + 2] - xpt[2]));

    if (F2FRealPart(t) < F2FRealPart(dist[nk - 1])) {
      insertIndex(dist, indx, &nk, t, n, K);
    }
  }
}

/*
  Locate the K closest points in the taxi-cab norm to the specified
  plane.

  Here the distance is measured as follows:

  d = xpt - Xpts[i]
  b = d - n*n^{T}*d

  dist[i] = n^{T}*d + sqrt(b^{T}*b)

  Note that this is the un-squared distance (since the squared
  distance would be equal to the Cartesian distance)
*/
void LocatePoint::locateClosestTaxi(int K, int indx[], F2FScalar dist[],
                                    const F2FScalar xpt[],
                                    const F2FScalar n[]) {
  dist[0] = 1e40;
  indx[0] = -1;

  // Keep track of the number of points found
  int nk = 1;

  for (int i = 0; i < npts; i++) {
    F2FScalar d[3], b[3];
    d[0] = xpt[0] - Xpts[3 * i];
    d[1] = xpt[1] - Xpts[3 * i + 1];
    d[2] = xpt[2] - Xpts[3 * i + 2];

    // Compute the normal distance to the specified plane
    F2FScalar a = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];

    // Compute the distance in the plane
    b[0] = d[0] - n[0] * a;
    b[1] = d[1] - n[1] * a;
    b[2] = d[2] - n[2] * a;

    // Compute the sum of the normal and tangent-plane distances
    F2FScalar t = F2Ffabs(a) + sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);

    // if the distance is shorter than the longest stored distance,
    // insert it into the list
    if (F2FRealPart(t) < F2FRealPart(dist[nk - 1])) {
      insertIndex(dist, indx, &nk, t, i, K);
    }
  }
}

/*
  Locate the closest point using the recursive plane-splitting method
*/
int LocatePoint::locateClosest(const F2FScalar xpt[]) {
  int root = 0;
  int index = -1;
  F2FScalar dist = 1e40;

  // Find the closest value
  locateClosest(root, xpt, &dist, &index);

  return index;
}

/*
  Locate the point closest to the given point

  Note that 'dist' is the distance squared to the point in this function!
*/
void LocatePoint::locateClosest(int root, const F2FScalar xpt[],
                                F2FScalar *dist, int *index) {
  int start = indices_ptr[root];
  int left_node = nodes[2 * root];
  int right_node = nodes[2 * root + 1];

  if (start != -1) {  // This node is a leaf
    // Do an exhaustive search of the points at the node
    int end = start + num_indices[root];
    for (int k = start; k < end; k++) {
      int n = indices[k];

      F2FScalar t = ((Xpts[3 * n] - xpt[0]) * (Xpts[3 * n] - xpt[0]) +
                     (Xpts[3 * n + 1] - xpt[1]) * (Xpts[3 * n + 1] - xpt[1]) +
                     (Xpts[3 * n + 2] - xpt[2]) * (Xpts[3 * n + 2] - xpt[2]));

      if (F2FRealPart(t) < F2FRealPart(*dist)) {
        *dist = t;
        *index = n;
      }
    }
  } else {
    F2FScalar *xav = &node_xav[3 * root];
    F2FScalar *normal = &node_normal[3 * root];

    // The normal distance
    F2FScalar ndist =
        ((xpt[0] - xav[0]) * normal[0] + (xpt[1] - xav[1]) * normal[1] +
         (xpt[2] - xav[2]) * normal[2]);

    if (F2FRealPart(ndist) <
        0.0) {  // The point lies to the 'left' of the plane
      locateClosest(left_node, xpt, dist, index);

      // If the minimum distance to the plane is less than the minimum
      // distance then search the other branch too - there could be a
      // point on that branch that lies closer than *dist
      if (F2FRealPart(ndist * ndist) < F2FRealPart(*dist)) {
        locateClosest(right_node, xpt, dist, index);
      }
    } else {  // The point lies to the 'right' of the plane
      locateClosest(right_node, xpt, dist, index);

      // If the minimum distance to the plane is less than the minimum
      // distance then search the other branch too - there could be a
      // point on that branch that lies closer than *dist
      if (F2FRealPart(ndist * ndist) < F2FRealPart(*dist)) {
        locateClosest(left_node, xpt, dist, index);
      }
    }
  }
}

/*
  Locate the K closest points in the domain to the point using
  the plane-splitting method.
*/
void LocatePoint::locateKClosest(int K, int indx[], F2FScalar dist[],
                                 const F2FScalar xpt[]) {
  int nk = 0;  // Length of the array
  int root = 0;

  locateKClosest(K, root, xpt, dist, indx, &nk);

  // Check that the array of indices is in fact sorted
  if (nk < K) {
    printf("Error nk = %d < K = %d \n", nk, K);
  }

  // Check if the list is properly sorted
  int flag = 0;
  for (int k = 0; k < nk - 1; k++) {
    if (!(F2FRealPart(dist[k]) <= F2FRealPart(dist[k + 1]))) {
      flag = 1;
      break;
    }
  }
  if (flag) {
    printf("Error: list not sorted \n");
    for (int k = 0; k < nk; k++) {
      printf("dist[%d] = %g \n", k, F2FRealPart(dist[k]));
    }
  }
}

/*!
  Insert a point into a sorted list based upon the distance from the
  given point
*/
void LocatePoint::insertIndex(F2FScalar *dist, int *indx, int *nk, F2FScalar d,
                              int dindex, int K) {
  if (*nk == 0) {
    dist[*nk] = d;
    indx[*nk] = dindex;
    *nk += 1;
    return;
  } else if (*nk < K && F2FRealPart(dist[*nk - 1]) <= F2FRealPart(d)) {
    dist[*nk] = d;
    indx[*nk] = dindex;
    *nk += 1;
    return;
  }

  // Place it into the list
  int i = 0;
  while (i < *nk && (F2FRealPart(d) >= F2FRealPart(dist[i]))) {
    i++;
  }

  for (; i < *nk; i++) {
    int tindex = indx[i];
    F2FScalar t = dist[i];
    indx[i] = dindex;
    dist[i] = d;
    dindex = tindex;
    d = t;
  }

  if (*nk < K) {
    indx[*nk] = dindex;
    dist[*nk] = d;
    *nk += 1;
  }
}

/*!
  Locate the K-closest points to a given point!

  dist  == A sorted list of the K-closest distances
  indx  == The indices of the K-closest values
  nk    == The actual number of points in the list nk <= K
*/
void LocatePoint::locateKClosest(int K, int root, const F2FScalar xpt[],
                                 F2FScalar *dist, int *indx, int *nk) {
  int start = indices_ptr[root];
  int left_node = nodes[2 * root];
  int right_node = nodes[2 * root + 1];

  if (start != -1) {  // This node is a leaf
    // Do an exhaustive search of the points at the node

    int end = start + num_indices[root];
    for (int k = start; k < end; k++) {
      int n = indices[k];

      F2FScalar t = ((Xpts[3 * n] - xpt[0]) * (Xpts[3 * n] - xpt[0]) +
                     (Xpts[3 * n + 1] - xpt[1]) * (Xpts[3 * n + 1] - xpt[1]) +
                     (Xpts[3 * n + 2] - xpt[2]) * (Xpts[3 * n + 2] - xpt[2]));

      if ((*nk < K) || (F2FRealPart(t) < F2FRealPart(dist[K - 1]))) {
        insertIndex(dist, indx, nk, t, n, K);
      }
    }
  } else {
    F2FScalar *xav = &node_xav[3 * root];
    F2FScalar *normal = &node_normal[3 * root];

    // The normal distance
    F2FScalar ndist =
        ((xpt[0] - xav[0]) * normal[0] + (xpt[1] - xav[1]) * normal[1] +
         (xpt[2] - xav[2]) * normal[2]);

    if (F2FRealPart(ndist) <
        0.0) {  // The point lies to the 'left' of the plane
      locateKClosest(K, left_node, xpt, dist, indx, nk);

      // If the minimum distance to the plane is less than the minimum
      // distance then search the other branch too - there could be a
      // point on that branch that lies closer than *dist
      if (*nk < K || F2FRealPart(ndist * ndist) < F2FRealPart(dist[*nk - 1])) {
        locateKClosest(K, right_node, xpt, dist, indx, nk);
      }
    } else {  // The point lies to the 'right' of the plane
      locateKClosest(K, right_node, xpt, dist, indx, nk);

      // If the minimum distance to the plane is less than the minimum
      // distance then search the other branch too - there could be a
      // point on that branch that lies closer than *dist
      if (*nk < K || F2FRealPart(ndist * ndist) < F2FRealPart(dist[*nk - 1])) {
        locateKClosest(K, left_node, xpt, dist, indx, nk);
      }
    }
  }
}

/*!
  Split the list of indices into approximately two.
  Those on one half of a plane and those on the other.
*/
int LocatePoint::split(int start, int end) {
  int root = num_nodes;

  num_nodes++;
  if (num_nodes >= max_nodes) {
    extendArrays(num_nodes, 2 * (num_nodes + 1));
    max_nodes = 2 * (num_nodes + 1);
  }

  if (end - start <= max_num_points) {
    nodes[2 * root] = -1;
    nodes[2 * root + 1] = -1;

    for (int k = 0; k < 3; k++) {
      node_xav[3 * root + k] = 0.0;
      node_normal[3 * root + k] = 0.0;
    }

    indices_ptr[root] = start;
    num_indices[root] = end - start;

    return root;
  }

  indices_ptr[root] = -1;
  num_indices[root] = 0;

  int mid = splitList(&node_xav[3 * root], &node_normal[3 * root],
                      &indices[start], end - start);

  if (mid == 0 || mid == end - start) {
    fprintf(stderr,
            "LocatePoint: Error, splitting points did nothing \
-- problem with your nodes?\n");
    return root;
  }

  // Now, split the right and left hand sides of the list
  int left_node = split(start, start + mid);
  int right_node = split(start + mid, end);

  nodes[2 * root] = left_node;
  nodes[2 * root + 1] = right_node;

  return root;
}

/*!
  Split the array of indices into two sets: those indices that correspond
  to points on either side of a plane in three-space.
*/
int LocatePoint::splitList(F2FScalar xav[], F2FScalar normal[], int *ind,
                           int np) {
  xav[0] = xav[1] = xav[2] = F2FScalar(0.0);
  normal[0] = normal[1] = normal[2] = F2FScalar(0.0);

  // lwork  = 1 + 6*N + 2*N**2
  // liwork = 3 + 5*N
  F2FReal eigs[3];
  int N = 3;
  int lwork = 1 + 6 * N + 2 * N * N;
  F2FReal work[1 + 6 * 3 + 2 * 3 * 3];
  int liwork = 3 + 5 * N;
  int iwork[3 + 5 * 3];

  F2FReal I[9];
  for (int i = 0; i < 9; i++) {
    I[i] = 0.0;
  }

  // Find the average point and the moment of inertia about the average point
  for (int i = 0; i < np; i++) {
    int n = ind[i];
    for (int k = 0; k < 3; k++) {
      xav[k] += F2FRealPart(Xpts[3 * n + k]);
    }

    // I[0] = Ix = y^2 + z^2
    I[0] += F2FRealPart(Xpts[3 * n + 1] * Xpts[3 * n + 1] +
                        Xpts[3 * n + 2] * Xpts[3 * n + 2]);
    // I[4] = Iy = x^2 + z^2
    I[4] += F2FRealPart(Xpts[3 * n] * Xpts[3 * n] +
                        Xpts[3 * n + 2] * Xpts[3 * n + 2]);
    // I[8] = Iz = x^2 + y^2
    I[8] += F2FRealPart(Xpts[3 * n] * Xpts[3 * n] +
                        Xpts[3 * n + 1] * Xpts[3 * n + 1]);

    I[1] += -F2FRealPart(Xpts[3 * n] * Xpts[3 * n + 1]);      // Ixy = - xy
    I[2] += -F2FRealPart(Xpts[3 * n] * Xpts[3 * n + 2]);      // Ixz = - xz
    I[5] += -F2FRealPart(Xpts[3 * n + 1] * Xpts[3 * n + 2]);  // Ixz = - yz
  }

  for (int k = 0; k < 3; k++) {
    xav[k] = xav[k] / (1.0 * np);
  }

  // Ix(cm) = Ix - np*(yav^2 + zav^2) ... etc
  I[0] = I[0] - np * F2FRealPart(xav[1] * xav[1] + xav[2] * xav[2]);
  I[4] = I[4] - np * F2FRealPart(xav[0] * xav[0] + xav[2] * xav[2]);
  I[8] = I[8] - np * F2FRealPart(xav[0] * xav[0] + xav[1] * xav[1]);

  I[1] = I[1] + np * F2FRealPart(xav[0] * xav[1]);
  I[2] = I[2] + np * F2FRealPart(xav[0] * xav[2]);
  I[5] = I[5] + np * F2FRealPart(xav[1] * xav[2]);

  I[3] = I[1];
  I[6] = I[2];
  I[7] = I[5];

  // Find the eigenvalues/eigenvectors
  int info;
  const char *jobz = "V";
  const char *uplo = "U";

  LAPACKsyevd(jobz, uplo, &N, I, &N, eigs, work, &lwork, iwork, &liwork, &info);

  normal[0] = I[0];
  normal[1] = I[1];
  normal[2] = I[2];

  int low = 0;
  int high = np - 1;

  // Now, split the index array such that
  while (high > low) {
    // (dot(Xpts[ind] - xav, n ) < 0 ) < 0.0 for i < low
    while (high > low &&
           F2FRealPart((Xpts[3 * ind[low]] - xav[0]) * normal[0] +
                       (Xpts[3 * ind[low] + 1] - xav[1]) * normal[1] +
                       (Xpts[3 * ind[low] + 2] - xav[2]) * normal[2]) < 0.0) {
      low++;
    }

    // (dot(Xpts[ind] - xav, n ) < 0 ) >= 0.0 for i >= high
    while (high > low &&
           F2FRealPart((Xpts[3 * ind[high]] - xav[0]) * normal[0] +
                       (Xpts[3 * ind[high] + 1] - xav[1]) * normal[1] +
                       (Xpts[3 * ind[high] + 2] - xav[2]) * normal[2]) >= 0.0) {
      high--;
    }

    if (high > low) {
      // Switch the two indices that don't match
      int temp = ind[high];
      ind[high] = ind[low];
      ind[low] = temp;
    }
  }

  if (low == 0 || low == np) {
    fprintf(stderr, "LocatePoint: Error split points\n");
  }

  return low;
}

/*!
  If not enough memory has been allocated, extend the arrays required
  to store all the nodes.
*/
void LocatePoint::extendArrays(int old_len, int new_len) {
  nodes = newIntArray(nodes, 2 * old_len, 2 * new_len);
  indices_ptr = newIntArray(indices_ptr, old_len, new_len);
  num_indices = newIntArray(num_indices, old_len, new_len);

  node_xav = newDoubleArray(node_xav, 3 * old_len, 3 * new_len);
  node_normal = newDoubleArray(node_normal, 3 * old_len, 3 * new_len);
}

/*
  Allocate more space for an integer array and copy the old
  array to the newly created array
*/
int *LocatePoint::newIntArray(int *array, int old_len, int new_len) {
  int *temp = new int[new_len];

  for (int i = 0; i < old_len; i++) {
    temp[i] = array[i];
  }
  for (int i = old_len; i < new_len; i++) {
    temp[i] = -1;
  }

  delete[] array;

  return temp;
}

/*
  Allocate space for a new double array and copy the old
  array to the newly created array
*/
F2FScalar *LocatePoint::newDoubleArray(F2FScalar *array, int old_len,
                                       int new_len) {
  F2FScalar *temp = new F2FScalar[new_len];

  for (int i = 0; i < old_len; i++) {
    temp[i] = array[i];
  }
  for (int i = old_len; i < new_len; i++) {
    temp[i] = 0.0;
  }
  delete[] array;

  return temp;
}
