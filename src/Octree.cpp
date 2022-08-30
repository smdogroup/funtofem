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

#include "Octree.h"

#include <stdio.h>

#include <cstring>

/*
  Constructor for octree object

  Arguments
  ---------
  points          : points coordinates x1, y1, z1, ..., xn, yn, zn
  num_points      : number of points
  min_point_count : minimum number of points in smallest bin
  min_edge_length : minimum edge length of bin
  max_tree_depth  : deepest allowed level of tree
*/
Octree::Octree(F2FScalar *points, int num_points, int min_point_count,
               double min_edge_length, int max_tree_depth) {
  // Allocate memory for points
  npts = num_points;
  Xpts = new double[3 * npts];

  // If in complex mode, strip off imaginary parts and only copy in real parts
#ifdef FUNTOFEM_USE_COMPLEX
  for (int i = 0; i < 3 * npts; i++) {
    Xpts[i] = F2FRealPart(points[i]);
  }
#else
  memcpy(Xpts, points, 3 * npts * sizeof(double));
#endif

  // Initialize recursion exit conditions
  min_points = min_point_count;
  min_edge = min_edge_length;
  max_depth = max_tree_depth;

  printf("Octree: creating octree with %i points...\n", npts);
}

/*
  Destructor
*/
Octree::~Octree() {
  // Delete tree data
  if (Xpts) delete[] Xpts;
  if (bin_depths) delete[] bin_depths;
  if (bin_parents) delete[] bin_parents;
  if (bin_corners) delete[] bin_corners;
  if (points_bins) delete[] points_bins;
  if (leaf_bins) delete[] leaf_bins;

  printf("Octree: freeing octree data...\n\n");
}

/*
  Allocate memory for tree data and run tree generator function
*/
void Octree::generate() {
  // Create base-level bin
  nbins = 1;
  bin_depths = new int[nbins];
  bin_depths[0] = 0;
  bin_parents = new int[nbins];
  bin_parents[0] = 0;
  bin_corners = new double[6 * nbins];
  points_bins = new int[npts];
  memset(points_bins, 0, npts * sizeof(int));  // all points start in base bin
  nleaf = 0;
  leaf_bins = new int[nbins];  // cannot allocate an array of length zero

  // Find corners of base bin
  double xmin[] = {Xpts[0], Xpts[1], Xpts[2]};
  double xmax[] = {Xpts[0], Xpts[1], Xpts[2]};
  for (int i = 1; i < npts; i++) {
    if (Xpts[3 * i + 0] < xmin[0]) xmin[0] = Xpts[3 * i + 0];  // min x
    if (Xpts[3 * i + 1] < xmin[1]) xmin[1] = Xpts[3 * i + 1];  // min y
    if (Xpts[3 * i + 2] < xmin[2]) xmin[2] = Xpts[3 * i + 2];  // min z

    if (Xpts[3 * i + 0] > xmax[0]) xmax[0] = Xpts[3 * i + 0];  // max x
    if (Xpts[3 * i + 1] > xmax[1]) xmax[1] = Xpts[3 * i + 1];  // max y
    if (Xpts[3 * i + 2] > xmax[2]) xmax[2] = Xpts[3 * i + 2];  // max z
  }
  memcpy(bin_corners, xmin, 3 * sizeof(double));
  memcpy(&bin_corners[3], xmax, 3 * sizeof(double));

  // Recursively divide the base bin to create the tree
  bool is_base_leaf_bin = divide(0);
  if (is_base_leaf_bin) {
    leaf_bins[nleaf] = 0;  // base-level bin is only leaf
    printf("Octree error: the base-level bin could not be subdivided.\n");
  }
}

/*
  Recursive function for generating tree

  Arguments
  ---------
  bin_id : ID of bin

  Returns
  -------
  divide : boolean indicating whether the bin satisfies the exit conditions
*/
bool Octree::divide(int bin_id) {
  // Count points in bin
  int bin_count = 0;
  for (int i = 0; i < npts; i++) {
    if (points_bins[i] == bin_id) {
      bin_count++;
    }
  }
  bool count_check = bin_count <= min_points;

  // Find smallest edge of bin
  double edge_x = bin_corners[6 * bin_id + 3] - bin_corners[6 * bin_id + 0];
  double edge_y = bin_corners[6 * bin_id + 4] - bin_corners[6 * bin_id + 1];
  double edge_z = bin_corners[6 * bin_id + 5] - bin_corners[6 * bin_id + 2];
  double edge_s;
  edge_s = (edge_y < edge_x) ? edge_y : edge_x;
  edge_s = (edge_z < edge_s) ? edge_z : edge_s;
  bool edge_check = edge_s < min_edge;

  // Find depth of bin
  int bin_depth = bin_depths[bin_id];
  bool depth_check = bin_depth >= max_depth;

  // Check if exit conditions are satisfied
  if (count_check || edge_check || depth_check) {
    if (bin_count > 0) {
      return true;
    } else {
      return false;
    }
  }

  // Find center of bin
  double xcen[] = {
      0.5 * (bin_corners[6 * bin_id + 3] + bin_corners[6 * bin_id + 0]),
      0.5 * (bin_corners[6 * bin_id + 4] + bin_corners[6 * bin_id + 1]),
      0.5 * (bin_corners[6 * bin_id + 5] + bin_corners[6 * bin_id + 2])};

  // Add 8 new bins
  for (int i = 0; i < 8; i++) {
    // Reallocate memory for the old data plus that of one new bin
    int *new_bin_depths = new int[nbins + 1];
    memcpy(new_bin_depths, bin_depths, nbins * sizeof(int));
    delete[] bin_depths;
    bin_depths = new_bin_depths;

    int *new_bin_parents = new int[nbins + 1];
    memcpy(new_bin_parents, bin_parents, nbins * sizeof(int));
    delete[] bin_parents;
    bin_parents = new_bin_parents;

    double *new_bin_corners = new double[6 * (nbins + 1)];
    memcpy(new_bin_corners, bin_corners, 6 * nbins * sizeof(double));
    delete[] bin_corners;
    bin_corners = new_bin_corners;

    // Update depths and parents arrays
    int new_bin_id = nbins;
    nbins += 1;
    bin_depths[new_bin_id] = bin_depths[bin_id] + 1;
    bin_parents[new_bin_id] = bin_id;

    // Extract bin indices from loop variable
    int xi = (i & 1) >> 0;
    int yi = (i & 2) >> 1;
    int zi = (i & 4) >> 2;

    // Determine the corners of the new bin
    if (xi == 0) {
      bin_corners[6 * new_bin_id + 0] = bin_corners[6 * bin_id + 0];
      bin_corners[6 * new_bin_id + 3] = xcen[0];
    } else {
      bin_corners[6 * new_bin_id + 0] = xcen[0];
      bin_corners[6 * new_bin_id + 3] = bin_corners[6 * bin_id + 3];
    }
    if (yi == 0) {
      bin_corners[6 * new_bin_id + 1] = bin_corners[6 * bin_id + 1];
      bin_corners[6 * new_bin_id + 4] = xcen[1];
    } else {
      bin_corners[6 * new_bin_id + 1] = xcen[1];
      bin_corners[6 * new_bin_id + 4] = bin_corners[6 * bin_id + 4];
    }
    if (zi == 0) {
      bin_corners[6 * new_bin_id + 2] = bin_corners[6 * bin_id + 2];
      bin_corners[6 * new_bin_id + 5] = xcen[2];
    } else {
      bin_corners[6 * new_bin_id + 2] = xcen[2];
      bin_corners[6 * new_bin_id + 5] = bin_corners[6 * bin_id + 5];
    }

    // Find the points that fall inside the new bin
    for (int i = 0; i < npts; i++) {
      if (points_bins[i] == bin_id) {
        double *x = &Xpts[3 * i];
        double *min_corner = &bin_corners[6 * new_bin_id];
        double *max_corner = &bin_corners[6 * new_bin_id + 3];
        bool is_in_bin = x[0] >= min_corner[0] and x[0] <= max_corner[0] and
                         x[1] >= min_corner[1] and x[1] <= max_corner[1] and
                         x[2] >= min_corner[2] and x[2] <= max_corner[2];
        if (is_in_bin) {
          points_bins[i] = new_bin_id;
        }
      }
    }

    // Divide the new bin and keep track of leaf bins
    bool is_leaf_bin = divide(new_bin_id);
    if (is_leaf_bin) {
      if (nleaf > 0) {
        int *new_leaf_bins = new int[nleaf + 1];
        memcpy(new_leaf_bins, leaf_bins, nleaf * sizeof(int));
        delete[] leaf_bins;
        leaf_bins = new_leaf_bins;
      }
      leaf_bins[nleaf] = new_bin_id;
      nleaf++;
    }
  }

  return false;
}
