#ifndef OCTREE_H
#define OCTREE_H

#include "TransferScheme.h"

/*
  A class implementing a matrix-based octree
*/
class F2F_API Octree {
 public:
  Octree(F2FScalar *points, int num_points, 
         int min_point_count, double min_edge_length, int max_tree_depth);

  ~Octree();

  // Create octree
  void generate();

  // Public tree data
  int nbins; // total number of bins created
  int *bin_depths; // depth of each bin in tree
  int *bin_parents; // ID of parent of each bin
  double *bin_corners; // min and max corners of bins
  int *points_bins; // ID of bin that each point is in
  int nleaf; // number of leaf bins
  int *leaf_bins; // IDs of leaf bins

 private:
  // Recursive function used by initialize to create octree
  bool divide(int bin_num);

  // Recursion exit conditions
  int min_points;
  double min_edge;
  int max_depth;

  // Private tree data
  int npts;
  double *Xpts;
};

#endif // OCTREE_H
