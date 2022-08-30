#ifndef LOCATE_POINT_H
#define LOCATE_POINT_H

#include "TransferScheme.h"

/*!
  Given a set of points in R^3, locate the closest one to a given
  point in O(log(N)) time -- after an initial O(N) setup time.

  Copyright (c) 2010 Graeme Kennedy. All rights reserved.
  Not for commercial purposes.
*/

class LocatePoint {
 public:
  LocatePoint(const F2FScalar *_Xpts, int _npts, int _max_num_points);
  ~LocatePoint();

  // Return the index of the point in the array
  // ------------------------------------------
  int locateClosest(const F2FScalar xpt[]);
  int locateExhaustive(const F2FScalar xpt[]);

  // Locate the K-closest points (note that dist/indices must of length K)
  // ---------------------------------------------------------------------
  void locateKClosest(int K, int indices[], F2FScalar dist[],
                      const F2FScalar xpt[]);
  void locateKExhaustive(int K, int indices[], F2FScalar dist[],
                         const F2FScalar xpt[]);

  // Find the point with the closest taxi-cab distance to the plane
  // --------------------------------------------------------------
  void locateClosestTaxi(int K, int indices[], F2FScalar dist[],
                         const F2FScalar xpt[], const F2FScalar n[]);

 private:
  // The recursive versions of the above functions
  void locateClosest(int root, const F2FScalar xpt[], F2FScalar *dist,
                     int *index);
  void locateKClosest(int K, int root, const F2FScalar xpt[], F2FScalar *dist,
                      int *indices, int *nk);

  // Insert the index into the sorted list of indices
  void insertIndex(F2FScalar *dist, int *indices, int *nk, F2FScalar d,
                   int dindex, int K);

  // Sort the list of initial indices into the tree data structure
  int split(int start, int end);
  int splitList(F2FScalar xav[], F2FScalar normal[], int *indices, int npts);

  // Functions for array management
  void extendArrays(int old_len, int new_len);
  int *newIntArray(int *array, int old_len, int new_len);
  F2FScalar *newDoubleArray(F2FScalar *array, int old_len, int new_len);

  // The cloud of points to match
  const F2FScalar *Xpts;
  int npts;

  int max_num_points;  // Maximum number of points stored at a leaf

  // Keep track of the nodes that have been created
  int max_nodes;
  int num_nodes;

  int *indices;         // Indices into the array of points
  int *nodes;           // Indices from the current node to the two child nodes
  int *indices_ptr;     // Pointer into the global indices array
  int *num_indices;     // Number of indices associated with this node
  F2FScalar *node_xav;  // Origin point for the array
  F2FScalar *node_normal;  // Normal direction of the plane
};

#endif
