# Connector level
set conParams(InitDim)                          11; # Initial connector dimension
set conParams(MaxDim)                           1024; # Maximum connector dimension
set conParams(MinDim)                           4; # Minimum connector dimension
set conParams(TurnAngle)                       1.000000; # Maximum turning angle on connectors for dimensioning (0 - not used)
set conParams(Deviation)                       0.000000; # Maximum deviation on connectors for dimensioning (0 - not used)
set conParams(SplitAngle)                      0.000000; # Turning angle on connectors to split (0 - not used)
set conParams(JoinCons)                          0; # Perform joining operation on 2 connectors at one endpoint
set conParams(ProxGrowthRate)                  1.200000; # Connector proximity growth rate
set conParams(AdaptSources)                     0; # Compute sources using connectors (0 - not used) V18.2+ (experimental)
set conParams(SourceSpacing)                    1; # Use source cloud for adaptive pass on connectors V18.2+
set conParams(TurnAngleHard)                   70.000000; # Hard edge turning angle limit for domain T-Rex (0.0 - not used)

# Domain level
set domParams(Algorithm)                    "AdvancingFront"; # Isotropic (Delaunay, AdvancingFront or AdvancingFrontOrtho)
set domParams(FullLayers)                       0; # Domain full layers (0 for multi-normals, >= 1 for single normal)
set domParams(MaxLayers)                        15; # Domain maximum layers
set domParams(GrowthRate)                      1.750000; # Domain growth rate for 2D T-Rex extrusion
set domParams(IsoType)                      "Triangle"; # Domain iso cell type (Triangle or TriangleQuad)
set domParams(TRexType)                     "Triangle"; # Domain T-Rex cell type (Triangle or TriangleQuad)
set domParams(TRexARLimit)                     40.000000; # Domain T-Rex maximum aspect ratio limit (0 - not used)
set domParams(TRexAngleBC)                     0.000000; # Domain T-Rex spacing from surface curvature
set domParams(Decay)                           0.500000; # Domain boundary decay
set domParams(MinEdge)                         0.000000; # Domain minimum edge length
set domParams(MaxEdge)                         0.000000; # Domain maximum edge length
set domParams(Adapt)                            0; # Set up all domains for adaptation (0 - not used) V18.2+ (experimental)
set domParams(WallSpacing)                     0.010000; # defined spacing when geometry attributed with $wall
set domParams(StrDomConvertARTrigger)          0.000000; # Aspect ratio to trigger converting domains to structured

# Block level
set blkParams(Algorithm)                    "Delaunay"; # Isotropic (Delaunay, Voxel) (V18.3+)
set blkParams(VoxelLayers)                      3; # Number of Voxel transition layers if Algorithm set to Voxel (V18.3+)
set blkParams(boundaryDecay)                   0.500000; # Volumetric boundary decay
set blkParams(collisionBuffer)                 1.000000; # Collision buffer for colliding T-Rex fronts
set blkParams(maxSkewAngle)                    170.000000; # Maximum skew angle for T-Rex extrusion
set blkParams(TRexSkewDelay)                    0; # Number of layers to delay enforcement of skew criteria
set blkParams(edgeMaxGrowthRate)               2.000000; # Volumetric edge ratio
set blkParams(fullLayers)                       1; # Full layers (0 for multi-normals, >= 1 for single normal)
set blkParams(maxLayers)                        100; # Maximum layers
set blkParams(growthRate)                      1.300000; # Growth rate for volume T-Rex extrusion
set blkParams(TRexType)                     "TetPyramid"; # T-Rex cell type (TetPyramid, TetPyramidPrismHex, AllAndConvertWallDoms)
set blkParams(volInitialize)                     1; # Initialize block after setup

# General
set genParams(SkipMeshing)                       1; # Skip meshing of domains during interim processing (V18.3+)
set genParams(CAESolver)                 "UGRID"; # Selected CAE Solver (Currently support CGNS, Gmsh and UGRID)
set genParams(outerBoxScale)                     0; # Enclose geometry in box with specified scale (0 - no box)
set genParams(sourceBoxLengthScale)            0.000000; # Length scale of enclosed viscous walls in source box (0 - no box)
set genParams(sourceBoxDirection)  { 1.000000 0.000000 0.000000 }; # Principal direction vector (i.e. normalized freestream vector)
set genParams(sourceBoxAngle)                  0.000000; # Angle for widening source box in the assigned direction
set genParams(sourceGrowthFactor)              10.000000; # Growth rate for spacing value along box
set genParams(ModelSize)                         0; # Set model size before CAD import (0 - get from file)
set genParams(writeGMA)                    "2.0"; # Write out geometry-mesh associativity file version (0.0 - none, 1.0 or 2.0)
set genParams(assembleTolMult)                 2.0; # Multiplier on model assembly tolerance for allowed MinEdge
set genParams(modelOrientIntoMeshVolume)         1; # Whether the model is oriented so normals point into the mesh

