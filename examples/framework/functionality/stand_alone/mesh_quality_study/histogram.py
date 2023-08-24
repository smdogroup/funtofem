"""
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
"""

"""
--------------------------------------------------------------------------------
Histogram
--------------------------------------------------------------------------------
The following script parses files from Tecplot that grid quality function
evaluations and creates histograms in order to visualize the change in quality
of the grid before and after it is transformed by displacement transfer.
"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

"""
--------------------------------------------------------------------------------
Parsing file
--------------------------------------------------------------------------------
"""


def loadGridQualityEval(filename, node_based=True):
    """
    Returns an array of evaluations of a grid quality function (at nodes or at
    cell-centers of grid) from a file

    Arguments:
    ----------
    filename: name of file to load
    node_based: True if node-based, False if cell-centered
    """
    f = open(filename, "r")

    # Assume node/element data is given on line 5 and extract data
    for i in range(4):
        line = f.readline()
    line5 = f.readline()
    info_list = [info.strip(",") for info in line5.split()]
    num_nodes = int(info_list[0].split("=")[1])
    num_cells = int(info_list[1].split("=")[1])

    # Skip next two lines (data begins after line 7)
    line = f.readline()
    line = f.readline()

    # Read in data
    num_lines = num_nodes if node_based else num_cells
    feval = np.array([])
    for i in range(num_lines):
        feval = np.append(feval, float(f.readline().strip()))

    return feval


"""
--------------------------------------------------------------------------------
Making histograms
--------------------------------------------------------------------------------
"""
# Make skewness histogram
skew_undeformed = loadGridQualityEval("aero_skew.dat")
skew_deformed = loadGridQualityEval("aero_def_skew.dat")

skew_hist = plt.figure()
nbins = 20
bins = np.linspace(0.0, skew_deformed.max(), nbins)
n, bins, patches = plt.hist(
    skew_undeformed, bins, facecolor="blue", label="undeformed", alpha=0.5, normed=1
)
n, bins, patches = plt.hist(
    skew_deformed, bins, facecolor="red", label="deformed", alpha=0.5, normed=1
)
plt.xlabel(r"k face skewness")
plt.ylabel(r"number of nodes")
plt.legend()
plt.grid(True)
skew_hist.savefig("skew_hist.png")


# Make aspect ratio histogram
aspect_undeformed = loadGridQualityEval("aero_aspect.dat")
aspect_deformed = loadGridQualityEval("aero_def_aspect.dat")

aspect_hist = plt.figure()
nbins = 20
mean_orig_aspect = np.mean(aspect_undeformed)
range_def_aspect = aspect_deformed.max() - aspect_deformed.min()
lb = mean_orig_aspect - 0.5 * range_def_aspect
ub = mean_orig_aspect + 0.5 * range_def_aspect
bins = np.linspace(lb, ub, 20)
n, bins, patches = plt.hist(
    aspect_undeformed, bins, facecolor="blue", label="undeformed", alpha=0.5, normed=1
)
n, bins, patches = plt.hist(
    aspect_deformed, bins, facecolor="red", label="deformed", alpha=0.5, normed=1
)
plt.xlabel(r"k aspect ratio")
plt.ylabel(r"number of nodes")
plt.legend()
plt.grid(True)
aspect_hist.savefig("aspect_hist.png")
