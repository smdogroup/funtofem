"""
Tests for ParameterSweep._make_output_dirs.

Covers:
1. Default construction produces "sweep_output/<key>/cfd" and
   "sweep_output/<key>/struct"
2. Both output directories are siblings under one per-point directory when
   cfd_root and struct_root are equal
3. Distinct roots split the two trees
4. The per-point directory is where SlurmSweepSubmitter writes point.json,
   so with shared roots the point file sits alongside cfd/ and struct/
"""

import os
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Load the sweep subpackage under a private alias.
#
# These tests must not import the top-level funtofem package, which pulls in
# the native TACS/FUN3D extensions. They also must not leave a stub named
# "funtofem" in sys.modules: testflo imports every test module into a single
# process, so a stub would shadow the real package for the tests that run
# later. Loading the subpackage as "_f2f_sweep_test.sweep" keeps the sweep
# modules' relative imports working while leaving "funtofem" untouched.
# ---------------------------------------------------------------------------
import importlib.util
import pathlib

_SWEEP_DIR = pathlib.Path(__file__).parents[3] / "funtofem" / "sweep"
SWEEP_ROOT_PKG = "_f2f_sweep_test"
SWEEP_PKG = f"{SWEEP_ROOT_PKG}.sweep"

if SWEEP_PKG not in sys.modules:
    _root = types.ModuleType(SWEEP_ROOT_PKG)
    # Empty __path__ so the lazy "from ..driver import FUNtoFEMnlbgs" inside
    # parameter_sweep raises ImportError (which that call site handles) instead
    # of reaching the real funtofem.driver and its native dependencies.
    _root.__path__ = []
    sys.modules[SWEEP_ROOT_PKG] = _root

    _spec = importlib.util.spec_from_file_location(
        SWEEP_PKG,
        str(_SWEEP_DIR / "__init__.py"),
        submodule_search_locations=[str(_SWEEP_DIR)],
    )
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules[SWEEP_PKG] = _pkg
    _root.sweep = _pkg
    _spec.loader.exec_module(_pkg)

_sweep = sys.modules[SWEEP_PKG]

ParameterSweep = _sweep.ParameterSweep


class OutputDirLayoutTest(unittest.TestCase):
    """Pin the per-point output directory layout."""

    KEY = "pt_a8f3c2d14e91"

    def test_default_roots(self):
        """Default construction nests both kinds under one per-point directory."""
        sweep = ParameterSweep({"alpha": [1, 2]})

        # Pins the defaults themselves, not just the templates: a silent change
        # to either root default relocates every user's output on disk.
        self.assertEqual(sweep.cfd_root, "sweep_output")
        self.assertEqual(sweep.struct_root, "sweep_output")

        cfd_dir, struct_dir = sweep._make_output_dirs(self.KEY)

        self.assertEqual(cfd_dir, f"sweep_output/{self.KEY}/cfd")
        self.assertEqual(struct_dir, f"sweep_output/{self.KEY}/struct")

    def test_shared_root_makes_siblings(self):
        """Equal roots put cfd/ and struct/ side by side under one point dir."""
        sweep = ParameterSweep({"alpha": [1]}, cfd_root="shared", struct_root="shared")
        cfd_dir, struct_dir = sweep._make_output_dirs(self.KEY)

        self.assertEqual(os.path.dirname(cfd_dir), os.path.dirname(struct_dir))
        self.assertEqual(os.path.dirname(cfd_dir), f"shared/{self.KEY}")

    def test_point_file_is_sibling_of_output_dirs(self):
        """point.json lands beside cfd/ and struct/, not inside either.

        SlurmSweepSubmitter builds the point file path as
        "<cfd_root>/<key>/point.json" independently of _make_output_dirs, so
        this pins the two constructions against drifting apart.
        """
        sweep = ParameterSweep({"alpha": [1]})
        cfd_dir, struct_dir = sweep._make_output_dirs(self.KEY)
        point_json = os.path.join(sweep.cfd_root, self.KEY, "point.json")

        self.assertEqual(os.path.dirname(point_json), os.path.dirname(cfd_dir))
        self.assertEqual(os.path.dirname(point_json), os.path.dirname(struct_dir))

    def test_distinct_roots_split_trees(self):
        """Different roots separate the two trees entirely."""
        sweep = ParameterSweep(
            {"alpha": [1]}, cfd_root="/scratch/cfd", struct_root="/home/struct"
        )
        cfd_dir, struct_dir = sweep._make_output_dirs(self.KEY)

        self.assertEqual(cfd_dir, f"/scratch/cfd/{self.KEY}/cfd")
        self.assertEqual(struct_dir, f"/home/struct/{self.KEY}/struct")
        self.assertNotEqual(os.path.dirname(cfd_dir), os.path.dirname(struct_dir))

    def test_key_is_not_interpreted(self):
        """The key is substituted verbatim, including custom key_fn output."""
        sweep = ParameterSweep({"alpha": [1]}, key_fn=lambda p: "alpha_1.5")
        cfd_dir, struct_dir = sweep._make_output_dirs("alpha_1.5")

        self.assertEqual(cfd_dir, "sweep_output/alpha_1.5/cfd")
        self.assertEqual(struct_dir, "sweep_output/alpha_1.5/struct")


if __name__ == "__main__":
    unittest.main()
