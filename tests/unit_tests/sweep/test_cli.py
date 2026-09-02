"""
Tests for load_point_file and cli_main (Task 8.2, sub-task 5).

Covers:
1. load_point_file correctly loads a JSON file
2. cli_main calls sweep.run_single with the correct design point
3. Invalid path raises FileNotFoundError
"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

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
load_point_file = _sweep.load_point_file
cli_main = _sweep.cli_main
MeshMode = _sweep.MeshMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_comm(rank: int = 0):
    """Return a minimal mock MPI communicator."""
    comm = MagicMock()
    comm.rank = rank
    comm.Barrier = MagicMock()
    return comm


def _write_point_json(
    tmpdir: str, design_point: dict, filename: str = "point.json"
) -> str:
    """Write a design point to a JSON file and return the path."""
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        json.dump(design_point, f)
    return path


# ---------------------------------------------------------------------------
# Test 1: load_point_file correctly loads a JSON file
# ---------------------------------------------------------------------------


class TestLoadPointFile(unittest.TestCase):
    def test_loads_simple_design_point(self):
        """load_point_file reads a JSON file and returns a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_point = {"xradius": 0.0381, "zradius": 0.041, "thick": 0.0127}
            path = _write_point_json(tmpdir, design_point)

            result = load_point_file(path)

            self.assertIsInstance(result, dict)
            self.assertAlmostEqual(result["xradius"], 0.0381)
            self.assertAlmostEqual(result["zradius"], 0.041)
            self.assertAlmostEqual(result["thick"], 0.0127)

    def test_loads_integer_values(self):
        """load_point_file correctly loads integer values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_point = {"n_layers": 5, "mesh_level": 2}
            path = _write_point_json(tmpdir, design_point)

            result = load_point_file(path)

            self.assertEqual(result["n_layers"], 5)
            self.assertEqual(result["mesh_level"], 2)

    def test_loads_string_values(self):
        """load_point_file correctly loads string values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_point = {"material": "aluminum", "profile": "NACA0012"}
            path = _write_point_json(tmpdir, design_point)

            result = load_point_file(path)

            self.assertEqual(result["material"], "aluminum")
            self.assertEqual(result["profile"], "NACA0012")

    def test_loads_single_parameter(self):
        """load_point_file works for a single-parameter design point."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_point = {"alpha": 5.0}
            path = _write_point_json(tmpdir, design_point)

            result = load_point_file(path)

            self.assertEqual(result, {"alpha": 5.0})

    def test_returns_dict_with_all_keys(self):
        """load_point_file returns a dict with all keys from the JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_point = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
            path = _write_point_json(tmpdir, design_point)

            result = load_point_file(path)

            self.assertEqual(set(result.keys()), {"a", "b", "c", "d"})


# ---------------------------------------------------------------------------
# Test 2: cli_main calls sweep.run_single with the correct design point
# ---------------------------------------------------------------------------


class TestCliMain(unittest.TestCase):
    def test_cli_main_calls_run_single_with_correct_point(self):
        """cli_main should parse --point-file and call sweep.run_single with the design point."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_point = {"xradius": 0.0381, "zradius": 0.041}
            path = _write_point_json(tmpdir, design_point)

            # Create a mock sweep object
            mock_sweep = MagicMock(spec=ParameterSweep)
            comm = _make_mock_comm()

            # Patch sys.argv to simulate CLI invocation
            with patch("sys.argv", ["sweep_script.py", "--point-file", path]):
                cli_main(mock_sweep, comm)

            # run_single must be called exactly once with the correct design_point
            mock_sweep.run_single.assert_called_once_with(
                comm, design_point, compute_adjoint=False
            )

    def test_cli_main_passes_compute_adjoint_true(self):
        """cli_main passes compute_adjoint=True to run_single when specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_point = {"alpha": 3.0}
            path = _write_point_json(tmpdir, design_point)

            mock_sweep = MagicMock(spec=ParameterSweep)
            comm = _make_mock_comm()

            with patch("sys.argv", ["sweep_script.py", "--point-file", path]):
                cli_main(mock_sweep, comm, compute_adjoint=True)

            mock_sweep.run_single.assert_called_once_with(
                comm, design_point, compute_adjoint=True
            )

    def test_cli_main_passes_compute_adjoint_false(self):
        """cli_main passes compute_adjoint=False by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_point = {"alpha": 1.5}
            path = _write_point_json(tmpdir, design_point)

            mock_sweep = MagicMock(spec=ParameterSweep)
            comm = _make_mock_comm()

            with patch("sys.argv", ["sweep_script.py", "--point-file", path]):
                cli_main(mock_sweep, comm)

            _, kwargs = mock_sweep.run_single.call_args
            self.assertFalse(kwargs.get("compute_adjoint", False))

    def test_cli_main_reads_correct_design_point_values(self):
        """cli_main passes the exact float values from the JSON file to run_single."""
        with tempfile.TemporaryDirectory() as tmpdir:
            design_point = {"xradius": 0.0381, "zradius": 0.041, "thick": 0.0127}
            path = _write_point_json(tmpdir, design_point)

            mock_sweep = MagicMock(spec=ParameterSweep)
            comm = _make_mock_comm()

            with patch("sys.argv", ["sweep_script.py", "--point-file", path]):
                cli_main(mock_sweep, comm)

            call_args = mock_sweep.run_single.call_args
            passed_point = call_args[0][1]  # positional arg at index 1

            self.assertAlmostEqual(passed_point["xradius"], 0.0381)
            self.assertAlmostEqual(passed_point["zradius"], 0.041)
            self.assertAlmostEqual(passed_point["thick"], 0.0127)


# ---------------------------------------------------------------------------
# Test 3: Invalid path raises FileNotFoundError
# ---------------------------------------------------------------------------


class TestLoadPointFileInvalidPath(unittest.TestCase):
    def test_nonexistent_file_raises_file_not_found_error(self):
        """load_point_file raises FileNotFoundError for a nonexistent path."""
        with self.assertRaises(FileNotFoundError):
            load_point_file("/nonexistent/path/that/does/not/exist/point.json")

    def test_nonexistent_file_in_tmpdir_raises(self):
        """load_point_file raises FileNotFoundError for any nonexistent path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = os.path.join(tmpdir, "does_not_exist.json")
            with self.assertRaises(FileNotFoundError):
                load_point_file(fake_path)

    def test_cli_main_with_missing_point_file_raises(self):
        """cli_main raises FileNotFoundError if the --point-file path does not exist."""
        mock_sweep = MagicMock(spec=ParameterSweep)
        comm = _make_mock_comm()

        with patch(
            "sys.argv", ["sweep_script.py", "--point-file", "/no/such/file.json"]
        ):
            with self.assertRaises(FileNotFoundError):
                cli_main(mock_sweep, comm)


# ---------------------------------------------------------------------------
# Test: load_point_file and cli_main are exported from sweep/__init__.py
# ---------------------------------------------------------------------------


class TestExports(unittest.TestCase):
    def test_load_point_file_accessible_from_module(self):
        """load_point_file must be importable from funtofem.sweep.parameter_sweep."""
        self.assertTrue(callable(load_point_file))

    def test_cli_main_accessible_from_module(self):
        """cli_main must be importable from funtofem.sweep.parameter_sweep."""
        self.assertTrue(callable(cli_main))


if __name__ == "__main__":
    unittest.main()
