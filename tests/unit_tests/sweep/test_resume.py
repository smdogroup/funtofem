"""
Tests for ParameterSweep resume logic (Task 8.2, sub-task 4).

Covers:
1. resume=True with existing CSV: skips successful points, re-runs failed and missing points
2. resume=True with no CSV: runs all points normally
3. resume=False: runs all points regardless of existing CSV
4. The resulting CSV has exactly one header line
"""

import csv
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
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
_make_key = _sweep.parameter_sweep._make_key
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


def _make_sweep(output_csv: str, params=None) -> ParameterSweep:
    """Return a minimal ParameterSweep with MeshMode.NONE and the given output CSV path."""
    if params is None:
        params = {"alpha": [1.0, 2.0, 3.0]}
    return ParameterSweep(
        params,
        mesh_mode=MeshMode.NONE,
        output_csv=output_csv,
    )


def _read_csv_rows(csv_path: str) -> list:
    """Read all rows from a CSV file and return them as a list of dicts."""
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _count_header_lines(csv_path: str) -> int:
    """Count how many lines in the CSV look like headers (contain 'case' and 'key')."""
    with open(csv_path, "r", newline="") as f:
        lines = f.readlines()
    return sum(
        1 for line in lines if "case" in line and "key" in line and "status" in line
    )


def _write_preexisting_csv(csv_path: str, rows: list[dict]) -> None:
    """Write a pre-existing CSV file to simulate a previous partial sweep."""
    if not rows:
        return
    # Collect all fieldnames across all rows to handle optional columns like "error"
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _setup_sweep_with_tracking(output_csv: str, params=None):
    """Create a sweep with a tracking model builder that records which points ran."""
    if params is None:
        params = {"alpha": [1.0, 2.0, 3.0]}

    sweep = _make_sweep(output_csv, params)
    executed_alphas = []

    def model_builder(comm, design_point, cfd_dir, struct_dir):
        executed_alphas.append(design_point.get("alpha"))
        return MagicMock()

    sweep.set_model_builder(model_builder)
    sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
    sweep.set_result_extractor(MagicMock(return_value={"result": 1.0}))

    return sweep, executed_alphas


# ---------------------------------------------------------------------------
# Test 1: resume=True with existing CSV — skip successes, re-run failed/missing
# ---------------------------------------------------------------------------


class TestResumeSkipsSuccessfulPoints(unittest.TestCase):
    def test_successful_points_are_skipped(self):
        """With resume=True, points that are 'success' in the existing CSV are not re-run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0, 3.0]}

            # Compute keys for alpha=1.0 and alpha=2.0
            key1 = _make_key({"alpha": 1.0})
            key2 = _make_key({"alpha": 2.0})

            # Pre-populate CSV: alpha=1 and alpha=2 succeeded, alpha=3 is missing
            _write_preexisting_csv(
                csv_path,
                [
                    {
                        "case": 1,
                        "key": key1,
                        "alpha": 1.0,
                        "status": "success",
                        "result": 0.1,
                    },
                    {
                        "case": 2,
                        "key": key2,
                        "alpha": 2.0,
                        "status": "success",
                        "result": 0.2,
                    },
                ],
            )

            sweep, executed_alphas = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            with redirect_stdout(io.StringIO()):
                sweep.run(comm, resume=True)

            # Only alpha=3.0 should have been executed
            self.assertEqual(
                executed_alphas,
                [3.0],
                f"Expected only alpha=3.0 to run, got: {executed_alphas}",
            )

    def test_failed_points_are_re_run(self):
        """With resume=True, points that are 'failed' in the existing CSV are re-run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0]}

            key1 = _make_key({"alpha": 1.0})
            key2 = _make_key({"alpha": 2.0})

            # alpha=1 succeeded, alpha=2 failed
            _write_preexisting_csv(
                csv_path,
                [
                    {
                        "case": 1,
                        "key": key1,
                        "alpha": 1.0,
                        "status": "success",
                        "result": 0.1,
                    },
                    {
                        "case": 2,
                        "key": key2,
                        "alpha": 2.0,
                        "status": "failed",
                        "error": "boom",
                    },
                ],
            )

            sweep, executed_alphas = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            with redirect_stdout(io.StringIO()):
                sweep.run(comm, resume=True)

            # Only alpha=2.0 (failed) should re-run
            self.assertEqual(
                executed_alphas,
                [2.0],
                f"Expected only alpha=2.0 to re-run, got: {executed_alphas}",
            )

    def test_missing_points_are_run(self):
        """With resume=True, points absent from the CSV are run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0, 3.0]}

            key1 = _make_key({"alpha": 1.0})

            # Only alpha=1.0 is in the CSV as success; 2.0 and 3.0 are missing
            _write_preexisting_csv(
                csv_path,
                [
                    {
                        "case": 1,
                        "key": key1,
                        "alpha": 1.0,
                        "status": "success",
                        "result": 0.1,
                    },
                ],
            )

            sweep, executed_alphas = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            with redirect_stdout(io.StringIO()):
                sweep.run(comm, resume=True)

            self.assertNotIn(1.0, executed_alphas)
            self.assertIn(2.0, executed_alphas)
            self.assertIn(3.0, executed_alphas)

    def test_skip_message_printed_for_skipped_points(self):
        """With resume=True, a SKIPPED message is printed for each skipped point."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0]}

            key1 = _make_key({"alpha": 1.0})

            _write_preexisting_csv(
                csv_path,
                [
                    {
                        "case": 1,
                        "key": key1,
                        "alpha": 1.0,
                        "status": "success",
                        "result": 0.1,
                    },
                ],
            )

            sweep, _ = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            output = io.StringIO()
            with redirect_stdout(output):
                sweep.run(comm, resume=True)

            printed = output.getvalue()
            self.assertIn("SKIPPED (resume)", printed)
            self.assertIn(key1, printed)

    def test_mixed_status_csv_runs_only_non_success(self):
        """With resume=True and mixed statuses, only non-success points run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0, 3.0, 4.0]}

            key1 = _make_key({"alpha": 1.0})
            key2 = _make_key({"alpha": 2.0})
            key3 = _make_key({"alpha": 3.0})

            _write_preexisting_csv(
                csv_path,
                [
                    {"case": 1, "key": key1, "alpha": 1.0, "status": "success"},
                    {
                        "case": 2,
                        "key": key2,
                        "alpha": 2.0,
                        "status": "failed",
                        "error": "err",
                    },
                    {"case": 3, "key": key3, "alpha": 3.0, "status": "success"},
                ],
            )

            sweep, executed_alphas = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            with redirect_stdout(io.StringIO()):
                sweep.run(comm, resume=True)

            # alpha=1.0 and 3.0 were success → skip; alpha=2.0 (failed) and 4.0 (missing) → run
            self.assertNotIn(1.0, executed_alphas)
            self.assertNotIn(3.0, executed_alphas)
            self.assertIn(2.0, executed_alphas)
            self.assertIn(4.0, executed_alphas)


# ---------------------------------------------------------------------------
# Test 2: resume=True with no CSV — runs all points normally
# ---------------------------------------------------------------------------


class TestResumeWithNoCsv(unittest.TestCase):
    def test_all_points_run_when_no_csv_exists(self):
        """With resume=True and no pre-existing CSV, all design points are run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            # No CSV file created

            params = {"alpha": [1.0, 2.0, 3.0]}
            sweep, executed_alphas = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            self.assertFalse(os.path.exists(csv_path))

            with redirect_stdout(io.StringIO()):
                sweep.run(comm, resume=True)

            self.assertEqual(sorted(executed_alphas), [1.0, 2.0, 3.0])

    def test_no_error_when_csv_missing_with_resume(self):
        """resume=True with no existing CSV must not raise any error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "nonexistent.csv")
            params = {"alpha": [1.0]}
            sweep, _ = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            # Must not raise
            try:
                with redirect_stdout(io.StringIO()):
                    sweep.run(comm, resume=True)
            except Exception as e:
                self.fail(f"run(resume=True) raised unexpectedly with no CSV: {e}")


# ---------------------------------------------------------------------------
# Test 3: resume=False — runs all points regardless of existing CSV
# ---------------------------------------------------------------------------


class TestResumeDisabled(unittest.TestCase):
    def test_all_points_run_when_resume_false(self):
        """With resume=False, all design points run even if CSV has successful entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0, 3.0]}

            key1 = _make_key({"alpha": 1.0})
            key2 = _make_key({"alpha": 2.0})
            key3 = _make_key({"alpha": 3.0})

            # All three points already succeeded in the pre-existing CSV
            _write_preexisting_csv(
                csv_path,
                [
                    {"case": 1, "key": key1, "alpha": 1.0, "status": "success"},
                    {"case": 2, "key": key2, "alpha": 2.0, "status": "success"},
                    {"case": 3, "key": key3, "alpha": 3.0, "status": "success"},
                ],
            )

            sweep, executed_alphas = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            with redirect_stdout(io.StringIO()):
                sweep.run(comm, resume=False)

            # All points must run despite existing successful CSV
            self.assertEqual(sorted(executed_alphas), [1.0, 2.0, 3.0])

    def test_default_resume_is_false(self):
        """The default for resume parameter is False — all points should run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0]}

            key1 = _make_key({"alpha": 1.0})

            _write_preexisting_csv(
                csv_path,
                [
                    {"case": 1, "key": key1, "alpha": 1.0, "status": "success"},
                ],
            )

            sweep, executed_alphas = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            # Call run() without passing resume (should default to False)
            with redirect_stdout(io.StringIO()):
                sweep.run(comm)

            # Both points should run since resume=False (default)
            self.assertIn(1.0, executed_alphas)
            self.assertIn(2.0, executed_alphas)


# ---------------------------------------------------------------------------
# Test 4: The resulting CSV has exactly one header line
# ---------------------------------------------------------------------------


class TestSingleHeaderInvariant(unittest.TestCase):
    def test_single_header_after_full_resume_sweep(self):
        """After a resume run, the output CSV must have exactly one header line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0, 3.0]}

            key1 = _make_key({"alpha": 1.0})

            # alpha=1.0 already succeeded; 2.0 and 3.0 will be written fresh
            _write_preexisting_csv(
                csv_path,
                [
                    {
                        "case": 1,
                        "key": key1,
                        "alpha": 1.0,
                        "status": "success",
                        "result": 0.1,
                    },
                ],
            )

            sweep, _ = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            with redirect_stdout(io.StringIO()):
                sweep.run(comm, resume=True)

            header_count = _count_header_lines(csv_path)
            self.assertEqual(
                header_count,
                1,
                f"Expected exactly one header line, found {header_count}",
            )

    def test_single_header_after_fresh_sweep(self):
        """After a sweep on a new CSV, there must be exactly one header line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0, 3.0]}

            sweep, _ = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            with redirect_stdout(io.StringIO()):
                sweep.run(comm)

            header_count = _count_header_lines(csv_path)
            self.assertEqual(
                header_count,
                1,
                f"Expected exactly one header line, found {header_count}",
            )

    def test_single_header_when_all_points_skipped(self):
        """When all points are skipped (resume), the pre-existing CSV header count stays at 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            params = {"alpha": [1.0, 2.0]}

            key1 = _make_key({"alpha": 1.0})
            key2 = _make_key({"alpha": 2.0})

            _write_preexisting_csv(
                csv_path,
                [
                    {"case": 1, "key": key1, "alpha": 1.0, "status": "success"},
                    {"case": 2, "key": key2, "alpha": 2.0, "status": "success"},
                ],
            )

            sweep, executed_alphas = _setup_sweep_with_tracking(csv_path, params)
            comm = _make_mock_comm()

            with redirect_stdout(io.StringIO()):
                sweep.run(comm, resume=True)

            # No new points executed
            self.assertEqual(executed_alphas, [])

            # CSV still has exactly one header
            header_count = _count_header_lines(csv_path)
            self.assertEqual(header_count, 1)


if __name__ == "__main__":
    unittest.main()
