"""
Tests for ParameterSweep._write_csv_row (Task 8.1).

Covers:
1. Header is written when CSV does not exist
2. Header is NOT rewritten on append (exactly one header line after multiple writes)
3. All required columns are present: case, key, param names, status, result keys
4. Non-rank-0 does NOT write
5. Non-scalar values are coerced to str
6. error column appears on failure, not on success
7. Results from multiple points are all appended correctly
"""

import csv
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

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
MeshMode = _sweep.MeshMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_comm(rank: int = 0):
    """Return a minimal mock MPI communicator."""
    comm = MagicMock()
    comm.rank = rank
    return comm


def _make_sweep(output_csv: str) -> ParameterSweep:
    """Return a ParameterSweep with MeshMode.NONE and the given output CSV path."""
    return ParameterSweep(
        {"alpha": [1, 2]},
        mesh_mode=MeshMode.NONE,
        output_csv=output_csv,
    )


def _read_csv_rows(csv_path: str) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Test 1: Header is written when CSV doesn't exist
# ---------------------------------------------------------------------------


class TestHeaderWrittenOnFirstWrite(unittest.TestCase):
    def test_csv_created_with_header_on_first_write(self):
        """When the CSV doesn't exist, _write_csv_row must create it with a header row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            self.assertFalse(os.path.exists(csv_path))

            sweep._write_csv_row(
                comm, 1, "pt_abc", {"alpha": 1.0}, "success", {"lift": 9.9}, None
            )

            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
            self.assertIn("case", header)
            self.assertIn("key", header)
            self.assertIn("status", header)


# ---------------------------------------------------------------------------
# Test 2: Header NOT rewritten on append (exactly one header line)
# ---------------------------------------------------------------------------


class TestSingleHeaderAcrossMultipleWrites(unittest.TestCase):
    def test_exactly_one_header_line_after_multiple_writes(self):
        """After multiple _write_csv_row calls the CSV must contain exactly one header line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            sweep._write_csv_row(
                comm, 1, "pt_001", {"alpha": 1.0}, "success", {"lift": 1.0}, None
            )
            sweep._write_csv_row(
                comm, 2, "pt_002", {"alpha": 2.0}, "success", {"lift": 2.0}, None
            )
            sweep._write_csv_row(
                comm, 3, "pt_003", {"alpha": 3.0}, "success", {"lift": 3.0}, None
            )

            header_count = _count_header_lines(csv_path)
            self.assertEqual(
                header_count,
                1,
                f"Expected exactly one header line, found {header_count}",
            )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), 3)


# ---------------------------------------------------------------------------
# Test 3: All required columns are present
# ---------------------------------------------------------------------------


class TestRequiredColumnsPresent(unittest.TestCase):
    def test_all_required_columns_written(self):
        """CSV must have case, key, param columns, status, and result key columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            design_point = {"alpha": 1.5, "beta": 0.3}
            results = {"lift": 9.9, "drag": 0.5}

            sweep._write_csv_row(
                comm, 1, "pt_req", design_point, "success", results, None
            )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), 1)
            row = rows[0]

            # Required metadata columns
            self.assertIn("case", row)
            self.assertIn("key", row)
            self.assertIn("status", row)

            # One column per parameter
            self.assertIn("alpha", row)
            self.assertIn("beta", row)

            # One column per result key
            self.assertIn("lift", row)
            self.assertIn("drag", row)

            # Verify values
            self.assertEqual(row["case"], "1")
            self.assertEqual(row["key"], "pt_req")
            self.assertEqual(row["status"], "success")

    def test_param_values_written_correctly(self):
        """Parameter values must be written to their named columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            design_point = {"xradius": 0.0381, "thick": 0.0127}
            sweep._write_csv_row(
                comm, 5, "pt_xyz", design_point, "success", {"cl": 1.2}, None
            )

            rows = _read_csv_rows(csv_path)
            row = rows[0]
            self.assertEqual(row["xradius"], "0.0381")
            self.assertEqual(row["thick"], "0.0127")
            self.assertEqual(row["cl"], "1.2")


# ---------------------------------------------------------------------------
# Test 4: Non-rank-0 does NOT write
# ---------------------------------------------------------------------------


class TestRankZeroGuard(unittest.TestCase):
    def test_non_rank_0_does_not_write(self):
        """Rank != 0 must not create or modify the CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)

            # Write from rank 1 — should be a no-op
            comm_rank1 = _make_mock_comm(rank=1)
            sweep._write_csv_row(
                comm_rank1,
                1,
                "pt_nowrite",
                {"alpha": 1.0},
                "success",
                {"lift": 9.9},
                None,
            )

            self.assertFalse(
                os.path.exists(csv_path),
                "CSV file must not be created when called from rank != 0",
            )

    def test_rank_0_does_write(self):
        """Rank 0 must create the CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)

            comm_rank0 = _make_mock_comm(rank=0)
            sweep._write_csv_row(
                comm_rank0,
                1,
                "pt_write",
                {"alpha": 1.0},
                "success",
                {"lift": 9.9},
                None,
            )

            self.assertTrue(os.path.exists(csv_path))

    def test_mixed_ranks_only_rank0_data_persists(self):
        """If rank 0 writes once, then rank 1 tries to write, only the first row is in CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)

            comm_rank0 = _make_mock_comm(rank=0)
            comm_rank1 = _make_mock_comm(rank=1)

            sweep._write_csv_row(
                comm_rank0, 1, "pt_r0", {"alpha": 1.0}, "success", {"lift": 1.0}, None
            )
            sweep._write_csv_row(
                comm_rank1, 2, "pt_r1", {"alpha": 2.0}, "success", {"lift": 2.0}, None
            )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["key"], "pt_r0")


# ---------------------------------------------------------------------------
# Test 5: Non-scalar values coerced to str
# ---------------------------------------------------------------------------


class TestNonScalarCoercion(unittest.TestCase):
    def test_list_result_coerced_to_str(self):
        """A list result value must be coerced to str, not fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            results = {"residuals": [1.0, 2.0, 3.0]}
            sweep._write_csv_row(
                comm, 1, "pt_list", {"alpha": 1.0}, "success", results, None
            )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), 1)
            # The list should be coerced to str
            self.assertEqual(rows[0]["residuals"], str([1.0, 2.0, 3.0]))

    def test_dict_result_coerced_to_str(self):
        """A dict result value must be coerced to str."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            result_val = {"x": 1, "y": 2}
            results = {"nested": result_val}
            sweep._write_csv_row(
                comm, 1, "pt_dict", {"alpha": 1.0}, "success", results, None
            )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(rows[0]["nested"], str(result_val))

    def test_scalar_values_not_coerced(self):
        """int, float, str, bool, None result values must pass through unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            results = {
                "int_val": 42,
                "float_val": 3.14,
                "str_val": "hello",
                "bool_val": True,
            }
            sweep._write_csv_row(
                comm, 1, "pt_scalar", {"alpha": 1.0}, "success", results, None
            )

            rows = _read_csv_rows(csv_path)
            row = rows[0]
            # CSV values are always strings when read back, but they should
            # match the str() representation of the original scalar values
            self.assertEqual(row["int_val"], "42")
            self.assertEqual(row["float_val"], "3.14")
            self.assertEqual(row["str_val"], "hello")
            self.assertEqual(row["bool_val"], "True")


# ---------------------------------------------------------------------------
# Test 6: error column appears on failure, not on success
# ---------------------------------------------------------------------------


class TestErrorColumn(unittest.TestCase):
    def test_error_column_present_on_failure(self):
        """When status='failed' and error_msg is set, the error column must appear."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            error_message = "RuntimeError: something went wrong"
            sweep._write_csv_row(
                comm, 1, "pt_fail", {"alpha": 1.0}, "failed", {}, error_message
            )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["status"], "failed")
            self.assertIn("error", row)
            self.assertEqual(row["error"], error_message)

    def test_error_column_absent_on_success(self):
        """When status='success' and error_msg is None, no error column must appear."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            sweep._write_csv_row(
                comm, 1, "pt_ok", {"alpha": 1.0}, "success", {"lift": 5.0}, None
            )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["status"], "success")
            self.assertNotIn("error", row)

    def test_failure_row_has_empty_results(self):
        """On failure the results dict is empty; no stale result columns from previous row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            # First write a success with result columns
            sweep._write_csv_row(
                comm, 1, "pt_s", {"alpha": 1.0}, "success", {"lift": 9.9}, None
            )
            # Then write a failure with no result columns
            sweep._write_csv_row(
                comm, 2, "pt_f", {"alpha": 2.0}, "failed", {}, "some error"
            )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), 2)
            # The failure row will have the 'lift' column but it should be empty
            fail_row = rows[1]
            self.assertEqual(fail_row["status"], "failed")
            # lift column exists in the CSV (because of the first row's header)
            # but the value for the failure row should be empty string
            self.assertEqual(fail_row.get("lift", ""), "")


# ---------------------------------------------------------------------------
# Test 7: Results from multiple points are all appended correctly
# ---------------------------------------------------------------------------


class TestMultiplePointsAppended(unittest.TestCase):
    def test_all_points_appended_in_order(self):
        """Multiple writes must produce rows in insertion order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            points = [
                (1, "pt_001", {"alpha": 1.0}, "success", {"lift": 1.1}, None),
                (2, "pt_002", {"alpha": 2.0}, "success", {"lift": 2.2}, None),
                (3, "pt_003", {"alpha": 3.0}, "failed", {}, "err3"),
                (4, "pt_004", {"alpha": 4.0}, "success", {"lift": 4.4}, None),
            ]

            for case_idx, key, dp, status, results, err in points:
                sweep._write_csv_row(comm, case_idx, key, dp, status, results, err)

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), 4)

            for i, (case_idx, key, dp, status, results, err) in enumerate(points):
                self.assertEqual(rows[i]["case"], str(case_idx))
                self.assertEqual(rows[i]["key"], key)
                self.assertEqual(rows[i]["status"], status)

    def test_row_count_equals_write_count(self):
        """After N writes there must be exactly N data rows (plus one header)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            n = 5
            for i in range(1, n + 1):
                sweep._write_csv_row(
                    comm,
                    i,
                    f"pt_{i:03d}",
                    {"alpha": float(i)},
                    "success",
                    {"result": float(i) * 2.0},
                    None,
                )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), n)

            # Verify header appears exactly once
            header_count = _count_header_lines(csv_path)
            self.assertEqual(header_count, 1)

    def test_mixed_success_and_failure_rows_all_present(self):
        """Both success and failure rows must be written and retrievable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            sweep = _make_sweep(csv_path)
            comm = _make_mock_comm(rank=0)

            # Write failure FIRST so error column is in the original header.
            # When error is already in the header, it persists for all subsequent rows.
            sweep._write_csv_row(
                comm, 1, "pt_f1", {"alpha": 1.0}, "failed", {}, "build error"
            )
            sweep._write_csv_row(
                comm, 2, "pt_s1", {"alpha": 2.0}, "success", {"lift": 9.9}, None
            )
            sweep._write_csv_row(
                comm, 3, "pt_s2", {"alpha": 3.0}, "success", {"lift": 8.8}, None
            )

            rows = _read_csv_rows(csv_path)
            self.assertEqual(len(rows), 3)

            statuses = [row["status"] for row in rows]
            self.assertEqual(statuses, ["failed", "success", "success"])

            error_row = rows[0]
            self.assertIn("error", error_row)
            self.assertEqual(error_row["error"], "build error")

            # Success rows should have empty string for the error column
            self.assertEqual(rows[1].get("error", ""), "")


if __name__ == "__main__":
    unittest.main()
