"""
Tests for SlurmSweepSubmitter: dry-run script content, point.json creation,
skip_completed logic, and extra_directives injection.
"""

import csv
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout

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

CartesianStrategy = _sweep.CartesianStrategy
MeshMode = _sweep.MeshMode
ParameterSweep = _sweep.ParameterSweep
SlurmSweepSubmitter = _sweep.SlurmSweepSubmitter
_make_key = _sweep.parameter_sweep._make_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sweep(tmp_path, params=None, output_csv=None):
    """Return a minimal ParameterSweep with no callbacks registered."""
    if params is None:
        params = {"alpha": [1.0, 2.0], "beta": [10.0]}
    if output_csv is None:
        output_csv = str(tmp_path / "sweep_results.csv")
    return ParameterSweep(
        params,
        strategy=CartesianStrategy(),
        mesh_mode=MeshMode.NONE,
        cfd_root=str(tmp_path / "cfd"),
        struct_root=str(tmp_path / "struct"),
        output_csv=output_csv,
    )


_BASE_SLURM_CONFIG = {
    "account": "TEST_ACCOUNT",
    "partition": "test_partition",
    "qos": "test_qos",
    "nodes": 2,
    "ntasks_per_node": 64,
    "walltime": "04:00:00",
}


def _make_submitter(sweep, extra_directives=None, preamble="", log_dir="sweep_logs"):
    return SlurmSweepSubmitter(
        sweep,
        _BASE_SLURM_CONFIG,
        sweep_script="my_sweep.py",
        preamble=preamble,
        log_dir=log_dir,
        extra_directives=extra_directives,
    )


class _SubmitterTestCase(unittest.TestCase):
    """Base class giving each test its own scratch directory as self.tmp_path."""

    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.tmp_path = pathlib.Path(tmpdir.name)

    def _submit_dry_run(self, submitter, **kwargs):
        """Run submit(dry_run=True) with stdout captured; return the output."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            submitter.submit(dry_run=True, **kwargs)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Test: _build_job_script produces expected content
# ---------------------------------------------------------------------------


class TestBuildJobScript(_SubmitterTestCase):
    def test_slurm_headers_present(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        key = "pt_abc123def456"
        point_json = f"{sweep.cfd_root}/{key}/point.json"
        script = submitter._build_job_script(key, point_json)

        self.assertIn("#!/bin/bash", script)
        self.assertIn("#SBATCH -A TEST_ACCOUNT", script)
        self.assertIn(f"#SBATCH --job-name={key}", script)
        self.assertIn("#SBATCH -p test_partition", script)
        self.assertIn("#SBATCH -q test_qos", script)
        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("#SBATCH --ntasks-per-node=64", script)
        self.assertIn("#SBATCH --time=04:00:00", script)

    def test_log_paths_use_key(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep, log_dir="my_logs")
        key = "pt_testkey000001"
        script = submitter._build_job_script(key, "some/path/point.json")

        self.assertIn(f"#SBATCH --output=my_logs/{key}.out", script)
        self.assertIn(f"#SBATCH --error=my_logs/{key}.err", script)

    def test_srun_command_with_point_file(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        key = "pt_abc123def456"
        point_json_path = f"{sweep.cfd_root}/{key}/point.json"
        script = submitter._build_job_script(key, point_json_path)

        self.assertIn(f"srun python my_sweep.py --point-file {point_json_path}", script)

    def test_preamble_inserted_before_srun(self):
        sweep = _make_sweep(self.tmp_path)
        preamble = "source $HOME/.bashrc\nmodule load python"
        submitter = _make_submitter(sweep, preamble=preamble)
        key = "pt_abc123def456"
        script = submitter._build_job_script(key, "some/point.json")

        preamble_pos = script.find(preamble)
        srun_pos = script.find("srun python")
        self.assertNotEqual(preamble_pos, -1, "Preamble not found in script")
        self.assertNotEqual(srun_pos, -1, "srun line not found in script")
        self.assertLess(preamble_pos, srun_pos, "Preamble must appear before srun")

    def test_extra_directives_injected(self):
        sweep = _make_sweep(self.tmp_path)
        extra = ["--mail-type=END", "--mail-user=me@example.com"]
        submitter = _make_submitter(sweep, extra_directives=extra)
        key = "pt_abc123def456"
        script = submitter._build_job_script(key, "some/point.json")

        self.assertIn("#SBATCH --mail-type=END", script)
        self.assertIn("#SBATCH --mail-user=me@example.com", script)

    def test_no_extra_directives_by_default(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        key = "pt_abc123def456"
        script = submitter._build_job_script(key, "some/point.json")

        # Only standard directives should appear
        standard_prefixes = {
            "#SBATCH -A",
            "#SBATCH --job-name",
            "#SBATCH -p",
            "#SBATCH -q",
            "#SBATCH --nodes",
            "#SBATCH --ntasks-per-node",
            "#SBATCH --time",
            "#SBATCH --output",
            "#SBATCH --error",
        }
        sbatch_lines = [
            line for line in script.splitlines() if line.startswith("#SBATCH")
        ]
        for line in sbatch_lines:
            self.assertTrue(
                any(line.startswith(d) for d in standard_prefixes),
                f"Unexpected #SBATCH directive: {line}",
            )


# ---------------------------------------------------------------------------
# Test: dry_run=True prints scripts without calling sbatch
# ---------------------------------------------------------------------------


class TestDryRun(_SubmitterTestCase):
    def test_dry_run_prints_script(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        out = self._submit_dry_run(submitter)

        self.assertIn("srun python my_sweep.py --point-file", out)
        self.assertIn("#!/bin/bash", out)
        self.assertIn("DRY RUN", out)

    def test_dry_run_writes_point_json(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter)

        # Verify point.json files were created for all design points
        for alpha in [1.0, 2.0]:
            point = {"alpha": alpha, "beta": 10.0}
            key = _make_key(point)
            point_json = self.tmp_path / "cfd" / key / "point.json"
            self.assertTrue(point_json.exists(), f"Expected {point_json} to exist")
            loaded = json.loads(point_json.read_text())
            self.assertEqual(loaded["alpha"], alpha)
            self.assertEqual(loaded["beta"], 10.0)

    def test_dry_run_creates_summary_file(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter)

        summary_path = self.tmp_path / "submission_summary.txt"
        self.assertTrue(summary_path.exists())
        # The summary must record that nothing was actually submitted.
        self.assertIn("DRY RUN", summary_path.read_text())


# ---------------------------------------------------------------------------
# Test: point.json creation and content
# ---------------------------------------------------------------------------


class TestPointJsonCreation(_SubmitterTestCase):
    def test_point_json_correct_content(self):
        sweep = _make_sweep(self.tmp_path, params={"x": [0.5], "y": [1.5]})
        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter)

        point = {"x": 0.5, "y": 1.5}
        key = _make_key(point)
        point_json_path = self.tmp_path / "cfd" / key / "point.json"
        self.assertTrue(point_json_path.exists())
        self.assertEqual(json.loads(point_json_path.read_text()), point)

    def test_point_json_directory_created(self):
        sweep = _make_sweep(self.tmp_path, params={"z": [3.14]})
        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter)

        key = _make_key({"z": 3.14})
        point_dir = self.tmp_path / "cfd" / key
        self.assertTrue(point_dir.is_dir())

    def test_point_json_path_in_script(self):
        sweep = _make_sweep(self.tmp_path, params={"v": [7]})
        submitter = _make_submitter(sweep)

        key = _make_key({"v": 7})
        expected_path = os.path.join(sweep.cfd_root, key, "point.json")
        script = submitter._build_job_script(key, expected_path)
        self.assertIn(f"--point-file {expected_path}", script)

    def test_multiple_points_each_get_point_json(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter)

        for alpha in [1.0, 2.0]:
            key = _make_key({"alpha": alpha, "beta": 10.0})
            self.assertTrue((self.tmp_path / "cfd" / key / "point.json").exists())


# ---------------------------------------------------------------------------
# Test: skip_completed logic
# ---------------------------------------------------------------------------


class TestSkipCompleted(_SubmitterTestCase):
    def _write_csv_with_status(self, csv_path, rows):
        fieldnames = ["key", "status", "alpha", "beta"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_skip_completed_skips_success_rows(self):
        sweep = _make_sweep(self.tmp_path)
        completed_point = {"alpha": 1.0, "beta": 10.0}
        completed_key = _make_key(completed_point)
        self._write_csv_with_status(
            sweep.output_csv,
            [{"key": completed_key, "status": "success", "alpha": 1.0, "beta": 10.0}],
        )

        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter, skip_completed=True)

        summary_path = self.tmp_path / "submission_summary.txt"
        content = summary_path.read_text()
        # The completed key should appear as skipped
        self.assertIn(completed_key, content)
        self.assertIn("SKIP", content)

        # The other point (alpha=2.0) should be a dry run
        other_key = _make_key({"alpha": 2.0, "beta": 10.0})
        self.assertIn(other_key, content)
        self.assertIn("DRY RUN", content)

    def test_skip_completed_failed_rows_are_rerun(self):
        sweep = _make_sweep(self.tmp_path)
        failed_point = {"alpha": 1.0, "beta": 10.0}
        failed_key = _make_key(failed_point)
        self._write_csv_with_status(
            sweep.output_csv,
            [{"key": failed_key, "status": "failed", "alpha": 1.0, "beta": 10.0}],
        )

        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter, skip_completed=True)

        summary_path = self.tmp_path / "submission_summary.txt"
        content = summary_path.read_text()
        # Failed points should be submitted (DRY RUN), not skipped
        self.assertIn(failed_key, content)
        self.assertIn("DRY RUN", content)
        # Should not appear as skipped
        skip_lines = [
            l for l in content.splitlines() if "SKIP" in l and failed_key in l
        ]
        self.assertFalse(skip_lines, "Failed point should not be skipped")

    def test_skip_completed_no_csv_processes_all(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        # No CSV exists — all points should be processed
        self._submit_dry_run(submitter, skip_completed=True)

        summary_path = self.tmp_path / "submission_summary.txt"
        content = summary_path.read_text()
        # Both design points should appear as DRY RUN
        self.assertGreaterEqual(content.count("DRY RUN"), 2)

    def test_skip_completed_false_does_not_skip_success(self):
        sweep = _make_sweep(self.tmp_path)
        completed_point = {"alpha": 1.0, "beta": 10.0}
        completed_key = _make_key(completed_point)
        self._write_csv_with_status(
            sweep.output_csv,
            [{"key": completed_key, "status": "success", "alpha": 1.0, "beta": 10.0}],
        )

        submitter = _make_submitter(sweep)
        # skip_completed=False (default) — all points should be submitted
        self._submit_dry_run(submitter, skip_completed=False)

        summary_path = self.tmp_path / "submission_summary.txt"
        content = summary_path.read_text()
        # Both points should appear as DRY RUN (no skipping)
        self.assertGreaterEqual(content.count("DRY RUN"), 2)


# ---------------------------------------------------------------------------
# Test: extra_directives injection
# ---------------------------------------------------------------------------


class TestExtraDirectives(_SubmitterTestCase):
    def test_single_extra_directive(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep, extra_directives=["--mail-type=END"])
        key = "pt_testkey"
        script = submitter._build_job_script(key, "some/path.json")
        self.assertIn("#SBATCH --mail-type=END", script)

    def test_multiple_extra_directives_all_present(self):
        sweep = _make_sweep(self.tmp_path)
        directives = [
            "--mail-type=END",
            "--mail-user=user@example.com",
            "--constraint=haswell",
        ]
        submitter = _make_submitter(sweep, extra_directives=directives)
        key = "pt_testkey"
        script = submitter._build_job_script(key, "some/path.json")

        for directive in directives:
            self.assertIn(f"#SBATCH {directive}", script)

    def test_none_extra_directives_treated_as_empty(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = SlurmSweepSubmitter(
            sweep,
            {
                "account": "A",
                "partition": "p",
                "qos": "q",
                "nodes": 1,
                "ntasks_per_node": 1,
                "walltime": "1:00:00",
            },
            sweep_script="run.py",
            extra_directives=None,
        )
        self.assertEqual(submitter.extra_directives, [])

    def test_extra_directives_appear_before_srun(self):
        """Extra directives must appear before the srun line."""
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(
            sweep, preamble="echo start", extra_directives=["--gres=gpu:1"]
        )
        key = "pt_testkey"
        script = submitter._build_job_script(key, "some/path.json")

        extra_pos = script.find("#SBATCH --gres=gpu:1")
        srun_pos = script.find("srun python")

        self.assertNotEqual(extra_pos, -1, "Extra directive not found")
        self.assertLess(extra_pos, srun_pos, "Extra directive must appear before srun")


# ---------------------------------------------------------------------------
# Test: submission summary
# ---------------------------------------------------------------------------


class TestSubmissionSummary(_SubmitterTestCase):
    def test_summary_path_is_alongside_csv(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter)

        expected = self.tmp_path / "submission_summary.txt"
        self.assertTrue(expected.exists())

    def test_summary_contains_all_keys(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter)

        keys = [
            _make_key({"alpha": 1.0, "beta": 10.0}),
            _make_key({"alpha": 2.0, "beta": 10.0}),
        ]
        content = (self.tmp_path / "submission_summary.txt").read_text()
        for key in keys:
            self.assertIn(key, content)

    def test_summary_has_header_line(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = _make_submitter(sweep)
        self._submit_dry_run(submitter)

        content = (self.tmp_path / "submission_summary.txt").read_text()
        # Summary should contain a timestamped banner and design point info
        self.assertIn("Parameter sweep launcher", content)
        self.assertIn("Total design points", content)


# ---------------------------------------------------------------------------
# Test: constructor stores attributes correctly
# ---------------------------------------------------------------------------


class TestConstructor(_SubmitterTestCase):
    def test_attributes_stored(self):
        sweep = _make_sweep(self.tmp_path)
        slurm_config = {
            "account": "ACC",
            "partition": "part",
            "qos": "q",
            "nodes": 4,
            "ntasks_per_node": 32,
            "walltime": "02:00:00",
        }
        submitter = SlurmSweepSubmitter(
            sweep,
            slurm_config,
            sweep_script="/path/to/script.py",
            preamble="echo hello",
            log_dir="my_logs",
            extra_directives=["--x=y"],
        )
        self.assertIs(submitter.sweep, sweep)
        self.assertIs(submitter.slurm_config, slurm_config)
        self.assertEqual(submitter.sweep_script, "/path/to/script.py")
        self.assertEqual(submitter.preamble, "echo hello")
        self.assertEqual(submitter.log_dir, "my_logs")
        self.assertEqual(submitter.extra_directives, ["--x=y"])

    def test_default_values(self):
        sweep = _make_sweep(self.tmp_path)
        submitter = SlurmSweepSubmitter(
            sweep,
            {
                "account": "A",
                "partition": "p",
                "qos": "q",
                "nodes": 1,
                "ntasks_per_node": 1,
                "walltime": "1:00:00",
            },
            sweep_script="run.py",
        )
        self.assertEqual(submitter.preamble, "")
        self.assertEqual(submitter.log_dir, "sweep_logs")
        self.assertEqual(submitter.extra_directives, [])


if __name__ == "__main__":
    unittest.main()
