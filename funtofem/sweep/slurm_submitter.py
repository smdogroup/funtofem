"""
funtofem.sweep.slurm_submitter — SLURM job submission for parameter sweeps.

Provides SlurmSweepSubmitter, which writes a point.json file for each design
point and submits a per-point SLURM batch job via sbatch.  Supports dry-run
mode and skip-completed logic.
"""

import csv
import datetime
import json
import os
import subprocess

from .parameter_sweep import _make_key


class SlurmSweepSubmitter:
    """Submit one SLURM batch job per design point in a :class:`ParameterSweep`.

    .. warning::

        **Experimental.** ``funtofem.sweep`` is under active development. Its
        API may change in a backwards-incompatible way without a deprecation
        cycle.

    For each design point the submitter:

    1. Writes ``point.json`` to ``<cfd_root>/<key>/point.json``.
    2. Builds a SLURM batch script that passes ``--point-file`` to the user's
       sweep script.
    3. Submits the script via ``sbatch`` (or prints it when ``dry_run=True``).
    4. Writes a submission summary text file alongside the result CSV.

    Parameters
    ----------
    sweep : ParameterSweep
        Configured :class:`ParameterSweep` instance.  Used to enumerate design
        points and to locate the result CSV and ``cfd_root``.  The point file is
        written one level above the CFD output directory, so with the default
        roots it sits alongside the ``cfd/`` and ``struct/`` directories for that
        point.  When ``cfd_root`` and ``struct_root`` differ, ``point.json``
        follows ``cfd_root``.
    slurm_config : dict
        SLURM scheduler options.  Required keys: ``account``, ``partition``,
        ``qos``, ``nodes``, ``ntasks_per_node``, ``walltime``.
    sweep_script : str
        Path to the user's Python sweep script that accepts ``--point-file``.
    preamble : str
        Shell lines inserted before the ``srun`` invocation (module loads,
        environment setup, etc.).  Defaults to an empty string.
    log_dir : str
        Directory name used for per-job stdout/stderr files.  Defaults to
        ``"sweep_logs"``.  The submitter does **not** create this directory;
        SLURM creates it when the job starts, or the user must pre-create it.
    extra_directives : list[str] or None
        Additional ``#SBATCH`` directives injected verbatim into each script
        (e.g. ``["--mail-type=END", "--mail-user=you@example.com"]``).
        Defaults to an empty list when ``None``.

    Examples
    --------
    >>> submitter = SlurmSweepSubmitter(
    ...     sweep,
    ...     slurm_config={
    ...         "account": "MY_ACCOUNT",
    ...         "partition": "general",
    ...         "qos": "standard",
    ...         "nodes": 1,
    ...         "ntasks_per_node": 128,
    ...         "walltime": "08:00:00",
    ...     },
    ...     sweep_script="sweep.py",
    ...     preamble="source $HOME/.bashrc",
    ...     log_dir="sweep_logs",
    ...     extra_directives=["--mail-type=END"],
    ... )
    >>> submitter.submit(dry_run=True)
    """

    def __init__(
        self,
        sweep,
        slurm_config: dict,
        *,
        sweep_script: str,
        preamble: str = "",
        log_dir: str = "sweep_logs",
        extra_directives: "list[str] | None" = None,
    ) -> None:
        self.sweep = sweep
        self.slurm_config = slurm_config
        self.sweep_script = sweep_script
        self.preamble = preamble
        self.log_dir = log_dir
        self.extra_directives = extra_directives if extra_directives is not None else []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        dry_run: bool = False,
        skip_completed: bool = False,
    ) -> None:
        """Submit (or preview) one SLURM job per design point.

        Parameters
        ----------
        dry_run : bool
            When ``True``, print each job script to stdout without calling
            ``sbatch``.  ``point.json`` files are still written.
        skip_completed : bool
            When ``True``, read the result CSV and skip design points that
            already have ``status = "success"``.
        """
        sweep = self.sweep
        cfg = self.slurm_config
        design_points = sweep.strategy.generate(sweep.params)
        n_total = len(design_points)

        # Create caps_subdir once here on the login node before any jobs are
        # submitted, so pyCAPS can create problem directories inside it without
        # racing across parallel jobs.
        if sweep.caps_subdir:
            os.makedirs(sweep.caps_subdir, exist_ok=True)

        # Build set of already-completed keys
        completed_keys: "set[str]" = set()
        if skip_completed:
            csv_path = sweep.output_csv
            if os.path.isfile(csv_path):
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("status") == "success":
                            completed_keys.add(row.get("key", ""))

        submission_results: "list[tuple[str, str, str]]" = []
        submitted = 0
        skipped = 0
        failed = 0

        for idx, design_point in enumerate(design_points):
            case_num = idx + 1
            key = _make_key(design_point, sweep.key_fn)

            if key in completed_keys:
                submission_results.append((key, "skipped", "already completed"))
                skipped += 1
                continue

            # 1. Write point.json
            point_dir = os.path.join(sweep.cfd_root, key)
            os.makedirs(point_dir, exist_ok=True)
            point_json_path = os.path.join(point_dir, "point.json")
            with open(point_json_path, "w") as f:
                json.dump(design_point, f)

            # 2. Build the job script
            script_str = self._build_job_script(key, point_json_path)

            if dry_run:
                submission_results.append((key, "dry_run", script_str))
            else:
                # 3. Submit via sbatch
                result = subprocess.run(
                    ["sbatch"],
                    input=script_str,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    submission_results.append((key, "submitted", job_id))
                    submitted += 1
                else:
                    reason = result.stderr.strip()
                    submission_results.append((key, "failed", reason))
                    failed += 1

        self._write_summary(
            submission_results,
            n_total=n_total,
            submitted=submitted,
            skipped=skipped,
            failed=failed,
            dry_run=dry_run,
            skip_completed=skip_completed,
            n_completed=len(completed_keys),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_job_script(self, key: str, point_json_path: str) -> str:
        """Build the SLURM batch script string for a single design point.

        Parameters
        ----------
        key : str
            Design key for the point (used as the SLURM job name and log stem).
        point_json_path : str
            Absolute or relative path to the ``point.json`` file for this point.

        Returns
        -------
        str
            Complete SLURM batch script as a string, ready to pipe into ``sbatch``.
        """
        cfg = self.slurm_config
        account = cfg["account"]
        partition = cfg["partition"]
        qos = cfg["qos"]
        nodes = cfg["nodes"]
        ntasks_per_node = cfg["ntasks_per_node"]
        walltime = cfg["walltime"]
        log_dir = self.log_dir

        # Extra directives — one #SBATCH line per entry
        extra_lines = "\n".join(
            f"#SBATCH {directive}" for directive in self.extra_directives
        )
        extra_block = f"\n{extra_lines}" if extra_lines else ""

        # Preamble block
        preamble_block = f"\n{self.preamble}\n" if self.preamble else "\n"

        script = (
            f"#!/bin/bash\n"
            f"#SBATCH -A {account}\n"
            f"#SBATCH --job-name={key}\n"
            f"#SBATCH -p {partition}\n"
            f"#SBATCH -q {qos}\n"
            f"#SBATCH --nodes={nodes}\n"
            f"#SBATCH --ntasks-per-node={ntasks_per_node}\n"
            f"#SBATCH --time={walltime}\n"
            f"#SBATCH --output={log_dir}/{key}.out\n"
            f"#SBATCH --error={log_dir}/{key}.err"
            f"{extra_block}"
            f"{preamble_block}"
            f"srun python {self.sweep_script} --point-file {point_json_path}\n"
        )
        return script

    def _write_summary(
        self,
        results: "list[tuple[str, str, str]]",
        *,
        n_total: int,
        submitted: int,
        skipped: int,
        failed: int,
        dry_run: bool,
        skip_completed: bool,
        n_completed: int,
    ) -> None:
        """Write the submission summary file and echo everything to stdout.

        The summary is appended to ``submission_summary.txt`` alongside the
        result CSV, so multiple submission runs accumulate in one place.  Each
        run is separated by a timestamped header and a footer with totals.

        Parameters
        ----------
        results : list[tuple[str, str, str]]
            Each element is ``(key, status, detail)`` where *status* is one of
            ``"submitted"``, ``"dry_run"``, ``"skipped"``, ``"failed"`` and
            *detail* is the SLURM job ID, batch script text, skip reason, or
            sbatch error message respectively.
        n_total, submitted, skipped, failed : int
            Counts used in the footer line.
        dry_run : bool
            Controls the footer wording and header flag line.
        skip_completed : bool
            When ``True``, the header notes how many points were skipped.
        n_completed : int
            Number of already-completed keys (printed when ``skip_completed``).
        """
        sweep = self.sweep
        cfg = self.slurm_config

        summary_dir = os.path.dirname(sweep.output_csv) or "."
        summary_path = os.path.join(summary_dir, "submission_summary.txt")

        # Generate the full key map for reference
        all_points = sweep.strategy.generate(sweep.params)
        all_keys = [_make_key(pt, sweep.key_fn) for pt in all_points]

        with open(summary_path, "a") as summary_file:

            def log(msg=""):
                print(msg)
                print(msg, file=summary_file)

            # --- Header ---
            nodes = cfg.get("nodes", "?")
            ntasks = cfg.get("ntasks_per_node", "?")
            nprocs = (
                nodes * ntasks
                if isinstance(nodes, int) and isinstance(ntasks, int)
                else "?"
            )
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log(f"\n{'='*60}")
            log(f"  Parameter sweep launcher  [{timestamp}]")
            log(f"  Total design points : {n_total}")
            for param_name, values in sweep.params.items():
                log(f"  {param_name:<20}: {values}")
            log(f"  Account             : {cfg.get('account', '?')}")
            log(
                f"  Partition / QOS     : {cfg.get('partition', '?')} / {cfg.get('qos', '?')}"
            )
            log(
                f"  Nodes / tasks       : {nodes} nodes x {ntasks} tasks = {nprocs} MPI ranks"
            )
            log(f"  Walltime            : {cfg.get('walltime', '?')}")
            log(f"  Sweep script        : {self.sweep_script}")
            log(f"  Output CSV          : {sweep.output_csv}")
            if skip_completed:
                log(f"  Skipping {n_completed} already-completed case(s).")
            if dry_run:
                log(f"  DRY RUN — no jobs will be submitted.")
            log(f"{'='*60}\n")

            # --- Key map ---
            log("Design point key mapping")
            log("-" * 60)
            for key, point in zip(all_keys, all_points):
                params_str = "  ".join(f"{k}={v}" for k, v in point.items())
                log(f"  {key}  {params_str}")
            log()

            # --- Per-job lines ---
            for idx, (key, status, detail) in enumerate(results):
                case_num = idx + 1
                if status == "skipped":
                    log(f"[{case_num}/{n_total}] SKIP       {key}  (already completed)")
                elif status == "submitted":
                    log(f"[{case_num}/{n_total}] SUBMITTED  {key}  → job {detail}")
                elif status == "failed":
                    log(f"[{case_num}/{n_total}] FAILED     {key}")
                    log(f"  sbatch stderr: {detail}")
                elif status == "dry_run":
                    log(f"[{case_num}/{n_total}] DRY RUN    {key}")
                    log("-" * 50)
                    log(detail)
                    log("-" * 50)

            # --- Footer ---
            if dry_run:
                log(
                    f"\nDry run complete. "
                    f"{n_total - skipped} script(s) previewed, {skipped} skipped.\n"
                )
            else:
                log(
                    f"\nDone. {submitted} submitted, {skipped} skipped, {failed} failed.\n"
                )

        print(f"[slurm] Submission summary written to: {summary_path}")
