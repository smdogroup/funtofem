"""
aerothermal_coupling_monitor.py
--------------------------------
Utility for recording per-coupling-step statistics of the aerothermal
quantities (wall temperature, thermal conductivity, heat flux) during a
FUNtoFEM aerothermal or aerothermoelastic solve.

Typical usage
-------------
In the run script, attach a monitor to the scenario before solving::

    from funtofem.interface.utils import AerothermalCouplingMonitor

    monitor = AerothermalCouplingMonitor(
        scenario, comm, csv_file="aerothermal_history.csv"
    )

Then pass it into the Fun3d14Interface constructor::

    solvers.flow = Fun3d14Interface(
        ...,
        aerothermal_monitor=monitor,
    )

After the solve::

    # Re-load and inspect
    df = AerothermalCouplingMonitor.from_csv("aerothermal_history.csv")
    print(df)

The CSV has one row per coupling step with columns::

    step, T_min, T_max, T_mean, k_min, k_max, k_mean, q_min, q_max, q_mean
"""

__all__ = ["AerothermalCouplingMonitor"]

import csv, os
import numpy as np


class AerothermalCouplingMonitor:
    """
    Records per-coupling-step statistics of aero surface temperatures,
    thermal conductivity, and heat flux during a FUNtoFEM aerothermal solve.

    Parameters
    ----------
    scenario : :class:`~funtofem.model.scenario.Scenario`
        The scenario being monitored.  Used for labelling only.
    comm : MPI communicator
        Only rank 0 writes files; other ranks still participate in the
        allreduce used to compute global min/max/mean across MPI partitions.
    csv_file : str or path-like, optional
        Path to the output CSV file.  Opened fresh (write mode) when the
        monitor is created so that re-runs produce a clean file.  If None,
        data are accumulated in memory only and can be retrieved via
        ``get_data()``.
    print_each_step : bool
        If True (default), print a one-line summary to stdout at each step
        (rank 0 only).
    """

    # Column names in the CSV / data dict
    COLUMNS = [
        "step",
        "T_min",
        "T_max",
        "T_mean",
        "k_min",
        "k_max",
        "k_mean",
        "q_min",
        "q_max",
        "q_mean",
    ]

    def __init__(self, scenario, comm, csv_file=None, print_each_step=True):
        self.scenario_name = scenario.name
        self.comm = comm
        self.csv_file = os.path.abspath(csv_file) if csv_file is not None else None
        self.print_each_step = print_each_step

        # In-memory list of dicts, one per step
        self._rows = []

        # Open (and write header to) the CSV on rank 0
        if self.comm.rank == 0 and self.csv_file is not None:
            with open(self.csv_file, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
                writer.writeheader()
            print(
                f"[AerothermalCouplingMonitor] writing to: {os.path.abspath(self.csv_file)}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, step, aero_temps, k_dim, heat_flux):
        """
        Record statistics for one coupling step.

        Call this once per coupling step, after ``k_dim`` and ``heat_flux``
        have been computed and before they are overwritten.  The method is
        safe to call on all MPI ranks; statistics are reduced to rank 0
        before writing.

        Parameters
        ----------
        step : int
            Coupling step index (1-based, matching the NLBGS loop counter).
        aero_temps : np.ndarray
            Aero surface temperatures at the current step (K).
        k_dim : np.ndarray
            Dimensional thermal conductivity at each aero node (W/m-K).
        heat_flux : np.ndarray
            Heating rate at each aero node (W, area-weighted).
        """
        if self.comm.rank == 0 and step == 1:
            print(
                f"[AerothermalCouplingMonitor] csv: {self.csv_file}",
                flush=True,
            )
        row = self._reduce_stats(step, aero_temps, k_dim, heat_flux)
        if row is None:
            return  # non-rank-0 processes return here

        self._rows.append(row)

        if self.print_each_step:
            print(
                f"[monitor {self.scenario_name}] step {step:4d} | "
                f"T: [{row['T_min']:.4g}, {row['T_max']:.4g}] K | "
                f"k: [{row['k_min']:.4g}, {row['k_max']:.4g}] W/m-K | "
                f"q: [{row['q_min']:.4g}, {row['q_max']:.4g}] W",
                flush=True,
            )

        if self.csv_file is not None:
            with open(self.csv_file, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
                writer.writerow(row)

    def get_data(self):
        """
        Return the recorded history as a dict of 1-D numpy arrays.

        Keys match the column names: ``step``, ``T_min``, ``T_max``,
        ``T_mean``, ``k_min``, ``k_max``, ``k_mean``, ``q_min``,
        ``q_max``, ``q_mean``.

        Returns an empty dict if no steps have been recorded yet or if
        called on a non-rank-0 process.
        """
        if not self._rows:
            return {}
        return {col: np.array([r[col] for r in self._rows]) for col in self.COLUMNS}

    def write_text_summary(self, filename):
        """
        Write a human-readable table of the recorded history to *filename*.

        Only meaningful on rank 0.

        Parameters
        ----------
        filename : str or path-like
        """
        if self.comm.rank != 0:
            return
        data = self.get_data()
        if not data:
            return

        header = (
            f"{'step':>6s}  "
            f"{'T_min':>12s}  {'T_max':>12s}  {'T_mean':>12s}  "
            f"{'k_min':>10s}  {'k_max':>10s}  {'k_mean':>10s}  "
            f"{'q_min':>14s}  {'q_max':>14s}  {'q_mean':>14s}"
        )
        divider = "-" * len(header)

        with open(filename, "w") as fh:
            fh.write(f"Aerothermal coupling history — scenario: {self.scenario_name}\n")
            fh.write(divider + "\n")
            fh.write(header + "\n")
            fh.write(divider + "\n")
            for row in self._rows:
                fh.write(
                    f"{int(row['step']):>6d}  "
                    f"{row['T_min']:>12.4g}  {row['T_max']:>12.4g}  {row['T_mean']:>12.4g}  "
                    f"{row['k_min']:>10.4g}  {row['k_max']:>10.4g}  {row['k_mean']:>10.4g}  "
                    f"{row['q_min']:>14.4g}  {row['q_max']:>14.4g}  {row['q_mean']:>14.4g}\n"
                )
            fh.write(divider + "\n")

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, csv_file):
        """
        Load a previously saved CSV back into a dict of numpy arrays.

        This is a lightweight loader — it does not require an MPI
        communicator and is suitable for post-processing scripts.

        Parameters
        ----------
        csv_file : str or path-like
            Path to the CSV file written by a monitor instance.

        Returns
        -------
        data : dict
            Dict mapping column name → 1-D numpy array.

        Examples
        --------
        ::

            data = AerothermalCouplingMonitor.from_csv("aerothermal_history.csv")
            import matplotlib.pyplot as plt
            plt.plot(data["step"], data["T_max"], label="T_wall max")
            plt.plot(data["step"], data["T_min"], label="T_wall min")
            plt.xlabel("Coupling step")
            plt.ylabel("Temperature (K)")
            plt.legend()
            plt.show()
        """
        rows = {col: [] for col in cls.COLUMNS}
        with open(csv_file, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for col in cls.COLUMNS:
                    rows[col].append(float(row[col]))
        return {col: np.array(rows[col]) for col in cls.COLUMNS}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reduce_stats(self, step, aero_temps, k_dim, heat_flux):
        """
        Compute global (across MPI ranks) min, max, mean for each field.

        Returns a row dict on rank 0, None on other ranks.
        """
        from mpi4py import MPI

        def _stats(arr):
            # local contributions (guard against empty local arrays)
            if arr is None or arr.size == 0:
                local_min = np.inf
                local_max = -np.inf
                local_sum = 0.0
                local_n = 0
            else:
                arr_real = np.real(arr)
                local_min = float(arr_real.min())
                local_max = float(arr_real.max())
                local_sum = float(arr_real.sum())
                local_n = arr_real.size

            global_min = self.comm.allreduce(local_min, op=MPI.MIN)
            global_max = self.comm.allreduce(local_max, op=MPI.MAX)
            global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
            global_n = self.comm.allreduce(local_n, op=MPI.SUM)
            global_mean = global_sum / global_n if global_n > 0 else float("nan")
            return global_min, global_max, global_mean

        T_min, T_max, T_mean = _stats(aero_temps)
        k_min, k_max, k_mean = _stats(k_dim)
        q_min, q_max, q_mean = _stats(heat_flux)

        if self.comm.rank != 0:
            return None

        return {
            "step": step,
            "T_min": T_min,
            "T_max": T_max,
            "T_mean": T_mean,
            "k_min": k_min,
            "k_max": k_max,
            "k_mean": k_mean,
            "q_min": q_min,
            "q_max": q_max,
            "q_mean": q_mean,
        }
