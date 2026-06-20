"""
aeroelastic_coupling_monitor.py
---------------------------------
Utility for recording per-coupling-step statistics of the aeroelastic
quantities (aero surface displacements and aerodynamic loads) during a
FUNtoFEM aeroelastic or aerothermoelastic solve.

Typical usage
-------------
In the run script, attach a monitor to the scenario before solving::

    from funtofem.interface.utils import AeroelasticCouplingMonitor

    monitor = AeroelasticCouplingMonitor(
        scenario, comm, csv_file="aeroelastic_history.csv"
    )

Then pass it into the Fun3d14Interface constructor::

    solvers.flow = Fun3d14Interface(
        ...,
        aeroelastic_monitor=monitor,
    )

After the solve::

    data = AeroelasticCouplingMonitor.from_csv("aeroelastic_history.csv")
    import matplotlib.pyplot as plt
    plt.plot(data["step"], data["disp_max"], label="max aero disp (m)")
    plt.plot(data["step"], data["load_max"], label="max aero load (N)")

The CSV has one row per coupling step with columns::

    step,
    disp_min, disp_max, disp_mean,   # magnitude of aero surface displacement vector
    load_min, load_max, load_mean    # magnitude of aerodynamic load vector
"""

__all__ = ["AeroelasticCouplingMonitor"]

import csv
import numpy as np


class AeroelasticCouplingMonitor:
    """
    Records per-coupling-step statistics of aero surface displacements and
    aerodynamic loads during a FUNtoFEM aeroelastic solve.

    Parameters
    ----------
    scenario : :class:`~funtofem.model.scenario.Scenario`
        The scenario being monitored.  Used for labelling only.
    comm : MPI communicator
        Only rank 0 writes files; statistics are reduced across all ranks
        via allreduce before writing.
    csv_file : str or path-like, optional
        Path to the output CSV file.  Created fresh on construction so that
        re-runs produce a clean file.  If None, data are accumulated in
        memory only and can be retrieved via ``get_data()``.
    print_each_step : bool
        If True (default), print a one-line summary to stdout at each step
        (rank 0 only).
    """

    COLUMNS = [
        "step",
        "disp_min", "disp_max", "disp_mean",
        "load_min", "load_max", "load_mean",
    ]

    def __init__(self, scenario, comm, csv_file=None, print_each_step=True):
        self.scenario_name = scenario.name
        self.comm = comm
        self.csv_file = csv_file
        self.print_each_step = print_each_step

        self._rows = []

        if self.comm.rank == 0 and self.csv_file is not None:
            with open(self.csv_file, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
                writer.writeheader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, step, aero_disps, aero_loads):
        """
        Record statistics for one coupling step.

        Safe to call on all MPI ranks; statistics are reduced to rank 0.
        Pass ``None`` for either array on ranks that own no aero nodes.

        Parameters
        ----------
        step : int
            Coupling step index (1-based).
        aero_disps : np.ndarray or None
            Aero surface displacement vector (x, y, z interleaved, metres).
            The per-node displacement magnitude is computed before reduction.
        aero_loads : np.ndarray or None
            Aerodynamic load vector (x, y, z interleaved, Newtons).
            The per-node load magnitude is computed before reduction.
        """
        row = self._reduce_stats(step, aero_disps, aero_loads)
        if row is None:
            return

        self._rows.append(row)

        if self.print_each_step:
            print(
                f"[monitor {self.scenario_name}] step {step:4d} | "
                f"disp: [{row['disp_min']:.4g}, {row['disp_max']:.4g}] m | "
                f"load: [{row['load_min']:.4g}, {row['load_max']:.4g}] N",
                flush=True,
            )

        if self.csv_file is not None:
            with open(self.csv_file, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
                writer.writerow(row)

    def get_data(self):
        """
        Return the recorded history as a dict of 1-D numpy arrays.

        Keys match the column names.  Returns an empty dict if no steps
        have been recorded yet or if called on a non-rank-0 process.
        """
        if not self._rows:
            return {}
        return {col: np.array([r[col] for r in self._rows]) for col in self.COLUMNS}

    def write_text_summary(self, filename):
        """
        Write a human-readable fixed-width table to *filename* (rank 0 only).

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
            f"{'disp_min':>14s}  {'disp_max':>14s}  {'disp_mean':>14s}  "
            f"{'load_min':>14s}  {'load_max':>14s}  {'load_mean':>14s}"
        )
        divider = "-" * len(header)

        with open(filename, "w") as fh:
            fh.write(
                f"Aeroelastic coupling history — scenario: {self.scenario_name}\n"
            )
            fh.write(divider + "\n")
            fh.write(header + "\n")
            fh.write(divider + "\n")
            for row in self._rows:
                fh.write(
                    f"{int(row['step']):>6d}  "
                    f"{row['disp_min']:>14.4g}  {row['disp_max']:>14.4g}  {row['disp_mean']:>14.4g}  "
                    f"{row['load_min']:>14.4g}  {row['load_max']:>14.4g}  {row['load_mean']:>14.4g}\n"
                )
            fh.write(divider + "\n")

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, csv_file):
        """
        Load a previously saved CSV back into a dict of numpy arrays.

        No MPI communicator required — suitable for post-processing scripts.

        Parameters
        ----------
        csv_file : str or path-like

        Returns
        -------
        data : dict
            Dict mapping column name → 1-D numpy array.

        Examples
        --------
        ::

            data = AeroelasticCouplingMonitor.from_csv("aeroelastic_history.csv")
            import matplotlib.pyplot as plt
            plt.plot(data["step"], data["disp_max"], label="max disp (m)")
            plt.plot(data["step"], data["load_max"], label="max load (N)")
            plt.xlabel("Coupling step")
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

    def _reduce_stats(self, step, aero_disps, aero_loads):
        """
        Compute global min/max/mean of per-node vector magnitudes.

        Returns a row dict on rank 0, None on other ranks.
        """
        from mpi4py import MPI

        def _vector_magnitude_stats(vec):
            """Compute per-node magnitudes from an interleaved (x,y,z) vector,
            then reduce min/max/mean across MPI ranks."""
            if vec is None or vec.size == 0:
                local_min = np.inf
                local_max = -np.inf
                local_sum = 0.0
                local_n = 0
            else:
                vec_real = np.real(vec)
                # reshape to (nnodes, 3) and compute per-node magnitude
                mags = np.linalg.norm(vec_real.reshape(-1, 3), axis=1)
                local_min = float(mags.min())
                local_max = float(mags.max())
                local_sum = float(mags.sum())
                local_n = mags.size

            global_min = self.comm.allreduce(local_min, op=MPI.MIN)
            global_max = self.comm.allreduce(local_max, op=MPI.MAX)
            global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
            global_n = self.comm.allreduce(local_n, op=MPI.SUM)
            global_mean = global_sum / global_n if global_n > 0 else float("nan")
            return global_min, global_max, global_mean

        d_min, d_max, d_mean = _vector_magnitude_stats(aero_disps)
        l_min, l_max, l_mean = _vector_magnitude_stats(aero_loads)

        if self.comm.rank != 0:
            return None

        return {
            "step":      step,
            "disp_min":  d_min,  "disp_max":  d_max,  "disp_mean": d_mean,
            "load_min":  l_min,  "load_max":  l_max,  "load_mean": l_mean,
        }
