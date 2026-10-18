"""
funtofem.sweep.parameter_sweep — ParameterSweep orchestration class.

Central class that drives the parameter sweep: enumerates design points via a
SweepStrategy, manages per-point output directories, dispatches mesh callbacks
according to MeshMode, invokes model/solver builders and result extractors,
writes incremental CSV output, and supports resume logic.
"""

import csv
import hashlib
import inspect
import json
import os
import warnings
from typing import Callable

from .strategy import SweepStrategy, CartesianStrategy
from .mesh_mode import MeshMode


def _make_key(point: dict, key_fn=None) -> str:
    """Return a deterministic, filesystem-safe design key for a design point.

    When *key_fn* is provided it is called with *point* and its return value is
    used as-is, allowing callers to supply human-readable keys for small sweeps.

    The default (no *key_fn*) produces a fixed 15-character string of the form
    ``pt_xxxxxxxxxxxx`` (the prefix ``pt_`` followed by the first 12 hex
    characters of the SHA-256 digest of the JSON-serialised point).
    ``json.dumps`` is called with ``sort_keys=True`` so the result is
    independent of dict iteration order.  Only ``hashlib`` and ``json`` from
    the standard library are required.

    Parameters
    ----------
    point:
        Design point dict mapping parameter names to their values.
    key_fn:
        Optional callable ``(point: dict) -> str``.  When not ``None`` its
        return value is used directly as the key.

    Returns
    -------
    str
        A filesystem-safe string that uniquely identifies *point*.  The default
        key is always exactly 15 characters long (``pt_`` + 12 hex chars).
    """
    if key_fn is not None:
        return key_fn(point)
    canonical = json.dumps(point, sort_keys=True)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"pt_{digest}"


def _default_result_extractor(
    model, driver, design_point, cfd_output_dir, struct_output_dir
):
    """Built-in default result extractor used when no Result_Extractor_Callback is registered.

    Collects:
    1. All function values from model.scenarios[*].functions[*].value
    2. Flow residual norm (L2 norm of solvers.flow._forward_resid) as flow_resid_norm
    3. Per-component residuals as flow_resid_comps (space-separated string)
    4. Forward iteration count from solvers.flow._last_forward_step as flow_iters

    All flow fields degrade gracefully to nan/"" when no flow solver is present
    (e.g. STRUCT_ONLY or NONE mesh modes).

    Parameters
    ----------
    model : FUNtoFEMmodel
        The FUNtoFEM model returned by the model builder.
    driver : FUNtoFEMnlbgs or None
        The driver used to run the solve, or None when a custom sweep_driver was used.
    design_point : dict
        Dict mapping parameter names to values for the current point.
    cfd_output_dir : str or None
        Per-point CFD output directory path.
    struct_output_dir : str or None
        Per-point structural output directory path.

    Returns
    -------
    dict[str, float | str]
        Dict of scalar results ready for CSV writing.
    """
    import numpy as np

    results = {}

    # 1. Function values from all scenarios
    if model is not None:
        for scenario in getattr(model, "scenarios", []):
            for func in getattr(scenario, "functions", []):
                results[func.name] = func.value

    # 2. Flow solver diagnostics — degrade gracefully when absent
    solvers = getattr(driver, "solvers", None)
    flow = getattr(solvers, "flow", None)

    fwd_vec = getattr(flow, "_forward_resid", None)
    if fwd_vec is not None:
        arr = np.asarray(fwd_vec).ravel()
        results["flow_resid_norm"] = float(np.linalg.norm(arr))
        results["flow_resid_comps"] = " ".join(f"{r:.6e}" for r in arr)
    else:
        results["flow_resid_norm"] = float("nan")
        results["flow_resid_comps"] = ""

    last_step = getattr(flow, "_last_forward_step", None)
    results["flow_iters"] = int(last_step) if last_step is not None else float("nan")

    return results


def _make_caps_problem_name(caps_subdir: "str | None", key: str) -> str:
    """Return the CAPS problem name for a design point.

    The CAPS problem name is the string passed to mesh callbacks as the
    ``caps_problem_name`` argument.  It is constructed from an optional
    sub-directory prefix and the design key.

    Parameters
    ----------
    caps_subdir:
        Optional sub-directory component.  When this is a non-empty string the
        result is ``"<caps_subdir>/<key>"``.  When it is ``None`` or an empty
        string the bare *key* is returned with no prefix or separator.
    key:
        The design key string (e.g. ``"pt_a8f3c2d14e91"``).

    Returns
    -------
    str
        ``f"{caps_subdir}/{key}"`` when *caps_subdir* is a non-empty string,
        otherwise the bare *key*.
    """
    if caps_subdir:
        return f"{caps_subdir}/{key}"
    return key


def _count_positional_params(callback) -> "int | None":
    """Return the number of positional parameters for *callback*.

    Returns ``None`` when the callback uses ``*args`` or ``**kwargs``, which
    signals that the arity check should be skipped.

    Parameters
    ----------
    callback:
        Any callable whose signature can be inspected via ``inspect.signature``.

    Returns
    -------
    int or None
        The count of positional (non-VAR_POSITIONAL, non-VAR_KEYWORD,
        non-KEYWORD_ONLY) parameters, or ``None`` if the signature contains
        ``*args`` or ``**kwargs``.
    """
    sig = inspect.signature(callback)
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return None
    # Count positional-or-keyword and positional-only parameters
    positional_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    return sum(1 for p in sig.parameters.values() if p.kind in positional_kinds)


def _validate_callback_arity(callback, expected: int, name: str) -> None:
    """Validate that *callback* has exactly *expected* positional parameters.

    Skips the check when the callback uses ``*args`` or ``**kwargs``.

    Parameters
    ----------
    callback:
        The callable to inspect.
    expected:
        The required number of positional parameters.
    name:
        Human-readable name of the callback (used in the error message).

    Raises
    ------
    TypeError
        If the callback has a different number of positional parameters than
        *expected* (and does not use ``*args`` / ``**kwargs``).
    """
    count = _count_positional_params(callback)
    if count is None:
        # *args or **kwargs present — skip arity check
        return
    if count != expected:
        raise TypeError(
            f"{name} must accept exactly {expected} positional parameter(s), "
            f"but the provided callback has {count}. "
            f"Check the expected signature in the ParameterSweep documentation."
        )


class ParameterSweep:
    """Callback-driven parameter sweep orchestration for FUNtoFEM.

    .. warning::

        **Experimental.** ``funtofem.sweep`` is under active development. Its
        API may change in a backwards-incompatible way without a deprecation
        cycle.

    :class:`ParameterSweep` accepts a parameter space and a set of user-supplied
    callbacks, then drives the full sweep lifecycle: design-point enumeration,
    per-point directory isolation, optional mesh regeneration, model and solver
    construction, forward/adjoint solves, result extraction, and incremental CSV
    output.

    Parameters
    ----------
    params:
        Parameter space mapping.  Each key is a parameter name; each value is a
        list of values to sweep.  The :class:`~funtofem.sweep.SweepStrategy`
        converts this into an ordered list of design points.
    strategy:
        Sweep strategy used to enumerate design points from *params*.  Defaults
        to :class:`~funtofem.sweep.CartesianStrategy` when ``None``.
    mesh_mode:
        Controls which mesh-generation callbacks are invoked per design point.
        Defaults to :attr:`~funtofem.sweep.MeshMode.FULL_REGEN`.
    caps_subdir:
        Sub-directory prefix used to build the CAPS problem name passed to mesh
        callbacks.  A non-empty string ``s`` produces ``"<s>/<key>"``; ``None``
        or ``""`` produces the bare key.  Defaults to ``"caps"``.
    cfd_root:
        Root directory for per-point CFD output.  Each design point writes to
        ``"<cfd_root>/<key>/cfd"``.  Defaults to ``"sweep_output"``.
    struct_root:
        Root directory for per-point structural output.  Each design point
        writes to ``"<struct_root>/<key>/struct"``.  Defaults to
        ``"sweep_output"``.

        ``cfd_root`` and ``struct_root`` share the same ``<root>/<key>/<kind>``
        layout, so leaving both at the default collects everything for a design
        point under a single directory::

            sweep_output/<key>/cfd/
            sweep_output/<key>/struct/

        Setting them to different roots splits the two trees instead — useful
        when CFD output is large enough to belong on scratch while structural
        results stay on a backed-up filesystem.  Note that
        :class:`~funtofem.sweep.SlurmSweepSubmitter` writes each point's
        ``point.json`` to ``"<cfd_root>/<key>/point.json"``, so when the roots
        differ the point file follows ``cfd_root``.
    output_csv:
        Path to the result CSV file written (and appended to) during the sweep.
        Defaults to ``"sweep_results.csv"``.
    key_fn:
        Optional callable ``(point: dict) -> str`` that replaces the default
        SHA-256-based key generation.  When ``None`` the default hash-based key
        is used.

    Examples
    --------
    >>> sweep = (
    ...     ParameterSweep({"alpha": [1, 2], "beta": [10, 20]})
    ...     .set_model_builder(my_model_fn)
    ...     .set_solver_builder(my_solver_fn)
    ...     .set_result_extractor(my_extractor_fn)
    ... )
    >>> sweep.run(comm)
    """

    def __init__(
        self,
        params: dict[str, list],
        *,
        strategy: "SweepStrategy | None" = None,
        mesh_mode: MeshMode = MeshMode.FULL_REGEN,
        caps_subdir: "str | None" = "caps",
        cfd_root: str = "sweep_output",
        struct_root: str = "sweep_output",
        output_csv: str = "sweep_results.csv",
        key_fn: "Callable[[dict], str] | None" = None,
    ) -> None:
        self.params = params
        self.strategy = strategy if strategy is not None else CartesianStrategy()
        self.mesh_mode = mesh_mode
        self.caps_subdir = caps_subdir
        self.cfd_root = cfd_root
        self.struct_root = struct_root
        self.output_csv = output_csv
        self.key_fn = key_fn

        # Callback attributes — all start as None
        self.cfd_mesh_callback = None
        self.struct_mesh_callback = None
        self.model_builder = None
        self.solver_builder = None
        self.result_extractor = None
        self.sweep_driver = None
        self.transfer_settings = None

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def set_cfd_mesh_callback(self, callback) -> "ParameterSweep":
        """Register the CFD mesh generation callback.

        The callback must accept exactly 4 positional parameters:
        ``(comm, design_point, caps_problem_name, cfd_output_dir)``.

        Parameters
        ----------
        callback:
            Callable with the expected signature.

        Returns
        -------
        ParameterSweep
            *self*, to allow method chaining.

        Raises
        ------
        TypeError
            If the callback does not have exactly 4 positional parameters
            (and does not use ``*args`` / ``**kwargs``).
        """
        _validate_callback_arity(callback, expected=4, name="cfd_mesh_callback")
        self.cfd_mesh_callback = callback
        return self

    def set_struct_mesh_callback(self, callback) -> "ParameterSweep":
        """Register the structural mesh generation callback.

        The callback must accept exactly 4 positional parameters:
        ``(comm, design_point, caps_problem_name, struct_output_dir)``.

        Parameters
        ----------
        callback:
            Callable with the expected signature.

        Returns
        -------
        ParameterSweep
            *self*, to allow method chaining.

        Raises
        ------
        TypeError
            If the callback does not have exactly 4 positional parameters
            (and does not use ``*args`` / ``**kwargs``).
        """
        _validate_callback_arity(callback, expected=4, name="struct_mesh_callback")
        self.struct_mesh_callback = callback
        return self

    def set_model_builder(self, callback) -> "ParameterSweep":
        """Register the model builder callback.

        The callback must accept exactly 4 positional parameters:
        ``(comm, design_point, cfd_output_dir, struct_output_dir)``.

        Parameters
        ----------
        callback:
            Callable with the expected signature.

        Returns
        -------
        ParameterSweep
            *self*, to allow method chaining.

        Raises
        ------
        TypeError
            If the callback does not have exactly 4 positional parameters
            (and does not use ``*args`` / ``**kwargs``).
        """
        _validate_callback_arity(callback, expected=4, name="model_builder")
        self.model_builder = callback
        return self

    def set_solver_builder(self, callback) -> "ParameterSweep":
        """Register the solver builder callback.

        The callback must accept exactly 5 positional parameters:
        ``(comm, model, design_point, cfd_output_dir, struct_output_dir)``.

        Parameters
        ----------
        callback:
            Callable with the expected signature.

        Returns
        -------
        ParameterSweep
            *self*, to allow method chaining.

        Raises
        ------
        TypeError
            If the callback does not have exactly 5 positional parameters
            (and does not use ``*args`` / ``**kwargs``).
        """
        _validate_callback_arity(callback, expected=5, name="solver_builder")
        self.solver_builder = callback
        return self

    def set_result_extractor(self, callback) -> "ParameterSweep":
        """Register the result extractor callback.

        The callback must accept exactly 5 positional parameters:
        ``(model, driver, design_point, cfd_output_dir, struct_output_dir)``.

        Parameters
        ----------
        callback:
            Callable with the expected signature.

        Returns
        -------
        ParameterSweep
            *self*, to allow method chaining.

        Raises
        ------
        TypeError
            If the callback does not have exactly 5 positional parameters
            (and does not use ``*args`` / ``**kwargs``).
        """
        _validate_callback_arity(callback, expected=5, name="result_extractor")
        self.result_extractor = callback
        return self

    def set_sweep_driver(self, callback) -> "ParameterSweep":
        """Register an optional custom sweep driver callback.

        When set, this callback replaces the default ``FUNtoFEMnlbgs``
        forward/adjoint path.  The callback must accept exactly 5 positional
        parameters:
        ``(comm, model, solvers, design_point, compute_adjoint)``.

        The callback may optionally return the driver object it constructs.
        If it does, the driver is passed through to the result extractor as the
        ``driver`` argument, allowing the extractor to read residuals or other
        solver state.  If the callback returns ``None`` (or has no return
        statement), the extractor receives ``None`` for the driver argument —
        the built-in default extractor handles this gracefully.

        Parameters
        ----------
        callback:
            Callable with the expected signature.

        Returns
        -------
        ParameterSweep
            *self*, to allow method chaining.

        Raises
        ------
        TypeError
            If the callback does not have exactly 5 positional parameters
            (and does not use ``*args`` / ``**kwargs``).
        """
        _validate_callback_arity(callback, expected=5, name="sweep_driver")
        if self.transfer_settings is not None:
            warnings.warn(
                "A sweep_driver callback is being registered but transfer_settings "
                "are already set. The transfer_settings will have no effect — "
                "the sweep_driver is responsible for constructing FUNtoFEMnlbgs "
                "with the desired TransferSettings.",
                UserWarning,
                stacklevel=2,
            )
        self.sweep_driver = callback
        return self

    def set_transfer_settings(self, transfer_settings) -> "ParameterSweep":
        """Set the ``TransferSettings`` used by the default ``FUNtoFEMnlbgs`` driver.

        Only applies when no ``sweep_driver`` callback is registered.  If a
        ``sweep_driver`` is also set, a ``UserWarning`` is issued at call time
        because the transfer settings will have no effect — the custom driver
        is responsible for constructing its own ``FUNtoFEMnlbgs`` instance.

        Parameters
        ----------
        transfer_settings :
            A ``TransferSettings`` instance (e.g.
            ``TransferSettings(npts=200, beta=0.5)``).

        Returns
        -------
        ParameterSweep
            *self*, to allow method chaining.
        """
        if self.sweep_driver is not None:
            warnings.warn(
                "set_transfer_settings has no effect when a sweep_driver callback "
                "is registered — the sweep_driver is responsible for constructing "
                "FUNtoFEMnlbgs with the desired TransferSettings.",
                UserWarning,
                stacklevel=2,
            )
        self.transfer_settings = transfer_settings
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, comm, *, compute_adjoint=False, resume=False):
        """Run the full parameter sweep.

        Performs pre-run validation of all required callbacks and the parameter
        space before dispatching to the internal sweep implementation.

        Parameters
        ----------
        comm:
            MPI communicator.
        compute_adjoint:
            When ``True``, solve the adjoint after each forward solve and
            collect gradient columns in the result CSV.
        resume:
            When ``True``, read the existing result CSV and skip design points
            that already have ``status = "success"``.

        Raises
        ------
        ValueError
            If ``params`` is empty, ``model_builder`` or ``solver_builder`` is
            not registered, or the current ``mesh_mode`` requires a mesh
            callback that has not been registered.
        """
        # --- Pre-run validation ---

        # 1. params must be non-empty
        if not self.params:
            raise ValueError("params must be non-empty")

        # 2. model_builder is required
        if self.model_builder is None:
            raise ValueError("model_builder callback is required but not registered")

        # 3. solver_builder is required
        if self.solver_builder is None:
            raise ValueError("solver_builder callback is required but not registered")

        # 4. Mesh mode callback validation
        if self.mesh_mode in (MeshMode.FULL_REGEN, MeshMode.CFD_ONLY):
            if self.cfd_mesh_callback is None:
                raise ValueError(
                    f"mesh_mode={self.mesh_mode.value!r} requires a "
                    f"cfd_mesh_callback but none is registered"
                )

        if self.mesh_mode in (MeshMode.FULL_REGEN, MeshMode.STRUCT_ONLY):
            if self.struct_mesh_callback is None:
                raise ValueError(
                    f"mesh_mode={self.mesh_mode.value!r} requires a "
                    f"struct_mesh_callback but none is registered"
                )

        # --- Dispatch to internal implementation ---
        self._run_sweep(comm, compute_adjoint=compute_adjoint, resume=resume)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_output_dirs(self, key: str) -> "tuple[str, str]":
        """Return ``(cfd_output_dir, struct_output_dir)`` for a design key.

        Parameters
        ----------
        key:
            The design key string for the current design point.

        Returns
        -------
        tuple[str, str]
            ``(cfd_output_dir, struct_output_dir)`` where:
            - ``cfd_output_dir = f"{self.cfd_root}/{key}/cfd"``
            - ``struct_output_dir = f"{self.struct_root}/{key}/struct"``
        """
        cfd_output_dir = f"{self.cfd_root}/{key}/cfd"
        struct_output_dir = f"{self.struct_root}/{key}/struct"
        return cfd_output_dir, struct_output_dir

    def _dispatch_mesh_callbacks(
        self,
        comm,
        key: str,
        design_point: dict,
        cfd_output_dir: str,
        struct_output_dir: str,
    ) -> None:
        """Invoke mesh callbacks according to the current ``mesh_mode``.

        Dispatches CFD and/or structural mesh callbacks based on
        :attr:`mesh_mode`.  Both callbacks receive the full ``comm`` and are
        called across all MPI ranks; any rank-gating (e.g. ``if comm.rank == 0``)
        is the responsibility of the user-supplied callback.  A
        ``comm.Barrier()`` is issued after the struct callback so that all
        ranks are synchronised before the per-point execution loop resumes.

        Parameters
        ----------
        comm:
            MPI communicator.
        key:
            Design key for the current point (used to build ``caps_problem_name``).
        design_point:
            Dict mapping parameter names to values for the current point.
        cfd_output_dir:
            Per-point CFD output directory path.
        struct_output_dir:
            Per-point structural output directory path.
        """
        caps_problem_name = _make_caps_problem_name(self.caps_subdir, key)

        # Ensure caps_subdir exists before any mesh callback tries to use it.
        # Using exist_ok=True makes this safe for concurrent MPI ranks.
        if self.caps_subdir:
            os.makedirs(self.caps_subdir, exist_ok=True)

        if self.mesh_mode == MeshMode.FULL_REGEN:
            self.cfd_mesh_callback(
                comm, design_point, caps_problem_name, cfd_output_dir
            )
            self.struct_mesh_callback(
                comm, design_point, caps_problem_name, struct_output_dir
            )
            comm.Barrier()

        elif self.mesh_mode == MeshMode.CFD_ONLY:
            self.cfd_mesh_callback(
                comm, design_point, caps_problem_name, cfd_output_dir
            )

        elif self.mesh_mode == MeshMode.STRUCT_ONLY:
            self.struct_mesh_callback(
                comm, design_point, caps_problem_name, struct_output_dir
            )
            comm.Barrier()

        # MeshMode.NONE: call neither callback

    def _execute_point(
        self,
        comm,
        case_idx: int,
        total: int,
        key: str,
        design_point: dict,
        cfd_output_dir: str,
        struct_output_dir: str,
        *,
        compute_adjoint: bool = False,
    ) -> None:
        """Execute a single design point: model builder → solver builder → driver → result extractor.

        Wraps execution in a ``try/except Exception`` block so that a failure in
        one point is recorded (``status="failed"``) and the sweep continues to
        the next point.

        Parameters
        ----------
        comm:
            MPI communicator.
        case_idx:
            1-based index of the current design point.
        total:
            Total number of design points in the sweep.
        key:
            Design key for the current point.
        design_point:
            Dict mapping parameter names to values.
        cfd_output_dir:
            Per-point CFD output directory path.
        struct_output_dir:
            Per-point structural output directory path.
        compute_adjoint:
            When ``True``, solve the adjoint after the forward solve.
        """
        rank = getattr(comm, "rank", 0)

        # --- Progress: starting ---
        if rank == 0:
            print(
                f"[sweep] case {case_idx}/{total} starting: "
                f"key={key}, point={design_point}"
            )

        results = {}
        status = "failed"
        error_msg = None

        try:
            # 1. Build the model
            model = self.model_builder(
                comm, design_point, cfd_output_dir, struct_output_dir
            )

            # 2. Build solvers
            solvers = self.solver_builder(
                comm, model, design_point, cfd_output_dir, struct_output_dir
            )

            # 3. Run the driver
            if self.sweep_driver is not None:
                # Custom driver callback handles the solve; it may optionally
                # return the driver object so the result extractor can access it
                # (e.g. to read residuals). If it returns None, the extractor
                # receives None for the driver argument.
                driver = self.sweep_driver(
                    comm, model, solvers, design_point, compute_adjoint
                )
                # When using a custom driver with compute_adjoint=True,
                # attempt gradient collection from the model directly
                if compute_adjoint:
                    results.update(self._collect_gradients(model))
            else:
                # Default path: construct FUNtoFEMnlbgs and run forward (+ adjoint)
                try:
                    from ..driver import FUNtoFEMnlbgs
                except ImportError:
                    FUNtoFEMnlbgs = None

                if FUNtoFEMnlbgs is None:
                    raise ImportError(
                        "FUNtoFEMnlbgs could not be imported — check that "
                        "funtofem native libraries are built"
                    )

                driver = FUNtoFEMnlbgs(
                    solvers, model=model, transfer_settings=self.transfer_settings
                )
                driver.solve_forward()
                if compute_adjoint:
                    driver.solve_adjoint()
                    # Collect dF/dp gradients for each (function, shape variable) pair
                    results.update(self._collect_gradients(model))

            # 4. Extract results
            extractor = (
                self.result_extractor
                if self.result_extractor is not None
                else _default_result_extractor
            )
            extractor_results = extractor(
                model, driver, design_point, cfd_output_dir, struct_output_dir
            )
            # Merge extractor results; gradient columns collected above take lower priority
            # than explicitly extracted results (extractor results win on collision)
            grad_results = results  # gradients collected so far (may be empty)
            results = {**grad_results, **extractor_results}

            status = "success"

            if rank == 0:
                print(
                    f"[sweep] case {case_idx}/{total} done: "
                    f"key={key}, status=success, results={results}"
                )

        except Exception as exc:
            import traceback

            status = "failed"
            error_msg = str(exc)

            if rank == 0:
                print(
                    f"[sweep] case {case_idx}/{total} FAILED: "
                    f"key={key}, error={error_msg}"
                )
                traceback.print_exc()

        # 5. Write the CSV row
        self._write_csv_row(
            comm,
            case_idx,
            key,
            design_point,
            status,
            results if status == "success" else {},
            error_msg if status == "failed" else None,
            compute_adjoint=compute_adjoint,
        )

    def _collect_gradients(self, model) -> dict:
        """Collect adjoint gradient values from the model after solve_adjoint().

        Reads ``func.derivatives`` (a dict keyed by ``Variable`` objects) for
        every function in every scenario, and for every design variable returned
        by ``model.get_variables()``.  Gradient columns are named
        ``d<func_name>_d<var_name>``.

        When a gradient value is unavailable (e.g. the function's derivatives
        dict does not contain the variable key) the column is set to
        ``float("nan")``.

        Parameters
        ----------
        model : FUNtoFEMmodel or None
            The FUNtoFEM model whose gradients are to be collected.  When
            ``None``, an empty dict is returned.

        Returns
        -------
        dict[str, float]
            Mapping of gradient column names to gradient values (or nan).
        """
        if model is None:
            return {}

        grad_results = {}

        # Collect all functions from all scenarios
        funcs = []
        for scenario in getattr(model, "scenarios", []):
            funcs.extend(getattr(scenario, "functions", []))

        # Collect all design variables
        variables = []
        try:
            variables = model.get_variables()
        except Exception:
            pass

        for func in funcs:
            for var in variables:
                col = f"d{func.name}_d{var.name}"
                try:
                    # Gradients are stored in func.derivatives dict keyed by variable
                    grad_val = func.derivatives.get(var, float("nan"))
                except Exception:
                    grad_val = float("nan")
                grad_results[col] = grad_val

        return grad_results

    def _write_csv_row(
        self,
        comm,
        case_idx,
        key,
        design_point,
        status,
        results,
        error_msg,
        *,
        compute_adjoint=False,
    ) -> None:
        """Write a single result row to the output CSV.

        Only rank 0 writes. Writes header if file doesn't exist yet.
        Appends without rewriting header if file already exists.
        Non-scalar/non-string result values are coerced to str.

        Parameters
        ----------
        comm:
            MPI communicator (or any object with a ``rank`` attribute).
        case_idx:
            1-based index of the current design point.
        key:
            Design key for the current point.
        design_point:
            Dict mapping parameter names to values.
        status:
            ``"success"`` or ``"failed"``.
        results:
            Result dict from the result extractor, or ``{}`` on failure.
        error_msg:
            Exception message string on failure, or ``None`` on success.
        compute_adjoint:
            Whether adjoint gradients were computed for this point.
        """
        rank = getattr(comm, "rank", 0)
        if rank != 0:
            return

        # Build the row dict
        row = {}
        row["case"] = case_idx
        row["key"] = key
        # One column per param
        for param_name, param_value in design_point.items():
            row[param_name] = param_value
        row["status"] = status
        if error_msg is not None:
            row["error"] = error_msg
        # Result columns — coerce non-scalar/non-string values to str
        for result_key, result_value in results.items():
            if not isinstance(result_value, (int, float, str, bool, type(None))):
                result_value = str(result_value)
            row[result_key] = result_value

        # Determine fieldnames from the row
        fieldnames = list(row.keys())

        csv_exists = os.path.exists(self.output_csv)

        if csv_exists:
            # Read existing header to check for new columns
            with open(self.output_csv, "r", newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
                # Pass fieldnames explicitly so DictReader doesn't re-consume
                # the first data row as a header (f is already past line 1).
                existing_rows = (
                    list(csv.DictReader(f, fieldnames=existing_header))
                    if existing_header
                    else []
                )

            if existing_header is not None:
                new_cols = [col for col in fieldnames if col not in existing_header]
                if new_cols:
                    # New columns discovered — rewrite the entire file with the
                    # expanded header so no data is silently dropped
                    updated_header = existing_header + new_cols
                    with open(self.output_csv, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=updated_header, extrasaction="ignore"
                        )
                        writer.writeheader()
                        writer.writerows(existing_rows)
                    fieldnames = updated_header
                else:
                    fieldnames = existing_header

        # Write to CSV
        mode = "a" if csv_exists else "w"
        with open(self.output_csv, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not csv_exists:
                writer.writeheader()
            writer.writerow(row)

    def _run_sweep(self, comm, *, compute_adjoint=False, resume=False):
        """Internal sweep implementation.

        Enumerates all design points via the configured strategy, computes
        per-point output directories, dispatches mesh callbacks, and
        executes each point.

        Parameters
        ----------
        comm:
            MPI communicator.
        compute_adjoint:
            When ``True``, solve the adjoint after each forward solve.
        resume:
            When ``True``, skip design points that already have
            ``status = "success"`` in the result CSV.
        """
        design_points = self.strategy.generate(self.params)
        total = len(design_points)

        # Build set of already-completed keys when resume=True
        completed_keys = set()
        if resume and os.path.exists(self.output_csv):
            with open(self.output_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == "success":
                        completed_keys.add(row.get("key", ""))

        rank = getattr(comm, "rank", 0)

        for case_idx, design_point in enumerate(design_points, start=1):
            key = _make_key(design_point, self.key_fn)

            if key in completed_keys:
                if rank == 0:
                    print(
                        f"[sweep] case {case_idx}/{total} SKIPPED (resume): key={key}"
                    )
                continue

            cfd_output_dir, struct_output_dir = self._make_output_dirs(key)
            self._dispatch_mesh_callbacks(
                comm, key, design_point, cfd_output_dir, struct_output_dir
            )
            self._execute_point(
                comm,
                case_idx,
                total,
                key,
                design_point,
                cfd_output_dir,
                struct_output_dir,
                compute_adjoint=compute_adjoint,
            )

    def run_single(self, comm, design_point, *, compute_adjoint=False):
        """Run the sweep for a single design point.

        Validates that the required callbacks are registered, then dispatches
        to the internal single-point implementation.

        Parameters
        ----------
        comm:
            MPI communicator.
        design_point:
            Dict mapping parameter names to values for the single point to run.
        compute_adjoint:
            When ``True``, solve the adjoint after the forward solve.

        Raises
        ------
        ValueError
            If ``model_builder`` or ``solver_builder`` is not registered.
        """
        # Minimal validation: model_builder and solver_builder are required
        if self.model_builder is None:
            raise ValueError("model_builder callback is required but not registered")

        if self.solver_builder is None:
            raise ValueError("solver_builder callback is required but not registered")

        # --- Dispatch to internal implementation ---
        self._run_single_point(comm, design_point, compute_adjoint=compute_adjoint)

    def _run_single_point(self, comm, design_point, *, compute_adjoint=False):
        """Internal single-point implementation.

        Computes output directories and dispatches mesh callbacks for the
        given design point, then executes it.

        Parameters
        ----------
        comm:
            MPI communicator.
        design_point:
            Dict mapping parameter names to values for the single point to run.
        compute_adjoint:
            When ``True``, solve the adjoint after the forward solve.
        """
        key = _make_key(design_point, self.key_fn)
        cfd_output_dir, struct_output_dir = self._make_output_dirs(key)
        self._dispatch_mesh_callbacks(
            comm, key, design_point, cfd_output_dir, struct_output_dir
        )
        self._execute_point(
            comm,
            1,
            1,
            key,
            design_point,
            cfd_output_dir,
            struct_output_dir,
            compute_adjoint=compute_adjoint,
        )


def load_point_file(path: str) -> dict:
    """Load a design point from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file (e.g. produced by SlurmSweepSubmitter).

    Returns
    -------
    dict
        Design point mapping parameter names to values.

    Raises
    ------
    FileNotFoundError
        If the file at *path* does not exist.
    """
    with open(path, "r") as f:
        return json.load(f)


def cli_main(sweep: "ParameterSweep", comm, *, compute_adjoint: bool = False) -> None:
    """Parse ``--point-file`` CLI argument and run a single design point.

    This is the standard entry point for SLURM job scripts. Call it at the end
    of your sweep script to enable single-point execution::

        if __name__ == "__main__":
            cli_main(sweep, comm)

    Parameters
    ----------
    sweep : ParameterSweep
        Configured :class:`ParameterSweep` instance with all callbacks registered.
    comm : MPI.Intracomm
        MPI communicator.
    compute_adjoint : bool
        When ``True``, solve the adjoint after the forward solve.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single design point specified by --point-file."
    )
    parser.add_argument(
        "--point-file",
        required=True,
        metavar="PATH",
        help="Path to a JSON file containing the design point parameters.",
    )
    args = parser.parse_args()

    design_point = load_point_file(args.point_file)
    sweep.run_single(comm, design_point, compute_adjoint=compute_adjoint)
