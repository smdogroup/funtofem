"""
funtofem.sweep.strategy — Sweep strategy classes.

Defines the SweepStrategy abstract base class and the two built-in
concrete implementations: CartesianStrategy (full Cartesian product) and
ZipStrategy (element-wise pairing).
"""

import itertools
from abc import ABC, abstractmethod


class SweepStrategy(ABC):
    """Abstract base class for parameter sweep strategies.

    A sweep strategy converts a parameter space mapping (dict of parameter
    names to lists of values) into an ordered list of design points, where
    each design point is a dict mapping parameter names to a single value.

    Subclasses must implement :meth:`generate`.
    """

    @abstractmethod
    def generate(self, params: dict[str, list]) -> list[dict]:
        """Convert a parameter space mapping to an ordered list of design points.

        Parameters
        ----------
        params : dict[str, list]
            Mapping of parameter name to list of values.

        Returns
        -------
        list[dict]
            Each dict maps parameter names to single values for that point.
        """


class CartesianStrategy(SweepStrategy):
    """Sweep strategy that produces the full Cartesian product of all parameter
    value lists.

    For a parameter space with k parameters having n_1, n_2, ..., n_k values
    respectively, this strategy produces n_1 * n_2 * ... * n_k design points
    covering every possible combination of one value per parameter.

    This is the default strategy for :class:`~funtofem.sweep.ParameterSweep`.

    Examples
    --------
    >>> strategy = CartesianStrategy()
    >>> points = strategy.generate({"alpha": [1, 2], "beta": ["a", "b"]})
    >>> len(points)
    4
    >>> points[0]
    {'alpha': 1, 'beta': 'a'}
    """

    def generate(self, params: dict[str, list]) -> list[dict]:
        """Produce the full Cartesian product of all parameter value lists.

        Parameters
        ----------
        params : dict[str, list]
            Mapping of parameter name to list of values.

        Returns
        -------
        list[dict]
            All combinations of one value per parameter. The number of points
            equals the product of all list lengths.
        """
        if not params:
            return [{}]
        keys = list(params.keys())
        values = list(params.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


class ZipStrategy(SweepStrategy):
    """Sweep strategy that pairs parameter values element-wise.

    All parameter value lists must have the same length N; the strategy
    produces exactly N design points where the i-th point contains the i-th
    value from every parameter's list.

    Raises
    ------
    ValueError
        If the parameter value lists do not all have the same length.

    Examples
    --------
    >>> strategy = ZipStrategy()
    >>> points = strategy.generate({"alpha": [1, 2, 3], "beta": [10, 20, 30]})
    >>> len(points)
    3
    >>> points[1]
    {'alpha': 2, 'beta': 20}
    """

    def generate(self, params: dict[str, list]) -> list[dict]:
        """Pair parameter values element-wise across all lists.

        Parameters
        ----------
        params : dict[str, list]
            Mapping of parameter name to list of values. All lists must have
            the same length.

        Returns
        -------
        list[dict]
            Exactly N design points where N is the (common) length of all
            value lists.

        Raises
        ------
        ValueError
            If the value lists are not all the same length.
        """
        if not params:
            return [{}]

        keys = list(params.keys())
        values = list(params.values())
        lengths = [len(v) for v in values]

        if len(set(lengths)) > 1:
            detail = ", ".join(f"{k!r}: {n}" for k, n in zip(keys, lengths))
            raise ValueError(
                f"ZipStrategy requires all parameter lists to have equal length, "
                f"but got differing lengths — {detail}"
            )

        return [dict(zip(keys, combo)) for combo in zip(*values)]
