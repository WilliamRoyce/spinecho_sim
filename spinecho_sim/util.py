from __future__ import annotations

import datetime
from functools import wraps
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure


def get_figure(ax: Axes | None = None) -> tuple[Figure | SubFigure, Axes]:
    """Get a figure and axes for plotting."""
    if ax is None:
        return plt.subplots(figsize=(10, 6))
    return ax.figure, ax


Measure = Literal["real", "imag", "abs", "arg"]


def get_measure(arr: np.ndarray, measure: Measure) -> np.ndarray:
    """Get a specific measure of a complex array."""
    if measure == "real":
        return np.real(arr)
    if measure == "imag":
        return np.imag(arr)
    if measure == "abs":
        return np.abs(arr)
    if measure == "arg":
        return np.angle(arr)
    return None


def timed[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """
    Log the time taken for f to run.

    Parameters
    ----------
    f : Callable[P, R]
        The function to time

    Returns
    -------
    Callable[P, R]
        The decorated function
    """

    @wraps(f)
    def wrap(*args: P.args, **kw: P.kwargs) -> R:
        ts = datetime.datetime.now(tz=datetime.UTC)
        try:
            result = f(*args, **kw)
        finally:
            te = datetime.datetime.now(tz=datetime.UTC)
            print(f"func: {f.__name__} took: {(te - ts).total_seconds()} sec")  # noqa: T201
        return result

    return wrap  # type: ignore[return-value]
