"""Provide functions to convert between Majorana stars and quantum state coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.polynomial import polynomial as p
from scipy.special import comb  # type: ignore[import]

if TYPE_CHECKING:
    from numpy.typing import NDArray

NUM_SPIN_PARAMS = 2  # Number of parameters per spin (theta, phi)


def _stars_to_polynomial(
    stars: np.ndarray[Any, np.dtype[np.float64]], tol: float = 1e-8
) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
    """Convert a list of Majorana stars (theta, phi) to a polynomial representation."""
    finite_mask = np.abs(stars[:, 0] - np.pi) >= tol

    finite_theta = stars[finite_mask, 0]
    finite_phi = stars[finite_mask, 1]
    finite_roots = np.exp(1j * finite_phi) * np.tan(finite_theta / 2)

    n_infinity = np.count_nonzero(~finite_mask)

    # polynomial from the finite roots (ascending order)
    a = np.asarray(
        p.polyfromroots(finite_roots), dtype=np.complex128
    )  # degree = len(finite)
    # each root at ∞ loses degree in P(z)   →   pad with one zero on the right
    return np.concatenate((a, np.zeros(n_infinity, dtype=a.dtype)))


def _polynomial_to_state(
    a: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Convert a polynomial representation to quantum state coefficients."""
    two_j = len(a) - 1
    binomial_weights = np.sqrt(
        np.asarray(comb(two_j, np.arange(two_j + 1)), dtype=np.float64)
    )
    c = (a / binomial_weights)[::-1].astype(np.complex128)
    c /= np.linalg.norm(c)

    return c


def stars_to_state(
    stars: np.ndarray[tuple[int, int], np.dtype[np.float64]], *, tol: float = 1e-10
) -> NDArray[np.complex128]:
    """Convert a list of Majorana stars (theta, phi) to the corresponding quantum state coefficients."""
    assert stars.ndim == 2, "Stars must be a 2D array"
    assert stars.shape[1] == NUM_SPIN_PARAMS, "Stars must have shape (n_stars, 2)"
    a = _stars_to_polynomial(stars, tol=tol)
    return _polynomial_to_state(a.astype(np.complex128))


def stars_to_states(
    stars: np.ndarray[tuple[int, int, int], np.dtype[np.float64]], *, tol: float = 1e-10
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Convert multiple sets of Majorana stars (theta, phi) to quantum state coefficients."""
    assert stars.ndim == 3, "Stars must be a 3D array"
    assert stars.shape[2] == NUM_SPIN_PARAMS, (
        "Stars must have shape (n_groups, n_stars, 2)"
    )
    # Conversion of all stars to states
    return np.stack([stars_to_state(stars[i], tol=tol) for i in range(stars.shape[0])])
