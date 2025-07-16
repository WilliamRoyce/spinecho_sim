"""Provide functions to convert between Majorana stars and quantum state coefficients."""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.polynomial import polynomial as p

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _get_majorana_coefficients_from_spin(
    spin_coefficients: np.ndarray[Any, np.dtype[np.complexfloating]], z_tol: float = 1e8
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute the Majorana points (Bloch sphere coordinates) for a given quantum state."""
    two_j = len(spin_coefficients) - 1

    # build polynomial coefficients a_k
    k_arr = np.arange(len(spin_coefficients))
    binomial_arr = np.sqrt([comb(two_j, k) for k in k_arr])
    polynomial_coefficients = binomial_arr * spin_coefficients[two_j - k_arr]

    z = p.polyroots(polynomial_coefficients)  # returns 2J complex roots
    abs_z = np.abs(z)
    angle_z = np.angle(z)

    theta = np.where(abs_z > z_tol, np.pi, 2 * np.arctan(abs_z))
    phi = np.where(abs_z > z_tol, 0, angle_z % (2 * np.pi))
    stars = np.column_stack((theta, phi))
    # ensure exactly 2j points, if lost degrees due to vanishing highest coefficients
    while stars.shape[0] < two_j:
        stars = np.vstack((stars, [np.pi, 0.0]))
    return stars


def majorana_stars_old(
    spin_coefficients: np.ndarray[Any, np.dtype[np.complexfloating]], z_tol: float = 1e8
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute Majorana points for multiple sets of spin coefficients."""
    points_list = [
        _get_majorana_coefficients_from_spin(c, z_tol=z_tol) for c in spin_coefficients
    ]
    # Calculate j from the length of the spin vector
    j = (spin_coefficients.shape[1] - 1) / 2  # Spin-j vector has 2j+1 coefficients
    num_points = int(2 * j)
    padded_points = np.empty((len(points_list), num_points, 2), dtype=np.float64)
    for i, points in enumerate(points_list):
        n = points.shape[0]
        if n < num_points:
            pad = np.tile([np.pi, 0.0], (num_points - n, 1))
            padded_points[i] = np.vstack((points, pad))
        else:
            padded_points[i] = points
    return padded_points


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
    a = p.polyfromroots(finite_roots)  # degree = len(finite)
    a = np.asarray(a, dtype=np.complex128)  # ensure correct dtype
    # each root at ∞ loses degree in P(z)   →   pad with one zero on the right
    return np.concatenate((a, np.zeros(n_infinity, dtype=a.dtype)))


def _polynomial_to_state(a: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Convert a polynomial representation to quantum state coefficients."""
    two_j = len(a) - 1
    c = np.empty(len(a), dtype=np.complex128)

    # Majorana conversion:  a_k = √C(2J,k) c_{J-k}
    for k in range(two_j + 1):
        c[two_j - k] = a[k] / np.sqrt(comb(two_j, k))

    # strip the arbitrary global phase and renormalize
    idx_max = np.argmax(np.abs(c))
    c *= np.exp(-1j * np.angle(np.asarray(c[idx_max])))
    c /= np.linalg.norm(c)
    return c


def _stars_to_state(
    stars: NDArray[np.float64], tol: float = 1e-10
) -> NDArray[np.complexfloating]:
    """Convert a list of Majorana stars (theta, phi) to the corresponding quantum state coefficients."""
    a = _stars_to_polynomial(stars, tol=tol)
    return _polynomial_to_state(a)


def stars_to_states(
    stars: NDArray[np.float64], tol: float = 1e-10
) -> np.ndarray[tuple[int, int], np.dtype[np.complexfloating]]:
    """Convert multiple sets of Majorana stars (theta, phi) to quantum state coefficients."""
    # Vectorized as all states have the same number of stars and output length
    return np.stack([_stars_to_state(stars[i], tol=tol) for i in range(stars.shape[0])])
