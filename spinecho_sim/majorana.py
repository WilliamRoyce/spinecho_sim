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
    # TODO: describe or cite algorithm
    # build polynomial coefficients a_k
    k_arr = np.arange(len(spin_coefficients))
    binomial_arr = np.sqrt([comb(two_j, k) for k in k_arr])
    # TODO: what do you mean by this comment?
    # Beware indexing of c_{j-k}!
    polynomial_coefficients = binomial_arr * spin_coefficients[two_j - k_arr]

    # roots
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


def _get_majorana_coefficients_from_spin_multiple(
    spin_coefficients: np.ndarray[Any, np.dtype[np.complexfloating]], z_tol: float = 1e8
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute Majorana points for multiple sets of spin coefficients."""
    return np.array(
        [
            _get_majorana_coefficients_from_spin(c, z_tol=z_tol)
            for c in spin_coefficients
        ]
    )


def majorana_points_by_index(
    spin_coefficients: np.ndarray[Any, np.dtype[np.complexfloating]], z_tol: float = 1e8
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute Majorana points for multiple arrays and group them by index."""
    polynomial_coefficients = _get_majorana_coefficients_from_spin_multiple(
        spin_coefficients, z_tol=z_tol
    )
    n_states = polynomial_coefficients.shape[0]
    num_points = max(points.shape[0] for points in polynomial_coefficients)

    # TODO: this should be done in _get_majorana_coefficients_from_spin
    # Pad each state's points to num_points with (pi, 0.0) if needed
    padded_points = np.empty((n_states, num_points, 2), dtype=np.float64)
    for i, points in enumerate(polynomial_coefficients):
        n = points.shape[0]
        if n < num_points:
            pad = np.tile([np.pi, 0.0], (num_points - n, 1))
            padded_points[i] = np.vstack((points, pad))
        else:
            padded_points[i] = points
    # TODO: I dont think this comment helps
    # Do NOT transpose; keep shape (n_states, n_points, 2)
    return padded_points


def _stars_to_polynomial(
    stars: np.ndarray[Any, np.dtype[np.float64]], tol: float = 1e-8
) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
    """Convert a list of Majorana stars (theta, phi) to a polynomial representation."""
    finite_mask = np.abs(stars[:, 0] - np.pi) >= tol

    finite_theta = stars[finite_mask, 0]
    finite_phi = stars[finite_mask, 1]
    finite_roots = np.exp(1j * finite_phi) * np.tan(finite_theta / 2)

    # TODO: np.count_nonzero(finite_mask) gives the number of finite roots
    n_infinity = int(np.sum(~finite_mask))

    # polynomial from the finite roots (ascending order)
    a = p.polyfromroots(finite_roots)  # degree = len(finite)
    a = np.asarray(a, dtype=np.complex128)  # ensure correct dtype
    # each root at ∞ loses degree in P(z)   →   pad with one zero on the right
    # TODO: I dont think an if statement is needed here
    if n_infinity:
        a = np.concatenate((a, np.zeros(n_infinity, dtype=a.dtype)))

    return a


def _polynomial_to_state(a: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Convert a polynomial representation to quantum state coefficients."""
    two_j = len(a) - 1
    c = np.empty(len(a), dtype=np.complex128)

    # Majorana conversion:  a_k = √C(2J,k) c_{J-k}
    for k in range(two_j + 1):
        # TODO: comment should say why beware of indexing
        c[two_j - k] = a[k] / np.sqrt(comb(two_j, k))  # Beware indexing of c_{J-k}!

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
) -> NDArray[np.complexfloating]:
    """Convert multiple sets of Majorana stars (theta, phi) to quantum state coefficients."""
    # TODO: comment is wrong, it is not vectorized
    # Vectorised if all states have the same number of stars and output length
    return np.array(
        [_stars_to_state(stars[i], tol=tol) for i in range(stars.shape[0])],
        dtype=np.complex128,
    )
