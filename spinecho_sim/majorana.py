from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import polynomial as p

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _majorana_points(
    c: NDArray[np.complexfloating], z_tol: float = 1e8
) -> NDArray[np.float64]:
    """Compute the Majorana points (Bloch sphere coordinates) for a given quantum state."""
    c = np.asarray(c, dtype=np.complex128)
    j = (len(c) - 1) / 2
    c /= np.linalg.norm(c)  # normalize

    two_j = int(2 * j)

    # build polynomial coefficients a_k
    k_arr = np.arange(two_j + 1)
    binomial_arr = np.sqrt(
        [comb(two_j, k) for k in k_arr]
    )  # Beware indexing of c_{j-k}!
    a = binomial_arr * c[two_j - k_arr]

    # roots
    z = p.polyroots(a)  # returns 2J complex roots
    abs_z = np.abs(z)
    angle_z = np.angle(z)

    theta = np.where(abs_z > z_tol, np.pi, 2 * np.arctan(abs_z))
    phi = np.where(abs_z > z_tol, 0, angle_z % (2 * np.pi))
    stars = list(zip(theta, phi, strict=True))
    stars = np.column_stack((theta, phi))
    # ensure exactly 2j points, if lost degrees due to vanishing highest coefficients
    while stars.shape[0] < 2 * j:
        stars = np.vstack((stars, [np.pi, 0.0]))
    return stars


def _majorana_points_multiple(
    arrays: NDArray[np.complexfloating], z_tol: float = 1e8
) -> NDArray[np.float64]:
    """Compute Majorana points for multiple arrays."""
    return np.array([_majorana_points(c, z_tol=z_tol) for c in arrays])


def majorana_points_by_index(
    arrays: NDArray[np.complexfloating], z_tol: float = 1e8
) -> NDArray[np.float64]:
    """Compute Majorana points for multiple arrays and group them by index."""
    points_list = _majorana_points_multiple(arrays, z_tol=z_tol)
    n_states = points_list.shape[0]
    num_points = max(points.shape[0] for points in points_list)

    # Pad each state's points to num_points with (pi, 0.0) if needed
    padded_points = np.empty((n_states, num_points, 2), dtype=np.float64)
    for i, points in enumerate(points_list):
        n = points.shape[0]
        if n < num_points:
            pad = np.tile([np.pi, 0.0], (num_points - n, 1))
            padded_points[i] = np.vstack((points, pad))
        else:
            padded_points[i] = points

    # Do NOT transpose; keep shape (n_states, n_points, 2)
    return padded_points


def _stars_to_polynomial(
    stars: NDArray[np.float64], tol: float = 1e-8
) -> NDArray[np.complexfloating]:
    """Convert a list of Majorana stars (theta, phi) to a polynomial representation."""
    finite_mask = np.abs(stars[:, 0] - np.pi) >= tol

    finite_theta = stars[finite_mask, 0]
    finite_phi = stars[finite_mask, 1]
    finite_roots = np.exp(1j * finite_phi) * np.tan(finite_theta / 2)

    n_infinity = int(np.sum(~finite_mask))

    # polynomial from the finite roots (ascending order)
    a = p.polyfromroots(finite_roots)  # degree = len(finite)
    a = np.asarray(a, dtype=np.complex128)  # ensure correct dtype
    # each root at ∞ loses degree in P(z)   →   pad with one zero on the right
    if n_infinity:
        a = np.concatenate((a, np.zeros(n_infinity, dtype=a.dtype)))

    return a


def _polynomial_to_state(a: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Convert a polynomial representation to quantum state coefficients."""
    j = (len(a) - 1) / 2
    c: NDArray[np.complexfloating] = np.empty(len(a), dtype=np.complex128)

    two_j = int(2 * j)

    # Majorana conversion:  a_k = √C(2J,k) c_{J-k}
    for k in range(two_j + 1):
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
    # Vectorised if all states have the same number of stars and output length
    return np.array(
        [_stars_to_state(stars[i], tol=tol) for i in range(stars.shape[0])],
        dtype=np.complex128,
    )
