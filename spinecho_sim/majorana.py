from __future__ import annotations

from math import comb  # type: ignore[import-untyped]
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import polynomial as p

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _majorana_points(c: np.ndarray, z_tol: float = 1e8) -> list[tuple[float, float]]:
    """Compute the Majorana points (Bloch sphere coordinates) for a given quantum state."""
    c = np.asarray(c, dtype=np.complex128)
    j = (len(c) - 1) / 2
    c /= np.linalg.norm(c)  # normalize

    two_j = int(2 * j)

    # build polynomial coefficients a_k
    a: NDArray[np.complexfloating] = np.zeros(two_j + 1, dtype=np.complex128)
    for k in range(two_j + 1):
        a[k] = np.sqrt(comb(two_j, k)) * c[two_j - k]  # Beware indexing of c_{j-k}!

    # roots
    z = p.polyroots(a)  # returns 2J complex roots

    # map to Bloch sphere
    stars: list[tuple[float, float]] = []
    for zk in z:
        abs_zk: float = float(np.abs(zk))
        angle_zk: float = float(np.angle(zk))
        if abs_zk > z_tol:  # treat as infinity
            theta = np.pi
            phi = angle_zk
        else:
            theta = 2 * np.arctan(abs_zk)
            phi = angle_zk % (2 * np.pi)
        stars.append((theta, phi))
    # ensure exactly 2j points, if lost degrees due to vanishing highest coefficients
    while len(stars) < 2 * j:
        stars.append((np.pi, 0.0))
    return stars


def _majorana_points_multiple(
    arrays: list[np.ndarray], z_tol: float = 1e8
) -> list[list[tuple[float, float]]]:
    """Compute Majorana points for multiple arrays."""
    return [_majorana_points(c, z_tol=z_tol) for c in arrays]


def majorana_points_by_index(
    arrays: list[np.ndarray], z_tol: float = 1e8
) -> list[list[tuple[float, float]]]:
    """Compute Majorana points for multiple arrays and group them by index."""
    points_list = _majorana_points_multiple(arrays, z_tol=z_tol)
    num_points = max(len(points) for points in points_list)
    result: list[list[tuple[float, float]]] = []
    for i in range(num_points):
        ith_points: list[tuple[float, float]] = []
        for points in points_list:
            if i < len(points):
                ith_points.append(points[i])
            else:
                ith_points.append(
                    (np.pi, 0.0)
                )  # pad with infinity roots if missing due to vanishing coefficients
        result.append(ith_points)
    return result


def _stars_to_polynomial(
    stars: list[tuple[float, float]], tol: float = 1e-8
) -> NDArray[np.complexfloating]:
    finite: list[complex] = []
    n_infinity = 0  # stars at south-pole (theta ≈ π)

    for theta, phi in stars:
        if abs(theta - np.pi) < tol:  # treat as root at z = ∞
            n_infinity += 1
        else:
            finite.append(np.exp(1j * phi) * np.tan(theta / 2))

    # polynomial from the finite roots (ascending order)
    a = p.polyfromroots(finite)  # degree = len(finite)
    # each root at ∞ loses degree in P(z)   →   pad with one zero on the right
    if n_infinity:
        a = np.concatenate((a, np.zeros(n_infinity, dtype=a.dtype)))

    return a


def _polynomial_to_state(a: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
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


def stars_to_state(stars: list[tuple[float, float]], tol: float = 1e-10) -> np.ndarray:
    """Convert a list of Majorana stars (theta, phi) to the corresponding quantum state coefficients."""
    a = _stars_to_polynomial(stars, tol=tol)
    return _polynomial_to_state(a)
