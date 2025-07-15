from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial import polynomial as p
from scipy.special import comb  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import NDArray


def majorana_points(c: np.ndarray, ztol: float = 1e8) -> list[tuple[float, float]]:
    c = np.asarray(c, dtype=np.complex128)
    j = (len(c) - 1) / 2
    # normalise
    c /= np.linalg.norm(c)

    two_j = int(2 * j)

    # build polynomial coefficients a_k
    a: NDArray[np.complex128] = np.zeros(two_j + 1, dtype=np.complex128)
    for k in range(two_j + 1):
        a[k] = np.sqrt(comb(two_j, k)) * c[two_j - k]  # Beware indexing of c_{j-k}!

    # roots
    z = p.polyroots(a)  # returns 2J complex roots

    # map to Bloch sphere
    stars = []
    for zk in z:
        if np.abs(zk) > ztol:  # treat as infinity
            theta = np.pi
            phi = np.angle(zk)
        else:
            theta = 2 * np.arctan(np.abs(zk))
            phi = np.angle(zk) % (2 * np.pi)
        stars.append((theta, phi))
    # ensure exactly 2j points, if lost degrees due to vanishing highest coefficients
    while len(stars) < 2 * j:
        stars.append((np.pi, 0.0))
    return stars


def majorana_points_multiple(
    arrays: list[np.ndarray], ztol: float = 1e8
) -> list[list[tuple[float, float]]]:
    return [majorana_points(c, ztol=ztol) for c in arrays]


def majorana_points_by_index(
    arrays: list[np.ndarray], ztol: float = 1e8
) -> list[list[tuple[float, float]]]:
    points_list = majorana_points_multiple(arrays, ztol=ztol)
    num_points = max(len(points) for points in points_list)
    result = []
    for i in range(num_points):
        ith_points = []
        for points in points_list:
            if i < len(points):
                ith_points.append(points[i])
            else:
                ith_points.append((np.pi, 0.0))  # pad if missing
        result.append(ith_points)
    return result


def _stars_to_polynomial(
    stars: list[tuple[float, float]], tol: float = 1e-8
) -> NDArray[np.complex128]:
    finite: list[complex] = []
    n_infty = 0  # stars at south-pole (theta ≈ π)

    for theta, phi in stars:
        if abs(theta - np.pi) < tol:  # treat as root at z = ∞
            n_infty += 1
        else:
            finite.append(np.exp(1j * phi) * np.tan(theta / 2))

    # polynomial from the finite roots (ascending order)
    a = p.polyfromroots(finite)  # degree = len(finite)
    # each root at ∞ loses degree in P(z)   →   pad with one zero on the right
    if n_infty:
        a = np.concatenate((a, np.zeros(n_infty, dtype=a.dtype)))

    return a


def _polynomial_to_state(a: NDArray[np.complex128]) -> NDArray[np.complex128]:
    j = (len(a) - 1) / 2
    c = np.empty(len(a), dtype=np.complex128)

    two_j = int(2 * j)

    # Majorana / Bargmann conversion:  a_k = √C(2J,k) c_{J−k}
    for k in range(two_j + 1):
        c[two_j - k] = a[k] / np.sqrt(comb(two_j, k))  # Beware indexing of c_{J-k}!

    # strip the arbitrary global phase and renormalise
    idx_max = np.argmax(np.abs(c))
    c *= np.exp(-1j * np.angle(np.asarray(c[idx_max])))
    c /= np.linalg.norm(c)
    return c


def stars_to_state(stars, tol=1e-10):
    a = _stars_to_polynomial(stars, tol=tol)
    return _polynomial_to_state(a)
