"""Provide functions to convert between Majorana stars and quantum state coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.polynomial import polynomial as p
from scipy.special import comb  # type: ignore[import]

from spinecho_sim import (
    ParticleState,
    Solenoid,
)
from spinecho_sim.majorana._companion_helper import (
    majorana_stars,
)
from spinecho_sim.state import (
    CoherentSpin,
    ParticleDisplacement,
    TrajectoryList,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _get_majorana_coefficients_from_spin_old(
    spin_coefficients: np.ndarray[Any, np.dtype[np.complexfloating]], z_tol: float = 1e8
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute the Majorana points (Bloch sphere coordinates) for a given quantum state."""
    two_j = len(spin_coefficients) - 1

    # build polynomial coefficients a_k
    k_arr = np.arange(len(spin_coefficients))
    binomial_arr = np.sqrt(np.asarray(comb(two_j, k_arr), dtype=np.float64))
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
        _get_majorana_coefficients_from_spin_old(c, z_tol=z_tol)
        for c in spin_coefficients
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
    a = np.asarray(
        p.polyfromroots(finite_roots), dtype=np.complex128
    )  # degree = len(finite)
    # each root at ∞ loses degree in P(z)   →   pad with one zero on the right
    return np.concatenate((a, np.zeros(n_infinity, dtype=a.dtype)))


def _polynomial_to_state(a: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Convert a polynomial representation to quantum state coefficients."""
    two_j = len(a) - 1
    k = np.arange(two_j + 1)
    binomial_weights = np.sqrt(np.asarray(comb(two_j, k), dtype=np.float64))
    c = (a / binomial_weights)[::-1].astype(np.complex128)

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


def _group_majorana_stars_by_index(
    spinor_states: np.ndarray[Any, np.dtype[np.complex128]],
    z_tol: float = 1e8,
) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    # Step 1: Get Majorana stars for each state
    stars_array = majorana_stars(
        spinor_states, z_tol=z_tol
    )  # shape: (n_states, n_stars, 2)

    _, n_stars, _ = stars_array.shape

    # Step 2: Group by star index
    return [stars_array[:, i, :] for i in range(n_stars)]


def _make_majorana_particle_groups(
    spinor_states: np.ndarray[Any, np.dtype[np.complex128]],
    displacements: Sequence[ParticleDisplacement],
    velocities: np.ndarray[Any, np.dtype[np.floating]],
    z_tol: float = 1e8,
) -> list[list[ParticleState]]:
    grouped_stars = _group_majorana_stars_by_index(spinor_states, z_tol=z_tol)
    all_groups: list[list[ParticleState]] = []
    n_groups = len(grouped_stars)
    n_states = grouped_stars[0].shape[0]

    for i in range(n_groups):
        group = grouped_stars[i]  # shape: (n_states, 2)
        group_states = [
            ParticleState(
                spin=CoherentSpin(theta=group[j, 0], phi=group[j, 1]),
                displacement=displacements[j],
                parallel_velocity=velocities[j],
            )
            for j in range(n_states)
        ]
        all_groups.append(group_states)
    return all_groups


def _simulate_majorana_groups(
    grouped_states: list[list[ParticleState]],
    solenoid: Solenoid,
    n_steps: int = 100,
) -> list[TrajectoryList]:
    """Simulate trajectories for each group of ParticleState objects and return a list of TrajectoryList objects."""
    all_trajectorylists: list[TrajectoryList] = []
    for group in grouped_states:
        # Simulate trajectories for each ParticleState in the group
        result = solenoid.simulate_trajectories(group, n_steps=n_steps)
        trajectory_list = result.trajectories
        all_trajectorylists.append(trajectory_list)
    return all_trajectorylists


def _reconstruct_spin_states_from_trajectories(
    grouped_trajectorylists: list[TrajectoryList],
    tol: float = 1e-10,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """
    Given a list of TrajectoryList objects (each for a Majorana star index), reconstruct the spin-J state coefficients for each initial state at every trajectory step.

    Returns an array of shape (n_states, n_steps, 2J+1) with the reconstructed spin-J coefficients
    for each state at each step along the trajectory.
    """
    n_groups = len(grouped_trajectorylists)  # number of Majorana stars (2J)
    n_states = len(grouped_trajectorylists[0])
    n_steps = len(
        grouped_trajectorylists[0][0].spins
    )  # assumes all trajectories have same length

    # Prepare array to hold Majorana stars for each state, step, and star index
    all_stars = np.empty((n_states, n_steps, n_groups, 2), dtype=np.float64)

    for i, trajectory_list in enumerate(grouped_trajectorylists):  # i: star index
        for j, trajectory in enumerate(trajectory_list):  # j: state index
            for k in range(n_steps):  # k: trajectory step
                spin = trajectory.spins[k]
                all_stars[j, k, i, 0] = spin.theta
                all_stars[j, k, i, 1] = spin.phi

    # For each state and each step, reconstruct the 2J+1 spin-J coefficients from the Majorana stars
    reconstructed_states = np.empty(
        (n_states, n_steps, n_groups + 1), dtype=np.complex128
    )
    for j in range(n_states):
        for k in range(n_steps):
            reconstructed_states[j, k] = _stars_to_state(all_stars[j, k], tol=tol)

    return reconstructed_states


def simulate_trajectories_majorana(
    spinor_states: np.ndarray[Any, np.dtype[np.complex128]],
    displacements: Sequence[ParticleDisplacement],
    velocities: np.ndarray[Any],
    solenoid: Solenoid,
    n_steps: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    grouped_states = _make_majorana_particle_groups(
        spinor_states, displacements, velocities
    )
    grouped_trajectories = _simulate_majorana_groups(
        grouped_states, solenoid, n_steps=n_steps
    )
    return _reconstruct_spin_states_from_trajectories(grouped_trajectories, tol=tol)
