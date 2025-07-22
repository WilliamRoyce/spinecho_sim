from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np
import pytest
from scipy.special import comb  # type: ignore[import]

from spinecho_sim.state import Spin
from spinecho_sim.state._companion_helper import majorana_stars
from spinecho_sim.state._majorana_representation import stars_to_states

NUM_SPIN_PARAMS = 2  # Number of parameters per spin (theta, phi)


def get_polynomial_product(
    states: Spin[tuple[int]],
) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
    """
    Compute the coefficients of product polynomial.

    P(z) = ∏ (a_i z + b_i), returned as a vector of coefficients.
    """
    a = np.cos(states.theta / 2)
    b = np.sin(states.theta / 2) * np.exp(1j * states.phi)
    factors = np.stack([a, b], axis=1)  # shape (N, 2)
    return reduce(np.convolve, factors)


def majorana_polynomial_components(
    states: Spin[tuple[int]],
) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
    """
    Compute A_m using the polynomial representation.

    Returns
    -------
    A : np.ndarray, shape (N+1,)
        Coefficients A_m for m = -j to j
    """
    coefficients = get_polynomial_product(states)
    k = np.arange(states.size + 1)
    binomial_weights = (
        np.sqrt(np.asarray(comb(states.size, k), dtype=np.float64)) * (-1) ** k
    )
    state = coefficients / binomial_weights
    return state / np.linalg.norm(state)


def stars_to_state_B(
    stars: np.ndarray[tuple[*tuple[int], int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
    """Convert a list of Majorana stars (theta, phi) to the corresponding quantum state coefficients."""
    assert stars.ndim == NUM_SPIN_PARAMS, (
        "Stars must be a 2D array of shape (n_stars, 2)"
    )
    states = Spin(stars)
    return majorana_polynomial_components(states)


spin_states = np.array(
    [
        [(1.0 + 0.0j) / np.sqrt(2), (0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2)],
        [(0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2), (1.0 + 0.0j) / np.sqrt(2)],
    ],
)
stars = majorana_stars(spin_states)
print(stars)
print(stars_to_states(stars))

print("\n", stars[1])
print(stars_to_state_B(stars[1]))


@pytest.mark.parametrize(
    ("spin_state", "expected_majorana"),
    [
        (
            np.array([1.0, 0.0, 1.0]),
            np.array([[np.pi / 2, np.pi / 2], [np.pi / 2, 3 * np.pi / 2]]),
        ),
        (
            np.array([0.0, 1.0, 1.0]),  # Not working
            np.array([[2 * np.arctan(1 / np.sqrt(2)), np.pi], [np.pi, 0.0]]),
        ),
        (
            np.array([1.0, 1.0, 0.0]),  # Not working
            np.array([[0.0, 0.0], [2 * np.arctan(np.sqrt(2)), np.pi]]),
        ),
        (np.array([1.0, 0.0, 0.0]), np.array([[0.0, 0.0], [0.0, np.pi]])),
        (np.array([0.0, 1.0, 0.0]), np.array([[0.0, np.pi], [np.pi, 0.0]])),
        (np.array([0.0, 0.0, 1.0]), np.array([[np.pi, 0.0], [np.pi, 0.0]])),
    ],
)
def test_spin_states_roundtrip(
    spin_state: np.ndarray[Any, np.dtype[np.complex128]],
    expected_majorana: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    spin_state /= np.linalg.norm(spin_state)

    majorana_point = majorana_stars(np.array([spin_state]))[0]
    np.testing.assert_array_almost_equal(
        expected_majorana,
        majorana_point,
        err_msg=f"Majorana points do not match expected: {majorana_point} vs {expected_majorana}",
    )

    recovered_state = stars_to_state_B(majorana_point)

    assert np.isclose(np.linalg.norm(spin_state), 1.0), (
        "Original state is not normalized"
    )
    assert np.isclose(np.linalg.norm(recovered_state), 1.0), (
        "Recovered state is not normalized"
    )
    phase = np.exp(-1j * np.angle(np.vdot(spin_state, recovered_state)))

    np.testing.assert_allclose(
        spin_state,
        recovered_state * phase,
        atol=1e-8,
        err_msg=f"Round-trip failed: {spin_state} vs {recovered_state}",
    )
