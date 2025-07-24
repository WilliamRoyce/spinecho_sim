from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from spinecho_sim.state import Spin


def test_spin_states_roundtrip() -> None:
    rng = np.random.default_rng()
    spin_states = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    spin_states /= np.linalg.norm(spin_states, axis=0)[np.newaxis, :]

    recovered_states = Spin.from_momentum_states(spin_states).momentum_states
    assert recovered_states.shape == spin_states.shape, (
        f"Expected shape {spin_states.shape}, got {recovered_states.shape}"
    )

    # Check if recovered states match original (up to global phase)
    for i in range(spin_states.shape[1]):
        original_state = spin_states[:, i]
        recovered_state = recovered_states[:, i]
        assert np.isclose(np.linalg.norm(original_state), 1.0), (
            "Original state is not normalized"
        )
        assert np.isclose(np.linalg.norm(recovered_state), 1.0), (
            "Recovered state is not normalized"
        )
        # Remove global phase
        phase: np.complex128 = np.exp(
            -1j * np.angle(np.vdot(original_state, recovered_state))
        )
        np.testing.assert_allclose(
            original_state,
            recovered_state * phase,
            atol=1e-8,
            err_msg=f"Round-trip failed: {original_state} vs {recovered_state}",
        )


@pytest.mark.parametrize(
    ("momentum_state", "expected_majorana"),
    [
        (
            np.array([1.0, 0.0, 1.0]),
            np.array([[np.pi / 2, np.pi / 2], [np.pi / 2, 3 * np.pi / 2]]),
        ),
        (
            np.array([0.0, 1.0, 1.0]),
            np.array([[2 * np.arctan(1 / np.sqrt(2)), np.pi], [np.pi, 0.0]]),
        ),
        (
            np.array([1.0, 1.0, 0.0]),
            np.array([[0.0, 0.0], [2 * np.arctan(np.sqrt(2)), np.pi]]),
        ),
        (np.array([1.0, 0.0, 0.0]), np.array([[0.0, 0.0], [0.0, np.pi]])),
        (np.array([0.0, 1.0, 0.0]), np.array([[0.0, np.pi], [np.pi, 0.0]])),
        (np.array([0.0, 0.0, 1.0]), np.array([[np.pi, 0.0], [np.pi, 0.0]])),
    ],
)
def test_spin_states_majorana(
    momentum_state: np.ndarray[Any, np.dtype[np.complex128]],
    expected_majorana: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    momentum_state /= np.linalg.norm(momentum_state)

    spin_state = Spin.from_momentum_states(np.array([momentum_state]).T)[0]
    np.testing.assert_array_almost_equal(
        expected_majorana[..., 0],
        spin_state.theta,
        err_msg="Majorana points do not match expected theta values",
    )
    np.testing.assert_array_almost_equal(
        expected_majorana[..., 1],
        spin_state.phi,
        err_msg="Majorana points do not match expected phi values",
    )
