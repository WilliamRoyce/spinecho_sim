from __future__ import annotations

import numpy as np

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
