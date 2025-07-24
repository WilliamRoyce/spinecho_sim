from __future__ import annotations

import numpy as np

from spinecho_sim.state import Spin


def test_spin_states_roundtrip() -> None:
    # spin_states = np.array(
    #     [
    #         [(1.0 + 0.0j) / np.sqrt(2), (0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2)],
    #         [(0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2), (1.0 + 0.0j) / np.sqrt(2)],
    #     ],
    # )
    rng = np.random.default_rng(seed=42)
    spin_states = []
    for _ in range(3):  # Generate 3 random spin states
        coeffs = rng.normal(size=3) + 1j * rng.normal(size=3)
        coeffs /= np.linalg.norm(coeffs)
        spin_states.append(coeffs)
    spin_states = np.array(spin_states, dtype=np.complex128)

    recovered_states = Spin.from_momentum_state(spin_states).as_momentum_states

    # Check if recovered states match original (up to global phase)
    for original_state, recovered_state in zip(
        spin_states, recovered_states, strict=True
    ):
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
