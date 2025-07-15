from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
from spinecho_sim.majorana import (
    majorana_points_by_index,
    stars_to_states,
)


def test_spin_states_roundtrip() -> None:
    spin_states = np.array(
        [
            [(1.0 + 0.0j) / np.sqrt(2), (0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2)],
            [(0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2), (1.0 + 0.0j) / np.sqrt(2)],
        ],
        dtype=np.complex128,
    )
    # Convert to Majorana points
    majorana_points = majorana_points_by_index(spin_states)
    # Convert back to states
    recovered_states = stars_to_states(majorana_points)
    # Check if recovered states match original (up to global phase)
    for orig, rec in zip(spin_states, recovered_states, strict=False):
        # Normalize both
        orig_norm: NDArray[np.complexfloating] = orig / np.linalg.norm(orig)
        rec_norm: NDArray[np.complexfloating] = rec / np.linalg.norm(rec)
        # Remove global phase
        phase: np.complex128 = np.exp(-1j * np.angle(np.vdot(orig_norm, rec_norm)))
        rec_norm *= phase
        assert np.allclose(orig_norm, rec_norm, atol=1e-8), (
            f"Roundtrip failed: {orig_norm} vs {rec_norm}"
        )
