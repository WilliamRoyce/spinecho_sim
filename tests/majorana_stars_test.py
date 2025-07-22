from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from spinecho_sim.state._companion_helper import majorana_stars


@pytest.mark.parametrize(
    ("spin_state", "expected_majorana"),
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
