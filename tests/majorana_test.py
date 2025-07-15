from __future__ import annotations

import numpy as np

from spinecho_sim.majorana import (
    majorana_points_by_index,
    stars_to_states,
)

if __name__ == "__main__":
    spin_states = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.complex128)
    result = majorana_points_by_index(spin_states)
    print("\n", result)
    multi_stars = np.array(
        [
            [[np.pi / 2, np.pi / 2], [np.pi / 2, 3 * np.pi / 2]],
            [[np.pi, 0.0], [np.pi / 2, 0.0]],
        ],
        dtype=np.float64,
    )
    print("\n", stars_to_states(result))
