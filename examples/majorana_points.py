from __future__ import annotations

import numpy as np

from spinecho_sim.majorana import (
    majorana_points_by_index,
    stars_to_state,
)

if __name__ == "__main__":
    spin_states = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.complex128)
    result = majorana_points_by_index(spin_states)
    print("\n", result)
    print("\n", stars_to_state([(np.pi / 2, np.pi / 2), (np.pi / 2, 3 * np.pi / 2)]))
