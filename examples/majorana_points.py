from __future__ import annotations

import numpy as np

from spinecho_sim.majorana import (
    majorana_points_by_index,
    stars_to_state,
)

if __name__ == "__main__":
    print("\n", majorana_points_by_index([np.array([1, 0, 1]), np.array([0, 1, 1])]))
    print("\n", stars_to_state([(np.pi / 2, np.pi / 2), (np.pi / 2, 3 * np.pi / 2)]))
