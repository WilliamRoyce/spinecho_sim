from __future__ import annotations

import numpy as np

from spinecho_sim.majorana import (
    majorana_points,
    majorana_points_by_index,
    stars_to_state,
)

if __name__ == "__main__":
    print(majorana_points([1, 0, 1]))
    # print(majorana_points([0, 1, 1]))
    # print("\n", majorana_points_multiple([[1, 0, 1], [0, 1, 1]]))
    print("\n", majorana_points_by_index([[1, 0, 1], [0, 1, 1]]))
    print("\n", stars_to_state([(np.pi / 2, np.pi / 2), (np.pi / 2, 3 * np.pi / 2)]))
