"""Module for simulating and plotting solenoid magnetic fields and particle trajectories."""

from __future__ import annotations

from spinecho_sim.majorana._companion_helper import (
    majorana_stars_new,
)
from spinecho_sim.majorana._majorana import (
    majorana_stars_old,
    stars_to_states,
)

__all__ = [
    "majorana_stars_new",
    "majorana_stars_old",
    "stars_to_states",
]
