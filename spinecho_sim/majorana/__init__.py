"""Module for simulating and plotting solenoid magnetic fields and particle trajectories."""

from __future__ import annotations

from spinecho_sim.majorana._companion_helper import (
    majorana_stars,
)
from spinecho_sim.majorana._majorana import (
    majorana_stars_old,
    simulate_trajectories_majorana,
    stars_to_states,
)

__all__ = [
    "majorana_stars",
    "majorana_stars_old",
    "simulate_trajectories_majorana",
    "stars_to_states",
]
