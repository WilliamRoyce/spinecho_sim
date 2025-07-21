"""Module for representing and manipulating spin states."""

from __future__ import annotations

from spinecho_sim.state._displacement import (
    ParticleDisplacement,
    ParticleDisplacementList,
)
from spinecho_sim.state._samples import (
    sample_gaussian_velocities,
    sample_uniform_displacement,
)
from spinecho_sim.state._spin import (
    CoherentSpin,
    CoherentSpinList,
    GenericSpin,
    GenericSpinList,
    Spin,
)
from spinecho_sim.state._state import (
    ParticleState,
)
from spinecho_sim.state._trajectory import (
    Trajectory,
    TrajectoryList,
)

__all__ = [
    "CoherentSpin",
    "CoherentSpinList",
    "GenericSpin",
    "GenericSpinList",
    "ParticleDisplacement",
    "ParticleDisplacementList",
    "ParticleState",
    "Spin",
    "Trajectory",
    "TrajectoryList",
    "sample_gaussian_velocities",
    "sample_uniform_displacement",
]
