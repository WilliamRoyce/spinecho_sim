from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinecho_sim.state._displacement import ParticleDisplacement
    from spinecho_sim.state._spin import GenericSpin


@dataclass(kw_only=True, frozen=True)
class ParticleState:
    """Represents the state of a particle in the simulation."""

    spin: GenericSpin
    displacement: ParticleDisplacement
    parallel_velocity: float
