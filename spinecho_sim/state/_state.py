from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from spinecho_sim.state._displacement import ParticleDisplacement
    from spinecho_sim.state._spin import GenericSpin


@dataclass(kw_only=True, frozen=True)
class ParticleState:
    """Represents the state of a particle in the simulation."""

    spin: GenericSpin
    displacement: ParticleDisplacement
    parallel_velocity: np.float64

    def as_coherent(self) -> list[CoherentParticleState]:
        """Convert to a CoherentParticleState."""
        return [
            CoherentParticleState(
                spin=s.as_generic(),
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )
            for s in self.spin.flat_iter()
        ]


@dataclass(kw_only=True, frozen=True)
class CoherentParticleState(ParticleState):
    """Represents the state of a coherent particle in the simulation."""

    def __post_init__(self) -> None:
        """Ensure that the spin is a CoherentSpin."""
        assert self.spin.size == 1, (
            "CoherentParticleState must represent a single coherent spin."
        )
