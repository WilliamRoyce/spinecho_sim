from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload, override

import numpy as np

from spinecho_sim.state._displacement import (
    ParticleDisplacement,
    ParticleDisplacementList,
)
from spinecho_sim.state._spin import CoherentSpin, Spin
from spinecho_sim.state._state import ParticleState


@dataclass(kw_only=True, frozen=True)
class Trajectory(Sequence[ParticleState]):
    """A trajectory of a particle through the simulation."""

    spins: Spin
    displacement: ParticleDisplacement
    parallel_velocity: float

    @staticmethod
    def from_states(
        states: Iterable[ParticleState],
    ) -> Trajectory:
        """Create a Trajectory from a list of ParticleStates."""
        states = list(states)
        velocities = np.array([state.parallel_velocity for state in states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )
        displacements = [state.displacement for state in states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )
        return Trajectory(
            spins=Spin.from_list([[s.spin for s in states]]),
            displacement=displacements[0],
            parallel_velocity=velocities[0],
        )

    @override
    def __len__(self) -> int:
        return self.spins.shape[0]

    @overload
    def __getitem__(self, index: int) -> ParticleState: ...

    @overload
    def __getitem__(self, index: slice) -> Trajectory: ...

    @override
    def __getitem__(self, index: int | slice) -> ParticleState | Trajectory:
        if isinstance(index, slice):
            return Trajectory(
                spins=self.spins[index],
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )
        return ParticleState(
            spin=self.spins[index],
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(kw_only=True, frozen=True)
class TrajectoryList(Sequence[Trajectory]):
    """A list of trajectories."""

    spins: Spin
    displacements: ParticleDisplacementList
    parallel_velocities: np.ndarray[Any, np.dtype[np.floating]]

    def __post_init__(self) -> None:
        if (
            self.spins.theta.ndim != 2  # noqa: PLR2004
            or self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape != self.displacements.shape
            or self.parallel_velocities.size != self.spins.shape[0]
        ):
            msg = "Spins must be a 2D array, parallel velocities must be a 1D array, and their shapes must match."
            raise ValueError(msg)

    @staticmethod
    def from_trajectories(
        trajectories: Iterable[Trajectory],
    ) -> TrajectoryList:
        """Create a TrajectoryList from a list of Trajectories."""
        trajectories = list(trajectories)
        spins = Spin(
            theta=np.array([t.spins.theta for t in trajectories]),
            phi=np.array([t.spins.phi for t in trajectories]),
        )
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in trajectories
        )
        parallel_velocities = np.array([t.parallel_velocity for t in trajectories])
        return TrajectoryList(
            spins=spins,
            displacements=displacements,
            parallel_velocities=parallel_velocities,
        )

    @override
    def __len__(self) -> int:
        return len(self.parallel_velocities)

    @overload
    def __getitem__(self, index: int) -> Trajectory: ...
    @overload
    def __getitem__(self, index: slice) -> TrajectoryList: ...

    @override
    def __getitem__(self, index: int | slice) -> Trajectory | TrajectoryList:
        if isinstance(index, slice):
            return TrajectoryList(
                spins=self.spins[index],
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return Trajectory(
            spins=Spin(
                theta=self.spins.theta[index],
                phi=self.spins.phi[index],
            ),
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[Trajectory]:
        for i in range(len(self)):
            yield Trajectory(
                spins=Spin.from_list(
                    [[CoherentSpin(theta=self.spins.theta[i], phi=self.spins.phi[i])]]
                ),
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )
