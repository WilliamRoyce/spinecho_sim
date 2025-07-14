"""Core simulation functionality for spin echo experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

from spinecho_sim.state import (
    CoherentSpin,
    CoherentSpinList,
    ParticleDisplacement,
    ParticleDisplacementList,
    ParticleState,
    Trajectory,
    TrajectoryList,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass(kw_only=True, frozen=True)
class Solenoid:
    """Dataclass representing a solenoid with its parameters."""

    length: float
    field: Callable[[float], np.ndarray[Any, np.dtype[np.floating]]]

    @classmethod
    def with_uniform_z(cls, length: float, strength: float) -> Solenoid:
        """Build a solenoid with a uniform field along the z-axis."""
        return cls(length=length, field=lambda _z: np.array([0.0, 0.0, strength]))

    @classmethod
    def with_nonuniform_z(
        cls, length: float, strength: Callable[[float], float]
    ) -> Solenoid:
        """Build a solenoid with a non-uniform field along the z-axis."""
        return cls(length=length, field=lambda z: np.array([0.0, 0.0, strength(z)]))

    @classmethod
    def from_experimental_parameters(
        cls, *, length: float, magnetic_constant: float, current: float
    ) -> Solenoid:
        """Build a solenoid from an experimental magnetic constant and current."""
        b_z = np.pi * magnetic_constant * current / (2 * length)
        return cls.with_nonuniform_z(
            length=length,
            strength=lambda z: b_z * np.sin(np.pi * z / length) ** 2,
        )

    def simulate_angle_trajectory(
        self,
        initial_state: ParticleState,
        n_steps: int = 100,
    ) -> SolenoidTrajectory:
        """Run the spin echo simulation using configured parameters."""
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)

        gyromagnetic_ratio = 2.04e8  # gyromagnetic ratio (rad s^-1 T^-1)

        def _dangles_dx(
            z: float, angles: tuple[float, float]
        ) -> NDArray[np.floating[Any]]:
            theta, phi = angles
            # Reconstruct spin vector from angles (unit vector)
            s_vec = np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )
            field = _get_field(z, initial_state.displacement, self)
            velocity = initial_state.parallel_velocity

            dsdx = (gyromagnetic_ratio / velocity) * np.cross(s_vec, field)
            s = np.linalg.norm(s_vec)
            # Chain rule for θ and φ
            dtheta = (dsdx[2] * s - s_vec[2] * np.dot(s_vec, dsdx) / s) / (
                s**2 * np.sin(theta)
            )
            dphi = (s_vec[0] * dsdx[1] - s_vec[1] * dsdx[0]) / (
                s**2 * np.sin(theta) ** 2
            )
            return np.array([dtheta, dphi])

        # Convert initial spin to angles
        s0 = initial_state.spin.cartesian
        theta0 = np.arccos(s0[2] / np.linalg.norm(s0))
        phi0 = np.arctan2(s0[1], s0[0])
        y0 = np.array([theta0, phi0])

        sol = solve_ivp(  # type: ignore[return-value]
            fun=_dangles_dx,
            t_span=(z_points[0], z_points[-1]),
            y0=y0,
            t_eval=z_points,
            vectorized=False,
            rtol=1e-8,
        )
        spins = CoherentSpinList.from_spins(
            [CoherentSpin(theta=ang[0], phi=ang[1]) for ang in sol.y.T]  # type: ignore[return-value]
        )
        return SolenoidTrajectory(
            trajectory=Trajectory(
                spins=spins,
                displacement=initial_state.displacement,
                parallel_velocity=initial_state.parallel_velocity,
            ),
            positions=z_points,
        )

    def simulate_angle_trajectories(
        self,
        initial_states: list[ParticleState],
        n_steps: int = 100,
    ) -> SolenoidSimulationResult:
        """Run a solenoid simulation for multiple initial states."""
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)
        return SolenoidSimulationResult(
            trajectories=TrajectoryList.from_trajectories(
                [
                    self.simulate_angle_trajectory(state, n_steps).trajectory
                    for state in initial_states
                ]
            ),
            positions=z_points,
        )


@dataclass(kw_only=True, frozen=True)
class SolenoidTrajectory:
    """Represents the trajectory of a particle as it moves through the simulation."""

    trajectory: Trajectory
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def spins(self) -> CoherentSpinList:
        """The spin components from the simulation states."""
        return self.trajectory.spins

    @property
    def displacement(self) -> ParticleDisplacement:
        """The displacement of the particle at the end of the trajectory."""
        return self.trajectory.displacement


@dataclass(kw_only=True, frozen=True)
class SolenoidSimulationResult:
    """Represents the result of a solenoid simulation."""

    trajectories: TrajectoryList
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def spins(self) -> CoherentSpinList:
        """Extract the spin components from the simulation states."""
        return self.trajectories.spins

    @property
    def displacements(self) -> ParticleDisplacementList:
        """Extract the displacements from the simulation states."""
        return self.trajectories.displacements


def _get_field(
    z: float,
    displacement: ParticleDisplacement,
    solenoid: Solenoid,
    dz: float = 1e-5,
) -> NDArray[np.floating[Any]]:
    if displacement.r == 0:
        return solenoid.field(z)

    # Assuming that there is no current in the solenoid, we can
    # calculate the field at any point using grad.B = 0. We do this
    b_z_values = [solenoid.field(zi)[2] for zi in (z - dz, z, z + dz)]

    b0_p = (b_z_values[1] - b_z_values[-1]) / (2 * dz)
    b0_pp = (b_z_values[2] - 2 * b_z_values[1] + b_z_values[0]) / (dz**2)

    b_r = -0.5 * displacement.r * b0_p
    db_z = -0.25 * displacement.r**2 * b0_pp

    return np.array(
        [
            b_r * np.cos(displacement.theta),
            b_r * np.sin(displacement.theta),
            b_z_values[1] + db_z,
        ]
    )
