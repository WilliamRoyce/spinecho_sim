from __future__ import annotations

from itertools import starmap
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from spinecho_sim.solenoid import Solenoid, SolenoidTrajectory
from spinecho_sim.state import (
    CoherentSpin,
    CoherentSpinList,
    ParticleDisplacement,
    ParticleState,
    Trajectory,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)


def _get_field(
    z: float,
    displacement: ParticleDisplacement,
    solenoid: Solenoid,
    dz: float = 1e-5,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
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


def simulate_trajectory_cartesean(
    solenoid: Solenoid,
    initial_state: ParticleState,
    n_steps: int = 100,
) -> SolenoidTrajectory:
    """Run the spin echo simulation using configured parameters."""
    z_points = np.linspace(0, solenoid.length, n_steps + 1, endpoint=True)

    gyromagnetic_ratio = -2.04e8  # gyromagnetic ratio (rad s^-1 T^-1)

    def _ds_dx(
        z: float,
        spin: tuple[float, float, float],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        field = _get_field(z, initial_state.displacement, solenoid)
        velocity = initial_state.parallel_velocity

        return (gyromagnetic_ratio / velocity) * np.cross(spin, field)

    sol = solve_ivp(  # type: ignore[return-value]
        fun=_ds_dx,
        t_span=(z_points[0], z_points[-1]),
        y0=initial_state.spin.cartesian,
        t_eval=z_points,
        vectorized=False,
        rtol=1e-8,
    )
    spins = CoherentSpinList.from_spins(
        list(starmap(CoherentSpin.from_cartesian, sol.y.T))  # type: ignore[return-value]
    )
    return SolenoidTrajectory(
        trajectory=Trajectory(
            spins=spins,
            displacement=initial_state.displacement,
            parallel_velocity=initial_state.parallel_velocity,
        ),
        positions=z_points,
    )


def test_simulate_trajectory() -> None:
    particle_velocity = 714

    initial_state = ParticleState(
        spin=CoherentSpin(theta=np.pi / 2, phi=0),
        displacement=sample_uniform_displacement(1, 1.16e-3)[0],
        parallel_velocity=sample_gaussian_velocities(
            1, particle_velocity, 0.225 * particle_velocity
        )[0],
    )

    solenoid = Solenoid.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.01,
    )
    n_steps = 300
    result = solenoid.simulate_trajectory(initial_state, n_steps=n_steps)

    assert result.spins.cartesian.shape == (n_steps + 1, 3)

    expected = simulate_trajectory_cartesean(solenoid, initial_state, n_steps=n_steps)
    np.testing.assert_allclose(
        result.spins.cartesian, expected.spins.cartesian, rtol=1e-5, atol=1e-5
    )
