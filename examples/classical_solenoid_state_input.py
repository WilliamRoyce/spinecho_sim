from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim import (
    ParticleState,
    Solenoid,
)
from spinecho_sim.solenoid import plot_spin_states
from spinecho_sim.state import (
    Spin,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)

if __name__ == "__main__":
    particle_velocity = 714
    num_spins = 1
    spin_states = np.array(
        [
            [(1.0 + 0.0j) / np.sqrt(2), (1.0 + 0.0j) / np.sqrt(2)],
        ],
    )
    initial_states = [
        ParticleState(
            spin=Spin.from_momentum_state(spin_states),
            displacement=displacement,
            parallel_velocity=velocity,
        )
        for displacement, velocity in zip(
            sample_uniform_displacement(num_spins, 1.16e-3),
            sample_gaussian_velocities(
                num_spins, particle_velocity, 0.225 * particle_velocity
            ),
            strict=True,
        )
    ]

    solenoid = Solenoid.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.1,
    )
    result = solenoid.simulate_trajectories(initial_states, n_steps=1000)

    fig, ax = plot_spin_states(result)

    plt.show()
