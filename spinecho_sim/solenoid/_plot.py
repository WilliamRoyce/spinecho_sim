from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from spinecho_sim.util import get_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.lines import Line2D

    from spinecho_sim.solenoid._solenoid import SolenoidSimulationResult


def plot_spin_component(
    result: SolenoidSimulationResult,
    idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    spins = result.spins.cartesian[idx]
    average_spins = np.average(spins, axis=0)

    label = [
        r"$\langle S_x \rangle$",
        r"$\langle S_y \rangle$",
        r"$\langle S_z \rangle$",
    ][idx]
    (average_line,) = ax.plot(positions, average_spins, label=label)
    color = average_line.get_color()
    ax.plot(
        positions,
        np.swapaxes(spins, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(spins, axis=0) / np.sqrt(len(spins))
    ax.fill_between(
        positions,
        np.clip(average_spins - std_spins, -1, 1).ravel(),
        np.clip(average_spins + std_spins, -1, 1).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
    )

    ax.set_xlim(positions[0], positions[-1])
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel("Spin components")

    return fig, ax


def plot_spin_intensity(
    result: SolenoidSimulationResult, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes, Line2D]:
    fig, ax = get_figure(ax)

    positions = result.positions
    intensity = np.average(
        result.spins.cartesian[0] ** 2 + result.spins.cartesian[1] ** 2, axis=0
    )
    (line,) = ax.plot(
        positions,
        intensity,
        label=r"$I_\parallel =\langle S_x\rangle^2+\langle S_y\rangle^2$",
        color="black",
        linestyle="--",
    )

    ax.set_xlim(positions[0], positions[-1])
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel("Spin Intensity")

    return fig, ax, line


def plot_spin_components(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in range(3):
        plot_spin_component(result, idx, ax=ax)

    plot_spin_intensity(result, ax=ax.twinx())
    return fig, ax


def plot_spin_phi(
    result: SolenoidSimulationResult, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)
    positions = result.positions
    phi = np.unwrap(result.spins.phi, axis=1) / np.pi
    average_phi = np.average(phi, axis=0)

    (average_line,) = ax.plot(positions, average_phi)
    color = average_line.get_color()
    average_line.set_label(r"$\langle \phi \rangle$")

    ax.plot(
        positions,
        np.swapaxes(phi, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(phi, axis=0) / np.sqrt(len(phi))
    ax.fill_between(
        positions,
        (average_phi - std_spins).ravel(),
        (average_phi + std_spins).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
    )
    ax.set_xlim(positions[0], positions[-1])
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel(r"Phase Angles (radians/$\pi$)")
    return fig, ax


def plot_spin_theta(
    result: SolenoidSimulationResult, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)
    positions = result.positions
    theta = np.unwrap(result.spins.theta, axis=1) / np.pi
    average_theta = np.average(theta, axis=0)

    (average_line,) = ax.plot(positions, average_theta)
    color = average_line.get_color()
    average_line.set_label(r"$\langle \theta \rangle$")

    ax.plot(
        positions,
        np.swapaxes(theta, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(theta, axis=0) / np.sqrt(len(theta))
    ax.fill_between(
        positions,
        (average_theta - std_spins).ravel(),
        (average_theta + std_spins).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
    )
    ax.set_xlim(positions[0], positions[-1])
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel(r"Phase Angles (radians/$\pi$)")
    return fig, ax


def plot_spin_angles(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_spin_theta(result, ax=ax)
    plot_spin_phi(result, ax=ax)

    return fig, ax


def plot_spin_state(
    result: SolenoidSimulationResult,
    idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    states = result.spins.as_momentum_states[idx]
    average_state_real = np.average(states.real, axis=0)
    average_state_imag = np.average(states.imag, axis=0)

    n_stars = result.spins.n_stars
    s = n_stars / 2
    ms_values = np.linspace(-s, s, n_stars + 1, endpoint=True)
    ms_labels = [
        rf"$|m_S={m:.1f} \rangle$"
        if not m.is_integer()
        else rf"$|m_S={int(m)} \rangle$"
        for m in ms_values
    ]

    # Plot real part
    (real_line,) = ax.plot(
        positions, average_state_real, label=f"{ms_labels[idx]}" + "(Re)"
    )
    color_real = real_line.get_color()
    ax.plot(positions, -average_state_real, color=color_real)
    ax.plot(
        positions,
        np.swapaxes(states.real, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color_real,
    )
    ax.plot(
        positions,
        np.swapaxes(-states.real, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color_real,
    )

    # Plot imaginary part
    (imag_line,) = ax.plot(
        positions, average_state_imag, label=f"{ms_labels[idx]}" + "(Im)"
    )
    color_imag = imag_line.get_color()
    ax.plot(positions, -average_state_imag, color=color_imag)
    ax.plot(
        positions,
        np.swapaxes(states.imag, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color_imag,
    )
    ax.plot(
        positions,
        np.swapaxes(-states.imag, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color_imag,
    )

    # Standard error of the mean for real part
    std_states_real = np.std(states.real, axis=0) / np.sqrt(len(states))
    ax.fill_between(
        positions,
        np.clip(average_state_real - std_states_real, -1, 1).ravel(),
        np.clip(average_state_real + std_states_real, -1, 1).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color_real,
    )
    ax.fill_between(
        positions,
        np.clip(-average_state_real - std_states_real, -1, 1).ravel(),
        np.clip(-average_state_real + std_states_real, -1, 1).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color_real,
    )

    # Standard error of the mean for imaginary part
    std_states_imag = np.std(states.imag, axis=0) / np.sqrt(len(states))
    ax.fill_between(
        positions,
        np.clip(average_state_imag - std_states_imag, -1, 1).ravel(),
        np.clip(average_state_imag + std_states_imag, -1, 1).ravel(),
        alpha=0.2,
        linestyle=":",
        color=color_imag,
    )
    ax.fill_between(
        positions,
        np.clip(-average_state_imag - std_states_imag, -1, 1).ravel(),
        np.clip(-average_state_imag + std_states_imag, -1, 1).ravel(),
        alpha=0.2,
        linestyle=":",
        color=color_imag,
    )

    plot_state_intensity(result, idx=idx, ax=ax.twinx())

    ax.set_ylabel(ms_labels[idx])
    ax.legend(loc="upper right", fontsize="small")
    ax.set_xlim(positions[0], positions[-1])
    ax.set_ylim(-1, 1)

    return fig, ax


def plot_spin_states(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    n_stars = result.spins.n_stars
    fig, axes = plt.subplots(n_stars + 1, 1, figsize=(10, 6), sharex=True)

    for idx, ax in enumerate(axes):
        plot_spin_state(result, idx, ax=ax)
    axes[-1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout()
    return fig, axes


def plot_state_intensity(
    result: SolenoidSimulationResult, idx: int, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes, Line2D]:
    fig, ax = get_figure(ax)

    positions = result.positions
    states = result.spins.as_momentum_states[idx]
    average_state_abs = np.average(np.abs(states), axis=0)

    (line,) = ax.plot(
        positions,
        average_state_abs,
        color="black",
        linestyle="--",
    )

    ax.set_xlim(positions[0], positions[-1])
    ax.set_ylim(-1, 1)

    return fig, ax, line
