from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, override

import numpy as np

from spinecho_sim.majorana import stars_to_state
from spinecho_sim.majorana._companion_helper import majorana_stars


class Spin(Sequence[Any]):
    """A class representing a collection of lists of CoherentSpin objects."""

    def __init__(self, spins: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        self._spins = spins
        assert self._spins.shape[-1] == 2  # Ensure last dimension is 2 (for theta, phi)

    @override
    def __len__(self) -> int:
        """Total number of CoherentSpin objects."""
        return sum(len(group) for group in self._spins)

    def item(self, index: int) -> CoherentSpin:
        """Iterate over all CoherentSpin objects."""
        thetas = self._spins[..., 0]
        phis = self._spins[..., 1]
        return CoherentSpin(theta=thetas.item(index), phi=phis.item(index))

    @override
    def __getitem__(self, index: int | slice) -> CoherentSpin | Spin:
        # Flatten the array to shape (n_total_spins, 2)
        flat = self._spins.reshape(-1, 2)
        if isinstance(index, int):
            theta, phi = flat[index]
            return CoherentSpin(theta=theta, phi=phi)
        # For slice, return a new Spin with the sliced flat array
        sliced = flat[index]
        # Reshape to (1, n_spins, 2) for consistency
        return Spin(sliced.reshape(1, -1, 2))

    def group(self, index: int) -> Spin:
        """Return the spins in a specific group."""
        return Spin(self._spins[index][np.newaxis, ...])

    @override
    def __iter__(self) -> Iterator[Spin]:
        """Iterate over all CoherentSpin objects (flattened)."""
        for group in self._spins:
            yield from group

    def flat_iter(self) -> Iterator[CoherentSpin]:
        """Iterate over all CoherentSpin objects in a flat manner."""
        for i in range(self.size):
            yield self.item(i)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape: (number of groups, spins per group...)."""
        return self._spins.shape[:-1]

    @property
    def size(self) -> int:
        """Return the total number of spins."""
        return np.prod(self.shape).item()

    @staticmethod
    def from_momentum_state(
        spin_coefficients: np.ndarray[Any, np.dtype[np.complex128]],
    ) -> Spin:
        """Create a Spin from a momentum state represented by complex coefficients."""
        if spin_coefficients.ndim == 1:
            spin_coefficients = spin_coefficients[np.newaxis, :]
        stars_array = majorana_stars(spin_coefficients)  # shape: (n_groups, n_stars, 2)
        # Convert each group of (theta, phi) pairs to CoherentSpin objects
        # Convert to ndarray of shape (n_groups, n_stars, 2)
        spins_array = np.array(stars_array, dtype=np.float64)
        return Spin(spins_array)

    @staticmethod
    def from_list(
        spins: list[list[CoherentSpin]],
    ) -> Spin:
        """Create a Spin from a nested list of CoherentSpin objects."""
        if not spins:
            msg = "Input list must not be empty."
            raise ValueError(msg)
        group_lengths = [len(group) for group in spins]
        if len(set(group_lengths)) != 1:
            msg = "All groups must have the same number of spins."
            raise ValueError(msg)

        """Create a Spin from a nested list of CoherentSpin objects."""
        spins_array = np.array(
            [[(spin.theta, spin.phi) for spin in group] for group in spins],
            dtype=np.float64,
        )
        return Spin(spins_array)


class CoherentSpin(Spin):
    """A class representing a single coherent spin with theta and phi angles."""

    def __init__(
        self,
        theta: float,
        phi: float,
    ) -> None:
        self._theta = theta
        self._phi = phi
        if not (0 <= self._theta <= np.pi):
            msg = f"Theta must be in [0, π]. Got {self._theta}."
            raise ValueError(msg)
        if not (0 <= self._phi < 2 * np.pi):
            msg = f"Phi must be in [0, 2π). Got {self._phi}."
            raise ValueError(msg)

    def as_momentum_state(self) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """Convert the coherent spin representation to a momentum state."""
        star_array = np.array([[self._theta, self._phi]], dtype=np.float64)
        return stars_to_state(star_array)

    @property
    def theta(self) -> float:
        """Return the theta angle of the spin."""
        return self._theta

    @property
    def phi(self) -> float:
        """Return the phi angle of the spin."""
        return self._phi


group1 = [CoherentSpin(theta=0.5, phi=1.0), CoherentSpin(theta=1.0, phi=2.0)]
group2 = [CoherentSpin(theta=0.2, phi=0.3), CoherentSpin(theta=1.0, phi=2.0)]
spin_collection = Spin.from_list([group1, group2])

for s in spin_collection.flat_iter():
    print(f"Theta: {s.theta}, Phi: {s.phi}")
print("\n")

spin_states = np.array(
    [
        [(1.0 + 0.0j) / np.sqrt(2), (0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2)],
        [(0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2), (1.0 + 0.0j) / np.sqrt(2)],
    ],
)

spin_from_state = Spin.from_momentum_state(spin_states)
for s in spin_from_state.flat_iter():
    print(f"Theta: {s.theta}, Phi: {s.phi}")
for s in spin_from_state.group(1).flat_iter():
    print(f"Theta: {s.theta}, Phi: {s.phi}")


spin_state = np.array(
    [(1.0 + 0.0j) / np.sqrt(2), (0.0 + 0.0j), (1.0 + 0.0j) / np.sqrt(2)],
    dtype=np.complex128,
)

coherent_spin = Spin.from_momentum_state(spin_state)
print(f"\nTheta: {coherent_spin.item(0).theta}, Phi: {coherent_spin.item(0).phi}")
print(f"Theta: {coherent_spin.item(1).theta}, Phi: {coherent_spin.item(1).phi}")
