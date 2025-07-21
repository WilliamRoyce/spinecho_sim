from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, overload, override

import numpy as np

from spinecho_sim.majorana import majorana_stars, stars_to_state


class Spin[S: tuple[int, ...]](Sequence[Any]):
    """A class representing a collection of lists of CoherentSpin objects."""

    def __init__(self, spins: np.ndarray[tuple[*S, int], np.dtype[np.float64]]) -> None:
        self._spins = spins
        # Spins are stored as an ndarray of shape (..., 2)
        # Where spin[..., 0] is theta and spin[..., 1] is phi
        assert self._spins.shape[-1] == 2  # noqa: PLR2004
        assert self._spins.ndim > 1, "Spins must have at least 2 dimensions."

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the spins array."""
        return self._spins.ndim - 1

    @override
    def __len__(self) -> int:
        """Total number of CoherentSpin objects."""
        return self.shape[0]

    @property
    def theta(self) -> np.ndarray[tuple[*S], np.dtype[np.floating]]:
        """Return the theta angle of the spin."""
        return self._spins[..., 0]

    @property
    def phi(self) -> np.ndarray[tuple[*S], np.dtype[np.floating]]:
        """Return the phi angle of the spin."""
        return self._spins[..., 1]

    def item(self, index: int) -> CoherentSpin:
        """Iterate over all CoherentSpin objects."""
        return CoherentSpin(theta=self.theta.item(index), phi=self.phi.item(index))

    @overload
    def __getitem__(self: Spin[tuple[int]], index: int) -> CoherentSpin: ...

    @overload
    def __getitem__[*S_](
        self: Spin[tuple[int, *S_]],  # type: ignore[override]
        index: int,
    ) -> Spin[tuple[*S_]]: ...  # type: ignore[override]

    @overload
    def __getitem__[*S_](
        self: Spin[tuple[int, *S_]],  # type: ignore[override]
        index: slice,
    ) -> Spin[tuple[int, *S_]]: ...  # type: ignore[override]

    @override
    def __getitem__(self, index: int | slice) -> CoherentSpin | Spin[tuple[Any, ...]]:
        """Get a single CoherentSpin object by index."""
        if isinstance(index, int) and self._spins.ndim == 2:  # noqa: PLR2004
            theta, phi = self._spins[index]
            return CoherentSpin(theta=theta, phi=phi)

        return Spin(self._spins[index])

    @override
    def __iter__[*S_](self: Spin[tuple[int, *S_]]) -> Iterator[Spin[tuple[*S_]]]:  # type: ignore[override]
        """Iterate over all CoherentSpin objects (flattened)."""
        for group in self._spins:
            yield from group

    def flat_iter(self) -> Iterator[CoherentSpin]:
        """Iterate over all CoherentSpin objects in a flat manner."""
        for i in range(self.size):
            yield self.item(i)

    @property
    def shape(self) -> tuple[*S]:
        """Return the shape of the spin list."""
        return self._spins.shape[:-1]

    @property
    def size(self) -> int:
        """Return the total number of spins."""
        return np.prod(self.shape).item()

    @staticmethod
    def from_momentum_state(
        spin_coefficients: np.ndarray[Any, np.dtype[np.complex128]],
    ) -> Spin[tuple[int]]:
        """Create a Spin from a momentum state represented by complex coefficients."""
        if spin_coefficients.ndim == 1:
            spin_coefficients = spin_coefficients[np.newaxis, :]
        stars_array = majorana_stars(spin_coefficients)  # shape: (n_groups, n_stars, 2)
        # Convert each group of (theta, phi) pairs to CoherentSpin objects
        # Convert to ndarray of shape (n_groups, n_stars, 2)
        spins_array = np.array(stars_array, dtype=np.float64)
        return Spin(spins_array)

    @staticmethod
    def from_iter[S_: tuple[int, ...]](
        spins: Iterable[Spin[S_]],
    ) -> Spin[tuple[int, *S_]]:
        """Create a Spin from a nested list of CoherentSpin objects."""
        spins_array = np.array(
            [(spin.theta, spin.phi) for spin in spins],
            dtype=np.float64,
        )
        return Spin(spins_array)


class CoherentSpin(Spin[tuple[()]]):
    """A class representing a single coherent spin with theta and phi angles."""

    def __init__(
        self,
        theta: float,
        phi: float,
    ) -> None:
        self._theta = theta
        self._phi = phi

    def as_momentum_state(self) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
        """Convert the coherent spin representation to a momentum state."""
        star_array = np.array([[self._theta, self._phi]], dtype=np.float64)
        return stars_to_state(star_array)

    @property
    @override
    def theta(self) -> np.ndarray[tuple[()], np.dtype[np.floating]]:
        """Return the theta angle of the spin."""
        return np.array(self._theta)

    @property
    @override
    def phi(self) -> np.ndarray[tuple[()], np.dtype[np.floating]]:
        """Return the phi angle of the spin."""
        return np.array(self._phi)

    @property
    @override
    def shape(self) -> tuple[()]:
        """Return the shape of a single coherent spin."""
        return ()

    def as_generic(self) -> Spin[tuple[int]]:
        """Return a generic Spin representation of this coherent spin."""
        return Spin.from_iter([self])


type GenericSpin = Spin[tuple[int]]
type GenericSpinList = Spin[tuple[int, int]]
type CoherentSpinList = Spin[tuple[int]]
