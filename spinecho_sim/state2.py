from __future__ import annotations

from typing import Any, override

import numpy as np

from spinecho_sim.majorana import (
    majorana_stars,
    stars_to_state,
)


class Spin:
    """A class representing a spin with theta and phi angles."""

    _majorana_theta: np.ndarray[tuple[int], np.dtype[np.float64]]
    _majorana_phi: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __init__(
        self,
        majorana_theta: np.ndarray[Any, np.dtype[np.float64]],
        majorana_phi: np.ndarray[Any, np.dtype[np.float64]],
    ) -> None:
        self._majorana_theta = majorana_theta.ravel()
        self._majorana_phi = majorana_phi.ravel()

    def as_momentum_state(self) -> np.ndarray[Any, np.dtype[np.complex128]]:
        """Convert the majorana star representation to a momentum state."""
        stars_array = np.column_stack(
            (self._majorana_theta, self._majorana_phi)
        ).astype(np.float64)
        state = stars_to_state(stars_array)
        return np.asarray(state, dtype=np.complex128)

    @property
    def n_stars(self) -> int:
        """Get the number of majorana stars represented by this spin."""
        return len(self._majorana_theta)

    def as_majorana_stars(self) -> list[CoherentSpin]:
        """Get the Majorana stars as a 2D array of shape (n_stars, 2)."""
        return [
            CoherentSpin(theta=theta, phi=phi, n_stars=1)
            for theta, phi in zip(self._majorana_theta, self._majorana_phi, strict=True)
        ]

    @staticmethod
    def from_momentum_state(momentum: np.ndarray[Any, np.dtype[np.complex128]]) -> Spin:
        """Create a Spin from a momentum state."""
        stars = majorana_stars(momentum)  # shape: (n_states, n_stars, 2)
        # Expect only one state, use stars[0]
        theta = stars[0, :, 0]
        phi = stars[0, :, 1]
        return Spin(
            majorana_theta=np.array(theta, dtype=np.float64),
            majorana_phi=np.array(phi, dtype=np.float64),
        )

    @staticmethod
    def from_majorana_stars(stars: list[CoherentSpin]) -> Spin:
        """Create a Spin from Majorana stars."""
        return Spin(
            majorana_theta=np.array(
                [(star.theta,) * star.n_stars for star in stars]
            ).ravel(),
            majorana_phi=np.array(
                [(star.phi,) * star.n_stars for star in stars]
            ).ravel(),
        )


class CoherentSpin(Spin):
    """A class representing a coherent spin with theta and phi angles."""

    def __init__(
        self,
        theta: float,
        phi: float,
        n_stars: int = 1,
    ) -> None:
        self._n_stars = n_stars
        self._theta = theta
        self._phi = phi

    @override
    def as_momentum_state(self) -> np.ndarray[Any, np.dtype[np.complex128]]:
        """Convert the coherent spin representation to a momentum state."""
        thetas = np.full(self._n_stars, self._theta, dtype=np.float64)
        phis = np.full(self._n_stars, self._phi, dtype=np.float64)
        stars_array = np.column_stack((thetas, phis))
        state = stars_to_state(stars_array)
        return np.asarray(state, dtype=np.complex128)

    @override
    def as_majorana_stars(self) -> list[CoherentSpin]:
        """Get the Majorana stars as a list of CoherentSpin."""
        return [CoherentSpin(self._theta, self._phi, n_stars=self._n_stars)]

    @property
    @override
    def n_stars(self) -> int:
        """Get the number of majorana stars represented by this coherent spin."""
        return self._n_stars

    @property
    def theta(self) -> float:
        """Get the polar angle of the spin vector."""
        return self._theta

    @property
    def phi(self) -> float:
        """Get the azimuthal angle of the spin vector."""
        return self._phi

    @property
    def x(self) -> float:
        """Get the x-component of the spin vector."""
        return np.sin(self.theta) * np.cos(self.phi)

    @property
    def y(self) -> float:
        """Get the y-component of the spin vector."""
        return np.sin(self.theta) * np.sin(self.phi)

    @property
    def z(self) -> float:
        """Get the z-component of the spin vector."""
        return np.cos(self.theta)

    @property
    def cartesian(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the Cartesian coordinates of the spin vector."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @staticmethod
    def from_cartesian(
        x: float, y: float, z: float, *, n_stars: int = 1
    ) -> CoherentSpin:
        """Create a Spin from Cartesian coordinates."""
        r = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(r, 1, rtol=1e-3), f"Spin vector must be normalized. r = {r}"
        return CoherentSpin(
            theta=np.arccos(z / r), phi=np.arctan2(y, x), n_stars=n_stars
        )


test = Spin.from_majorana_stars(
    [CoherentSpin(theta=0.1, phi=0.2), CoherentSpin(theta=0.3, phi=0.4)]
)
