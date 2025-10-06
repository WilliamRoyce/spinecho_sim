from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from spinecho_sim.field._field import (
    AxisDataFieldRegion,
    FieldRegion,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

MU0 = 4e-7 * np.pi  # [H/m]


@dataclass(frozen=True, kw_only=True)
class ProportionalPitchWinding:
    """Finite cylindrical winding with constant pitch across layers."""

    length: float
    radii: Sequence[float]  # layer radii [m]
    turns_per_layer: Sequence[int]  # turns per layer (finite)
    current_per_layer: Sequence[float]

    @property
    def _pitch(self) -> float:
        n_max = max(self.turns_per_layer) if self.turns_per_layer else 1
        return self.length / max(n_max, 1)

    @property
    def loop_lengths(self) -> np.ndarray:
        return self._pitch * np.asarray(self.turns_per_layer)


def _bz_axis_from_loops(
    z_vals: np.ndarray,
    winding: ProportionalPitchWinding,
) -> np.ndarray:
    """Sum the on-axis field Bz(z) from a finite set of circular loops."""
    currents = np.asarray(winding.current_per_layer)
    turns = np.asarray(winding.turns_per_layer)
    radii = np.asarray(winding.radii)
    dz = z_vals[None, :] - winding.length / 2  # (L, M), centered about middle
    lengths = winding.loop_lengths

    denom_plus = np.sqrt(radii[:, None] ** 2 + (dz + 0.5 * lengths[:, None]) ** 2)
    denom_minus = np.sqrt(radii[:, None] ** 2 + (dz - 0.5 * lengths[:, None]) ** 2)
    contrib = (
        0.5
        * MU0
        * currents[:, None]
        * turns[:, None]
        * (
            (dz + 0.5 * lengths[:, None]) / denom_plus
            - (dz - 0.5 * lengths[:, None]) / denom_minus
        )
    )
    return contrib.sum(axis=0)  # (M,)


def _bz_p_axis_from_loops(
    z_vals: np.ndarray,
    winding: ProportionalPitchWinding,
) -> np.ndarray:
    """Sum the on-axis field Bz'(z) from a finite set of circular loops."""
    currents = np.asarray(winding.current_per_layer)
    turns = np.asarray(winding.turns_per_layer)
    radii = np.asarray(winding.radii)
    dz = z_vals[None, :] - winding.length / 2
    lengths = winding.loop_lengths

    denom_plus = (radii[:, None] ** 2 + (dz + 0.5 * lengths[:, None]) ** 2) ** 1.5
    denom_minus = (radii[:, None] ** 2 + (dz - 0.5 * lengths[:, None]) ** 2) ** 1.5

    contrib = (
        0.5
        * MU0
        * currents[:, None]
        * turns[:, None]
        * radii[:, None] ** 2
        * (1 / denom_plus - 1 / denom_minus)
    )
    return contrib.sum(axis=0)  # (M,)


def _bz_pp_axis_from_loops(
    z_vals: np.ndarray,
    winding: ProportionalPitchWinding,
) -> np.ndarray:
    """Sum the on-axis field Bz''(z) from a finite set of circular loops."""
    currents = np.asarray(winding.current_per_layer)
    turns = np.asarray(winding.turns_per_layer)
    radii = np.asarray(winding.radii)
    dz = z_vals[None, :] - winding.length / 2
    lengths = winding.loop_lengths

    denom_plus = (radii[:, None] ** 2 + (dz + 0.5 * lengths[:, None]) ** 2) ** 2.5
    denom_minus = (radii[:, None] ** 2 + (dz - 0.5 * lengths[:, None]) ** 2) ** 2.5

    contrib = (
        1.5
        * MU0
        * currents[:, None]
        * turns[:, None]
        * radii[:, None] ** 2
        * (
            -(dz + 0.5 * lengths[:, None]) / denom_plus
            + (dz - 0.5 * lengths[:, None]) / denom_minus
        )
    )
    return contrib.sum(axis=0)  # (M,)


def make_axis_region_from_winding(
    z_vals: np.ndarray,
    winding: ProportionalPitchWinding,
    *,
    include_derivatives: bool = True,
) -> FieldRegion:
    """Compute on-axis data (and optional derivatives) for a finite set of turns laid out by 'winding', returns AxisDataFieldRegion."""
    bz = _bz_axis_from_loops(
        z_vals,
        winding,
    )

    if include_derivatives:
        dbz = _bz_p_axis_from_loops(z_vals, winding)
        d2bz = _bz_pp_axis_from_loops(z_vals, winding)
        return AxisDataFieldRegion(
            z_vals=z_vals, bz_vals=bz, bz_deriv_vals=dbz, bz_second_deriv_vals=d2bz
        )

    return AxisDataFieldRegion(z_vals=z_vals, bz_vals=bz)
