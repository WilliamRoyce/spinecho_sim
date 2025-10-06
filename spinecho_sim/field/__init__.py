"""Module for flexible and extensible field representation framework."""

from __future__ import annotations

from spinecho_sim.field._cylindrical_winding import (
    ProportionalPitchWinding,
    make_axis_region_from_winding,
)
from spinecho_sim.field._field import (
    AnalyticFieldRegion,
    AxisDataFieldRegion,
    DataFieldRegion,
    FieldRegion,
    FieldSequence,
    FieldSuperposition,
    RotatedFieldRegion,
    ScaledFieldRegion,
    SolenoidRegion,
    TranslatedFieldRegion,
    UniformFieldRegion,
    ZeroField,
)
from spinecho_sim.field._plotting import (
    FieldPlotConfig,
    HeatmapConfig,
    plot_field_along_axis,
    plot_field_heatmap,
)

__all__ = [
    "AnalyticFieldRegion",
    "AxisDataFieldRegion",
    "DataFieldRegion",
    "FieldPlotConfig",
    "FieldRegion",
    "FieldSequence",
    "FieldSuperposition",
    "HeatmapConfig",
    "ProportionalPitchWinding",
    "RotatedFieldRegion",
    "ScaledFieldRegion",
    "SolenoidRegion",
    "TranslatedFieldRegion",
    "UniformFieldRegion",
    "ZeroField",
    "make_axis_region_from_winding",
    "plot_field_along_axis",
    "plot_field_heatmap",
]
