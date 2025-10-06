from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from spinecho_sim.field import (
    FieldPlotConfig,
    HeatmapConfig,
    ProportionalPitchWinding,
    make_axis_region_from_winding,
    plot_field_along_axis,
    plot_field_heatmap,
)

if __name__ == "__main__":
    # Axis sampling grid
    z = np.linspace(0.0, 0.75, 2001)

    # Three finite layers, small turn counts (discrete windings)
    w = ProportionalPitchWinding(
        length=0.75,
        radii=[1.65e-2, 2.295e-2],  # 16.5mm inner radius, 2.15mm coil radius
        turns_per_layer=[1370, 995],  # finite, discrete turns
        current_per_layer=[1.0 / 1370, -1.0 / 995],
    )

    axis_region = make_axis_region_from_winding(z, w, include_derivatives=True)
    # -> hand 'axis_region' to your new AxisDataFieldRegion pipeline

    # Plot the field along the axis
    fig1, ax1 = plot_field_along_axis(
        axis_region,
        config=FieldPlotConfig(),
    )
    ax1.set_title("On-Axis Field from Measured Data")

    # Plot a heatmap of the field
    fig2, ax2 = plot_field_heatmap(
        axis_region,
        component="Bz",
        x_max=1.16e-3,  # beam radius
        config=HeatmapConfig(
            cmap="coolwarm", symmetric_scale=True, show_field_lines=True
        ),
    )
    ax2.set_title("Field from Measured Data - Bz Component")
    fig2.savefig("./examples/nested_solenoids_heatmap_bz.png")

    # Plot a heatmap of the field magnitude
    fig3, ax3 = plot_field_heatmap(
        axis_region,
        component="magnitude",
        x_max=1.16e-3,  # beam radius
        config=HeatmapConfig(show_field_lines=True),
    )
    ax3.set_title("Nested Solenoids - Field Magnitude")
    plt.show()
