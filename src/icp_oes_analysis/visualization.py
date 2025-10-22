import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from icp_oes_analysis.core import Experiment


def plot_results(exp: Experiment) -> Figure:
    """
    Create single plot with:
    - Calibration curve with standards
    - Sample points overlaid

    That's all you need.
    """
    if exp.results is None or exp.calibration_fit is None:
        raise ValueError("Run analysis before plotting")

    num_of_samples = len(exp.config["groups"])
    cmap = plt.cm.get_cmap(exp.config["plotting"].get("cmap", "tab20c"), num_of_samples)
    colors = [cmap(i) for i in range(num_of_samples)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get calibration standards
    wavelength = exp.wavelength
    cal_data = (
        exp.raw_data.filter(pl.col("Sample").str.starts_with("STD"))
        .filter(~pl.col("Sample").str.ends_with("DNU"))
        .select(
            [
                "Sample",
                pl.col(f"Raw.Average Fe {wavelength}").alias("Signal"),
                pl.col(f"Raw.STD Fe {wavelength}").alias("Signal_STD"),
            ]
        )
        .with_columns(
            pl.col("Sample").str.slice(3, None).cast(pl.Float64).alias("Concentration")
        )
        .collect()
    )

    # Plot calibration curve
    x_cal = cal_data["Concentration"].to_numpy()
    y_cal = cal_data["Signal"].to_numpy()
    yerr_cal = cal_data["Signal_STD"].to_numpy()

    ax.errorbar(
        x_cal,
        y_cal,
        yerr=yerr_cal,
        fmt="o",
        color="red",
        markersize=8,
        label="Calibration Standards",
        capsize=4,
    )

    # Plot fit line
    x_line = np.linspace(0, x_cal.max() * 1.1, 100)
    y_line = exp.calibration_fit.slope * x_line + exp.calibration_fit.intercept
    ax.plot(
        x_line,
        y_line,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Calibration Fit",
    )

    # Plot samples
    for i, (group, samples) in enumerate(exp.config["groups"].items()):
        x_samples = exp.results.filter(pl.col("Sample").is_in(samples))[
            "Fe_concentration_mgL"
        ].to_numpy()
        y_samples = exp.results.filter(pl.col("Sample").is_in(samples))[
            "Signal"
        ].to_numpy()

        pc = ax.scatter(x_samples, y_samples, s=100, label=group, color=colors[i])

        ax.axhline(y_samples.mean(), color=colors[i])

    # Add fit equation
    fit_text = (
        f"y = {exp.calibration_fit.slope:.2f}x + {exp.calibration_fit.intercept:.2f}\n"
        f"R² = {exp.calibration_fit.r_squared:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        fit_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        verticalalignment="top",
        fontsize=10,
    )

    ax.set_xlabel("Fe Concentration (mg/L)", fontsize=12)
    ax.set_ylabel("Signal Intensity", fontsize=12)
    ax.set_title(f"Calibration Curve - {wavelength}nm", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
