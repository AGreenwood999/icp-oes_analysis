from icp_oes_analysis.core import Experiment, CalibrationFit
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import polars as pl


def get_averages(
    trials: pl.DataFrame, all_data: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    avg = np.zeros((5,), dtype=np.float64)
    std = np.zeros((5,), dtype=np.float64)
    if all_data:
        data_all = np.zeros((15,), dtype=np.float64)
    else:
        data_all = None

    for i, conc in enumerate([1, 2, 4, 8, 16]):
        data = (
            trials.filter(
                pl.col("Sample")
                .str.starts_with(f"{conc}A")
                .or_(
                    pl.col("Sample").str.starts_with(f"{conc}B"),
                    pl.col("Sample").str.starts_with(f"{conc}C"),
                )
            )
            .select("Fe3O4_mass_mg")
            .to_numpy()
            .flatten()
        )
        if data_all is not None:
            data_all[3 * i : 3 * i + 3] = data

        avg[i] = data.mean()
        std[i] = data.std()

    if data_all is not None:
        return avg, std, data_all
    else:
        return avg, std


def bar_plot(exp: Experiment):
    phantom = exp.results.filter(  # type: ignore
        pl.col("Sample")
        .str.ends_with("CC")
        .not_()
        .and_(
            pl.col("Sample")
            .str.ends_with("A")
            .or_(
                pl.col("Sample").str.ends_with("B"),
                pl.col("Sample").str.ends_with("C"),
            )
        )
    ).select(["Sample", "Fe3O4_mass_mg"])
    conc_cntrl = exp.results.filter(  # type: ignore
        pl.col("Sample")
        .str.ends_with("A CC")
        .or_(
            pl.col("Sample").str.ends_with("B CC"),
            pl.col("Sample").str.ends_with("C CC"),
        )
    ).select(["Sample", "Fe3O4_mass_mg"])

    categories = ["1mg/kg", "2mg/kg", "4mg/kg", "8mg/kg", "16mg/kg"]

    avg_phantom, std_phantom = get_averages(phantom)
    avg_conc_cntrl, std_conc_cntrl = get_averages(conc_cntrl)

    fig, ax_phantom = plt.subplots(1, 1, sharey=True)

    ax_phantom.bar(
        categories,
        avg_phantom,
        yerr=std_phantom,
    )
    ax_phantom.set_ylabel("$Fe_3O_4$ mass (ug)")
    ax_phantom.set_xlabel("Dosing")

    plt.show()


def actual_vs_dosage(exp: Experiment):
    phantom = exp.results.filter(  # type: ignore
        pl.col("Sample")
        .str.ends_with("CC")
        .not_()
        .and_(
            pl.col("Sample")
            .str.ends_with("A")
            .or_(
                pl.col("Sample").str.ends_with("B"),
                pl.col("Sample").str.ends_with("C"),
            )
        )
    ).select(["Sample", "Fe3O4_mass_mg"])
    conc_cntrl = exp.results.filter(  # type: ignore
        pl.col("Sample")
        .str.ends_with("A CC")
        .or_(
            pl.col("Sample").str.ends_with("B CC"),
            pl.col("Sample").str.ends_with("C CC"),
        )
    ).select(["Sample", "Fe3O4_mass_mg"])

    categories = ["1mg/kg", "2mg/kg", "4mg/kg", "8mg/kg", "16mg/kg"]

    avg_phantom, std_phantom, phantom_all = get_averages(phantom, True)  # type: ignore
    avg_conc_cntrl, std_conc_cntrl, conc_all = get_averages(conc_cntrl, True)  # type: ignore

    fig, ax = plt.subplots(1, 1)

    slope, intercept = np.polyfit(conc_all, phantom_all, deg=1)
    y_pred = slope * conc_all + intercept
    ss_res = np.sum((phantom_all - y_pred) ** 2)
    ss_tot = np.sum((phantom_all - np.mean(phantom_all)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    for i, conc in enumerate(categories):
        line = ax.scatter(avg_conc_cntrl[i], avg_phantom[i], s=50, label=conc)
        ax.errorbar(
            avg_conc_cntrl[i],
            avg_phantom[i],
            xerr=std_conc_cntrl[i],
            yerr=std_phantom[i],
            ecolor=line.get_facecolor(),
            linestyle=None,
        )

    ax.plot(conc_all, slope * conc_all + intercept)

    fit_text = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.4f}"
    ax.text(
        0.5,
        0.95,
        fit_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        verticalalignment="top",
        fontsize=10,
    )
    ax.set_xlabel("Input mass ($ug_{Fe_3O_4}$)")
    ax.set_ylabel("Output mass ($ug_{Fe_3O_4}$)")
    plt.legend(loc="upper left")
    plt.show()


def for_next_protocol(exp: Experiment):
    conc_cntrl = exp.results.filter(  # type: ignore
        pl.col("Sample")
        .str.ends_with("A CC")
        .or_(
            pl.col("Sample").str.ends_with("B CC"),
            pl.col("Sample").str.ends_with("C CC"),
        )
    ).select(["Sample", "Fe3O4_mass_mg"])

    riffle_sample_low = (
        exp.results.filter(pl.col("Sample").str.ends_with("2712"))
        .select("Fe3O4_mass_mg")
        .item()
    )
    riffle_sample_high = (
        exp.results.filter(pl.col("Sample").str.ends_with("2702"))
        .select("Fe3O4_mass_mg")
        .item()
    )
    masses = np.array(
        [
            1 * riffle_sample_low,
            1 * riffle_sample_low,
            1 * riffle_sample_low,
            2 * riffle_sample_low,
            2 * riffle_sample_low,
            2 * riffle_sample_low,
            1 * riffle_sample_high,
            1 * riffle_sample_high,
            1 * riffle_sample_high,
            2 * riffle_sample_high,
            2 * riffle_sample_high,
            2 * riffle_sample_high,
            4 * riffle_sample_high,
            4 * riffle_sample_high,
            4 * riffle_sample_high,
        ]
    )
    volumes = 19 + 24 + np.array([2, 2, 2, 4, 4, 4, 2, 2, 2, 4, 4, 4, 8, 8, 8])

    avg_conc_cntrl, std_conc_cntrl, conc_all = get_averages(conc_cntrl, True)  # type: ignore
    std_conc_cntrl = std_conc_cntrl / 0.261
    avg_conc_cntrl = avg_conc_cntrl / 0.261
    conc_all = conc_all / 0.261

    conc_all = np.array([a * b for (a, b) in zip(conc_all, volumes)])
    avg_conc_cntrl = np.array([a * b for (a, b) in zip(avg_conc_cntrl, volumes[::3])])
    std_conc_cntrl = np.array([a * b for (a, b) in zip(std_conc_cntrl, volumes[::3])])

    fig, ax = plt.subplots(1, 1)

    slope, intercept = np.polyfit(masses, conc_all, deg=1)
    y_pred = slope * masses + intercept
    ss_res = np.sum((conc_all - y_pred) ** 2)
    ss_tot = np.sum((conc_all - np.mean(conc_all)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    line = ax.scatter(masses[::3], avg_conc_cntrl, s=50, color="red")
    ax.errorbar(
        masses[::3],
        avg_conc_cntrl,
        yerr=std_conc_cntrl,
        ecolor=line.get_facecolor(),  # type: ignore
        linestyle="",
    )

    ax.plot(masses[::3], slope * masses[::3] + intercept, color="blue")

    fit_text = f"y = {slope:.4f}x + {intercept:.4f}\nR² = {r_squared:.4f}"
    ax.text(
        0.5,
        0.95,
        fit_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        verticalalignment="top",
        fontsize=10,
    )
    ax.set_xlabel("Input mass ($mg_{Fe_3O_4}$)")
    ax.set_ylabel("Output mass ($mg_{Fe_3O_4}$)")

    # ax.legend(loc="upper left")
    plt.show()


def extension(exp: Experiment):
    bar_plot(exp)
    actual_vs_dosage(exp)
    for_next_protocol(exp)
