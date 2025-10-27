from icp_oes_analysis.core import Experiment
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import polars as pl


def get_averages(trials: pl.DataFrame):
    avg = []
    std = []
    for i in [1, 2, 4, 8, 16]:
        data = (
            trials.filter(
                pl.col("Sample")
                .str.starts_with(f"{i}A")
                .or_(
                    pl.col("Sample").str.starts_with(f"{i}B"),
                    pl.col("Sample").str.starts_with(f"{i}C"),
                )
            )
            .select("Fe3O4_mass_mg")
            .to_numpy()
            .flatten()
        )

        avg.append(data.mean())
        std.append(data.std())

    return avg, std


def extension(exp: Experiment):
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

    fig, (ax_phantom, ax_conc_cntrl) = plt.subplots(1, 2, sharey=True)

    ax_phantom.bar(
        categories,
        [i * 1000 for i in avg_phantom],
        yerr=[1000 * i for i in std_phantom],
    )
    ax_phantom.set_title("Phantoms")
    ax_phantom.set_ylabel("$Fe_3O_4$ mass (ug)")
    ax_conc_cntrl.bar(
        categories,
        [i * 1000 for i in avg_conc_cntrl],
        yerr=[i * 1000 for i in std_conc_cntrl],
    )
    ax_conc_cntrl.set_title("Control")

    plt.show()
