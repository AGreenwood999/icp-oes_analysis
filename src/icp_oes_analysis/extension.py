from icp_oes_analysis.core import Experiment
import matplotlib.pyplot as plt
import polars as pl


def bar_plot(exp: Experiment):
    groups: dict[str, list[str]] = exp.config["groups"]
    groups.pop("Riffled")
    groups.pop("Glycerin control")
    names = groups.keys()

    data = {}
    for name in names:
        data[name] = (
            exp.results.filter(pl.col("Sample").is_in(groups[name]))
            .select("Fe3O4_mass_ug")
            .to_numpy()
            .flatten()
        )

    plt.bar(
        list(names),
        [d.mean() for d in data.values()],
        yerr=[d.std() for d in data.values()],
    )

    plt.show()


def extension(exp: Experiment):
    bar_plot(exp)
