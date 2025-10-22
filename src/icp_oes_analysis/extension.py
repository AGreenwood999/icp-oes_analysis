from icp_oes_analysis.core import Experiment
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def get_coordinates(plane: pl.DataFrame):
    coordinates = plane.select("Sample").to_numpy()

    xs = np.zeros(len(coordinates), dtype=np.float64)
    ys = np.zeros(len(coordinates), dtype=np.float64)
    for i, coord in enumerate(coordinates):
        xs[i], ys[i], _ = [float(i) for i in coord[0].split(",")]

    return xs, ys


def extension(exp: Experiment):
    plane_12 = exp.results.filter(pl.col("Sample").str.ends_with("12"))  # type: ignore
    plane_15 = exp.results.filter(pl.col("Sample").str.ends_with("15"))  # type: ignore

    x_12, y_12 = get_coordinates(plane_12)
    x_15, y_15 = get_coordinates(plane_15)

    mass_12 = plane_12.select("Fe3O4_mass_mg").to_numpy()
    mass_15 = plane_15.select("Fe3O4_mass_mg").to_numpy()

    fig, (ax12, ax15) = plt.subplots(1, 2, sharey=True)

    ax12.scatter(x_12, y_12, c=mass_12, s=150)
    ax15.scatter(x_15, y_15, c=mass_15, s=150)
    plt.show()
