import tomllib
from pathlib import Path

import polars as pl

from icp_oes_analysis.core import Experiment


def _get_column_names(data_file: Path, separator: str = "\t"):
    with open(data_file, "r") as fid:
        data_type = fid.readline().strip().split(separator)
        data_type[0] = "Index"
        data_type[1] = "Sample"

        fid.readline()

        raw_ele_wave_data = fid.readline().strip().split(separator)

    element_wavelength = [ele_wave[:6] for ele_wave in raw_ele_wave_data]
    element_wavelength.insert(0, "")
    element_wavelength.insert(0, "")

    return [
        f"{dtype} {ele_wave}".strip()
        for (dtype, ele_wave) in zip(data_type, element_wavelength)
    ]


def load_experiment(data_file: Path, config_file: Path) -> Experiment:
    """
    Load data and config. That's it.

    Assumes:
    - Tab-separated file
    - First 4 rows are header
    - Columns follow pattern: Fe{wavelength} - Raw.Average, Fe{wavelength} - Raw.STD
    """
    # Load config
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    # Load data - simplified column naming
    # Just keep it simple: read with proper headers from row 4
    separator = config["io"].get("separator", "\t")
    raw_data = (
        pl.scan_csv(
            data_file,
            separator=separator,
            skip_rows=4,  # Skip header rows
            has_header=False,
            new_columns=_get_column_names(data_file, separator),
        )
        .drop(config["io"].get("drop_cols", ["^column_.*$"]))
        .filter(
            pl.col("Sample")
            .str.contains(config["io"].get("drop_rows", ""))
            .not_()
            .and_(pl.col("Sample").str.ends_with("DNU").not_())
        )
    )

    return Experiment(data_file=data_file, raw_data=raw_data, config=config)
