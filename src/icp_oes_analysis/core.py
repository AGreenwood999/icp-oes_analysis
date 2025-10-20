from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class CalibrationFit:
    """Linear calibration curve: y = mx + b"""

    slope: float  # m
    intercept: float  # b
    r_squared: float

    def predict_concentration(self, signal: float) -> float:
        """Convert signal → concentration using calibration curve."""
        return (signal - self.intercept) / self.slope

    def __str__(self) -> str:
        return (
            f"y = {self.slope:.2f}x + {self.intercept:.2f}, R² = {self.r_squared:.4f}"
        )


@dataclass
class Experiment:
    """Container for experiment data."""

    data_file: Path
    raw_data: pl.LazyFrame
    config: dict

    # Results (populated during analysis)
    calibration_fit: CalibrationFit | None = None
    wavelength: int | None = None
    results: pl.DataFrame | None = None
