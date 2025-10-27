import logging

import numpy as np
import polars as pl

from icp_oes_analysis.core import CalibrationFit, Experiment

logger = logging.getLogger(__name__)

# Constants
FE_TO_FE3O4_RATIO = 231.533 / (55.845 * 3)  # Molecular weight ratio


def analyze_experiment(exp: Experiment) -> Experiment:
    """
    Complete analysis pipeline:
    1. Try each wavelength
    2. Pick best R²
    3. Calculate concentrations
    4. Convert to Fe3O4 mass
    5. Return results
    """
    logger.info("Starting analysis...")

    # Step 1 & 2: Fit all wavelengths and pick best
    best_wavelength = 239  # default to 239
    best_fit = None
    best_r2 = 0

    for wavelength in [238, 239, 240, 259]:
        try:
            fit = _fit_wavelength(exp, wavelength)
            logger.info(f"Wavelength {wavelength}nm: {fit}")

            if fit.r_squared > best_r2:
                best_r2 = fit.r_squared
                best_fit = fit
                best_wavelength = wavelength
        except Exception as e:
            logger.warning(f"Could not fit {wavelength}nm: {e}")

    if best_fit is None:
        raise ValueError("No valid calibration curves found")

    logger.info(f"Selected wavelength: {best_wavelength}nm (R² = {best_r2:.4f})")

    # Step 3 & 4: Calculate sample results
    results = _calculate_sample_masses(exp, best_wavelength, best_fit)

    # Store results
    exp.calibration_fit = best_fit
    exp.wavelength = best_wavelength
    exp.results = results

    logger.info(f"Analysis complete. Processed {len(results)} samples.")
    return exp


def _fit_wavelength(exp: Experiment, wavelength: int) -> CalibrationFit:
    """Fit calibration curve for a wavelength."""

    # Extract calibration standards (samples starting with "STD")
    cal_data = (
        exp.raw_data.filter(pl.col("Sample").str.starts_with("STD"))
        .select(["Sample", pl.col(f"Raw.Average Fe {wavelength}").alias("Signal")])
        .with_columns(
            # Extract concentration from sample name (e.g., "STD5" → 5.0)
            pl.col("Sample")
            .str.strip_chars()
            .str.slice(3, None)
            .cast(pl.Float64)
            .alias("Concentration")
        )
        .collect()
    )

    if len(cal_data) < 2:
        raise ValueError(f"Not enough calibration standards for {wavelength}nm")

    # Simple linear fit
    x = cal_data["Concentration"].to_numpy()
    y = cal_data["Signal"].to_numpy()

    # Weighted fit (weight by 1/concentration to emphasize low end)
    weights = 1.0 / np.maximum(x, 1e-10)
    coeffs = np.polyfit(x, y, deg=1, w=weights)

    slope, intercept = coeffs

    # Calculate R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return CalibrationFit(slope=slope, intercept=intercept, r_squared=r_squared)


# Apply dilution factors and calculate total Fe mass
def get_dilution_factor(exp: Experiment, sample_name: str) -> float:
    dilution = exp.config["dilutions"].get(sample_name)
    V_initial = dilution["V_initial"]
    V_digestion = dilution["V_digestion"]
    V_aliquot = dilution["V_aliquot"]
    V_final = dilution["V_final"]
    return (V_initial * V_aliquot) / (V_digestion * V_final)


def get_initial_volume(exp: Experiment, sample_name: str) -> float:
    dilution = exp.config["dilutions"].get(sample_name)
    return dilution.get("V_initial", 0.0)


def _calculate_sample_masses(
    exp: Experiment, wavelength: int, fit: CalibrationFit
) -> pl.DataFrame:
    """Calculate Fe3O4 mass for all samples."""

    # Extract sample data (non-standards)
    samples = (
        exp.raw_data.filter(~pl.col("Sample").str.starts_with("STD"))
        .filter(~pl.col("Sample").str.contains("blank"))
        .select(
            "Sample",
            Signal=pl.col(f"Raw.Average Fe {wavelength}"),
            Signal_STD=pl.col(f"Raw.STD Fe {wavelength}"),
        )
        # Get iron concentration from calibration curve
        .with_columns(
            Fe_concentration_mgL=((pl.col("Signal") - fit.intercept) / fit.slope)
        )
        # Put dilution factor and intial volume in dataframe to be used after this
        .with_columns(
            dilution_factor=pl.col("Sample").map_elements(
                lambda x: get_dilution_factor(exp, x), return_dtype=pl.Float64
            ),
            initial_volume_mL=pl.col("Sample")
            .map_elements(
                lambda x: get_dilution_factor(exp, x), return_dtype=pl.Float64
            )
            .alias("initial_volume_mL"),
        )
        # Use the dilution factors and initial volume from before to get
        # Concentration and initial mass in mg
        # Fe mass (mg) = concentration (mg/L) / dilution_factor * initial_volume (L)
        .with_columns(
            Fe_mass_mg=pl.col("Fe_concentration_mgL")
            / pl.col("dilution_factor")
            * pl.col("initial_volume_mL")
            / 1000
        )
        # Convert from iron mass to iron oxide mass
        .with_columns(Fe3O4_mass_mg=pl.col("Fe_mass_mg") * FE_TO_FE3O4_RATIO)
        .collect()
    )

    # Select final columns for display
    return samples.select(
        [
            "Sample",
            "Signal",
            "Fe_concentration_mgL",
            "dilution_factor",
            "Fe_mass_mg",
            "Fe3O4_mass_mg",
        ]
    )
