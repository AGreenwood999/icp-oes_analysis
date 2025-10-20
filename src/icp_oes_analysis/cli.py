import logging
from pathlib import Path

import matplotlib.pyplot as plt
import typer

from icp_oes_analysis.analysis import analyze_experiment
from icp_oes_analysis.io import load_experiment
from icp_oes_analysis.visualization import plot_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="ICP-OES Iron Quantification Tool")


@app.command()
def analyze(
    data_file: Path,
    config_file: Path,
    output_dir: Path | None = typer.Option(None, "--output", "-o"),
    show_plot: bool = typer.Option(False, "--show", "-s"),
):
    """
    Analyze ICP-OES data and calculate Fe3O4 mass.

    Simple workflow:
    1. Load data
    2. Fit calibration curves
    3. Calculate Fe3O4 masses
    4. Display results table
    5. Show/save plot
    """

    # Load
    print(f"Loading data from {data_file}...")
    exp = load_experiment(data_file, config_file)

    # Analyze
    print("Running analysis...")
    exp = analyze_experiment(exp)

    # Display results table
    print("\n" + "=" * 80)
    print(f"RESULTS - Wavelength: {exp.wavelength}nm")
    print(f"Calibration: {exp.calibration_fit}")
    print("=" * 80)
    print("\nSample Results:")
    print(exp.results)
    print("=" * 80 + "\n")

    # Create plot
    fig = plot_results(exp, exp.config)

    # Save outputs
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save table
        csv_file = output_dir / "results.csv"
        #  TODO: Figure out how to fix this type hint error
        exp.results.write_csv(csv_file)  # type: ignore
        print(f"✓ Results saved to {csv_file}")

        # Save plot
        plot_file = output_dir / f"calibration_{exp.wavelength}nm.png"
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {plot_file}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print("✓ Analysis complete!")


if __name__ == "__main__":
    app()
