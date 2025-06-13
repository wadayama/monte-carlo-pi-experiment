"""Monte Carlo method for calculating π using random sampling.

This module implements a Monte Carlo experiment to estimate the value of π
by randomly sampling points in a unit square and determining what fraction
fall within the inscribed unit circle.

The theoretical basis is that the ratio of the area of a unit circle (π/4)
to the area of the unit square (1) equals π/4. Therefore, by counting the
fraction of random points that fall within the circle, we can estimate π.
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import patches


class Point(NamedTuple):
    """Represents a 2D point with x and y coordinates."""

    x: float
    y: float


class ExperimentResult(NamedTuple):
    """Results from a Monte Carlo π estimation experiment."""

    n_total: int
    n_inside: int
    pi_estimate: float
    error: float
    points_inside: list[Point]
    points_outside: list[Point]


def generate_random_point(rng: np.random.Generator) -> Point:
    """Generate a random point in the unit square [0,1] x [0,1].

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducible results.

    Returns
    -------
    Point
        A random point with coordinates in [0,1] x [0,1].
    """
    x = rng.uniform(0.0, 1.0)
    y = rng.uniform(0.0, 1.0)
    return Point(x, y)


def check_inside_circle(point: Point) -> bool:
    """Check if a point lies inside the unit circle centered at origin.

    Parameters
    ----------
    point : Point
        The point to check.

    Returns
    -------
    bool
        True if the point is inside or on the unit circle, False otherwise.
    """
    return point.x**2 + point.y**2 <= 1.0


def run_monte_carlo_experiment(
    n: int,
    seed: int,
    visualization_sample: int | None = None,
) -> ExperimentResult:
    """Run Monte Carlo experiment to estimate π.

    Parameters
    ----------
    n : int
        Number of random points to generate.
    seed : int
        Random seed for reproducible results.
    visualization_sample : int | None, optional
        Maximum number of points to store for visualization.
        If None, stores all points.

    Returns
    -------
    ExperimentResult
        Complete results of the experiment including π estimate and sampled points.
    """
    if n <= 0:
        raise ValueError("Number of points must be positive")

    rng = np.random.default_rng(seed)

    n_inside = 0
    points_inside: list[Point] = []
    points_outside: list[Point] = []

    max_points_to_store = visualization_sample or n

    for _ in range(n):
        point = generate_random_point(rng)

        if check_inside_circle(point):
            n_inside += 1
            if len(points_inside) < max_points_to_store:
                points_inside.append(point)
        elif len(points_outside) < max_points_to_store:
            points_outside.append(point)

    pi_estimate = 4.0 * n_inside / n
    error = abs(pi_estimate - np.pi)

    return ExperimentResult(
        n_total=n,
        n_inside=n_inside,
        pi_estimate=pi_estimate,
        error=error,
        points_inside=points_inside,
        points_outside=points_outside,
    )


def create_scatter_plot(result: ExperimentResult) -> "Figure":
    """Create scatter plot visualization of Monte Carlo sampling.

    Parameters
    ----------
    result : ExperimentResult
        Results from Monte Carlo experiment containing sampled points.

    Returns
    -------
    plt.Figure
        Matplotlib figure showing points inside and outside the unit circle.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot points inside circle (red)
    if result.points_inside:
        x_inside = [p.x for p in result.points_inside]
        y_inside = [p.y for p in result.points_inside]
        ax.scatter(x_inside, y_inside, c="red", s=1, alpha=0.6, label="Inside circle")

    # Plot points outside circle (blue)
    if result.points_outside:
        x_outside = [p.x for p in result.points_outside]
        y_outside = [p.y for p in result.points_outside]
        ax.scatter(
            x_outside,
            y_outside,
            c="blue",
            s=1,
            alpha=0.6,
            label="Outside circle",
        )

    # Draw unit circle
    circle = patches.Circle(
        (0, 0),
        1,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
        label="Unit circle",
    )
    ax.add_patch(circle)

    # Draw unit square
    square = patches.Rectangle(
        (0, 0),
        1,
        1,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
        label="Unit square",
    )
    ax.add_patch(square)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()

    # Add title with results
    n_shown = len(result.points_inside) + len(result.points_outside)
    ax.set_title(
        f"Monte Carlo π Estimation\n"
        f"Total points: {result.n_total:,}, Shown: {n_shown:,}\n"
        f"π estimate: {result.pi_estimate:.6f} (error: {result.error:.6f})",
        fontsize=12,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig


def calculate_pi_statistics(results: list[ExperimentResult]) -> dict[str, float]:
    """Calculate statistical measures for multiple π estimates.

    Parameters
    ----------
    results : list[ExperimentResult]
        List of experiment results to analyze.

    Returns
    -------
    dict[str, float]
        Dictionary containing statistical measures: mean, std, min, max, etc.
    """
    if not results:
        raise ValueError("Results list cannot be empty")

    estimates = [r.pi_estimate for r in results]
    errors = [r.error for r in results]

    return {
        "mean_estimate": float(np.mean(estimates)),
        "std_estimate": float(np.std(estimates)),
        "min_estimate": float(np.min(estimates)),
        "max_estimate": float(np.max(estimates)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "theoretical_pi": float(np.pi),
    }


def save_scatter_plot_pdf(
    result: ExperimentResult,
    output_path: Path,
    commit_hash: str,
) -> None:
    """Save scatter plot as PDF with metadata.

    Parameters
    ----------
    result : ExperimentResult
        Experiment results to visualize.
    output_path : Path
        Path where PDF file should be saved.
    commit_hash : str
        Git commit hash for reproducibility tracking.
    """
    figure = create_scatter_plot(result)

    # Add commit hash to figure
    figure.text(0.02, 0.02, f"Git commit: {commit_hash}", fontsize=8, alpha=0.7)

    try:
        figure.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        logging.info("Scatter plot saved to %s", output_path)
    except Exception:
        logging.exception("Failed to save scatter plot")
        raise
    finally:
        plt.close(figure)


def load_config(config_path: Path) -> dict:
    """Load experiment configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary loaded from YAML.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    yaml.YAMLError
        If YAML file is malformed.
    """
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded from %s", config_path)
        return config
    except FileNotFoundError:
        logging.exception("Configuration file not found")
        raise
    except yaml.YAMLError:
        logging.exception("Error parsing YAML configuration")
        raise


def find_config_files(config_dir: Path) -> list[Path]:
    """Find all YAML configuration files in a directory.

    Parameters
    ----------
    config_dir : Path
        Directory to search for configuration files.

    Returns
    -------
    list[Path]
        List of paths to YAML configuration files, sorted by name.
    """
    if not config_dir.exists():
        logging.warning("Configuration directory does not exist: %s", config_dir)
        return []

    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    yaml_files.sort()

    logging.info("Found %d configuration files in %s", len(yaml_files), config_dir)
    return yaml_files


def get_git_commit_hash() -> str:
    """Get current git commit hash for reproducibility tracking.

    Returns
    -------
    str
        Git commit hash, with '-dirty' suffix if working directory has
        uncommitted changes, or 'not-a-git-repo' if not in a git repository.
    """
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )

        status = (
            subprocess.check_output(["git", "status", "--porcelain"])
            .strip()
            .decode("utf-8")
        )

        if status:
            return f"{commit_hash}-dirty"
        return commit_hash

    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not-a-git-repo"


def run_single_experiment(
    config: dict, output_dir: Path, config_name: str | None = None
) -> ExperimentResult:
    """Run a single Monte Carlo experiment based on configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing experiment parameters.
    output_dir : Path
        Directory where outputs should be saved.
    config_name : str | None, optional
        Name of the configuration file (without extension) for output filename.
        If None, uses generic naming.

    Returns
    -------
    ExperimentResult
        Results from the experiment.
    """
    experiment_config = config["experiment"]
    output_config = config.get("output", {})

    n = experiment_config["n"]
    seed = experiment_config["seed"]
    visualization_sample = experiment_config.get("visualization_sample")

    logging.info("Running Monte Carlo experiment with n=%s, seed=%s", f"{n:,}", seed)

    # Run the experiment
    result = run_monte_carlo_experiment(n, seed, visualization_sample)

    # Log results
    logging.info(
        "Experiment completed: π estimate = %.6f, error = %.6f",
        result.pi_estimate,
        result.error,
    )

    # Save visualization if requested
    if output_config.get("save_plot", True):
        commit_hash = get_git_commit_hash()
        if commit_hash.endswith("-dirty"):
            logging.warning(
                "Working directory has uncommitted changes. "
                "Reproducibility might be compromised."
            )

        # Create filename with config name for better traceability
        if config_name:
            plot_filename = (
                f"monte_carlo_pi_{config_name}_n{n}_seed{seed}_{commit_hash}.pdf"
            )
        else:
            plot_filename = f"monte_carlo_pi_n{n}_seed{seed}_{commit_hash}.pdf"
        plot_path = output_dir / plot_filename

        output_dir.mkdir(parents=True, exist_ok=True)
        save_scatter_plot_pdf(result, plot_path, commit_hash)

    return result


def main() -> None:
    """Main entry point for Monte Carlo π estimation experiment."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo method for estimating π",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment.yaml"),
        help="Path to experiment configuration file",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch experiments using all configs in config/experiments/",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output files",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    commit_hash = get_git_commit_hash()
    logging.info("Experiment running on git commit: %s", commit_hash)
    if commit_hash.endswith("-dirty"):
        logging.warning(
            "Working directory has uncommitted changes. "
            "Reproducibility might be compromised."
        )

    try:
        if args.batch:
            # Run batch experiments
            experiments_dir = Path("config/experiments")
            config_files = find_config_files(experiments_dir)

            if not config_files:
                logging.error("No configuration files found in %s", experiments_dir)
                return

            results = []
            for config_file in config_files:
                logging.info("Running experiment with config: %s", config_file.name)
                config = load_config(config_file)
                # Extract config name (without extension) for filename
                config_name = config_file.stem
                result = run_single_experiment(config, args.output_dir, config_name)
                results.append(result)

            # Calculate and log batch statistics
            if len(results) > 1:
                stats = calculate_pi_statistics(results)
                logging.info("Batch experiment statistics:")
                logging.info(
                    "  Mean π estimate: %.6f ± %.6f",
                    stats["mean_estimate"],
                    stats["std_estimate"],
                )
                logging.info(
                    "  Best estimate: %.6f to %.6f",
                    stats["min_estimate"],
                    stats["max_estimate"],
                )
                logging.info("  Mean error: %.6f", stats["mean_error"])

        else:
            # Run single experiment
            config = load_config(args.config)
            # Extract config name for filename if available
            config_name = (
                args.config.stem if args.config.name != "experiment.yaml" else None
            )
            run_single_experiment(config, args.output_dir, config_name)

    except Exception:
        logging.exception("Experiment failed")
        raise


if __name__ == "__main__":
    main()
