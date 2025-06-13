"""Comprehensive tests for Monte Carlo π estimation experiment."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from monte_carlo_pi_experiment import (
    ExperimentResult,
    Point,
    calculate_pi_statistics,
    check_inside_circle,
    create_scatter_plot,
    find_config_files,
    generate_random_point,
    get_git_commit_hash,
    load_config,
    run_monte_carlo_experiment,
    run_single_experiment,
    save_scatter_plot_pdf,
)


class TestPoint:
    """Test the Point named tuple."""

    def test_point_creation(self):
        """Test Point creation and attribute access."""
        point = Point(0.5, 0.7)
        assert point.x == 0.5
        assert point.y == 0.7

    def test_point_equality(self):
        """Test Point equality comparison."""
        point1 = Point(0.5, 0.7)
        point2 = Point(0.5, 0.7)
        point3 = Point(0.3, 0.7)

        assert point1 == point2
        assert point1 != point3


class TestGenerateRandomPoint:
    """Test random point generation."""

    def test_generate_random_point_in_unit_square(self):
        """Test that generated points are within unit square."""
        rng = np.random.default_rng(42)

        for _ in range(100):
            point = generate_random_point(rng)
            assert 0.0 <= point.x <= 1.0
            assert 0.0 <= point.y <= 1.0

    def test_generate_random_point_reproducibility(self):
        """Test that same seed produces same sequence of points."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        point1 = generate_random_point(rng1)
        point2 = generate_random_point(rng2)

        assert point1 == point2

    def test_generate_random_point_different_seeds(self):
        """Test that different seeds produce different points."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        point1 = generate_random_point(rng1)
        point2 = generate_random_point(rng2)

        assert point1 != point2


class TestCheckInsideCircle:
    """Test circle membership checking."""

    def test_point_at_origin(self):
        """Test that origin is inside circle."""
        point = Point(0.0, 0.0)
        assert check_inside_circle(point) is True

    def test_point_on_circle_boundary(self):
        """Test that points on circle boundary are considered inside."""
        point = Point(1.0, 0.0)  # Distance = 1.0
        assert check_inside_circle(point) is True

        point = Point(0.0, 1.0)  # Distance = 1.0
        assert check_inside_circle(point) is True

        point = Point(1.0 / np.sqrt(2), 1.0 / np.sqrt(2))  # Distance = 1.0
        assert check_inside_circle(point)

    def test_point_inside_circle(self):
        """Test that points inside circle are correctly identified."""
        point = Point(0.5, 0.5)  # Distance < 1.0
        assert check_inside_circle(point) is True

        point = Point(0.3, 0.4)  # Distance = 0.5
        assert check_inside_circle(point) is True

    def test_point_outside_circle(self):
        """Test that points outside circle are correctly identified."""
        point = Point(1.0, 1.0)  # Distance = sqrt(2) > 1.0
        assert check_inside_circle(point) is False

        point = Point(0.9, 0.9)  # Distance > 1.0
        assert check_inside_circle(point) is False


class TestRunMonteCarloExperiment:
    """Test Monte Carlo experiment execution."""

    def test_experiment_with_small_n(self):
        """Test experiment with small number of points."""
        result = run_monte_carlo_experiment(n=100, seed=42)

        assert result.n_total == 100
        assert 0 <= result.n_inside <= 100
        assert 0.0 <= result.pi_estimate <= 8.0  # Generous bounds
        assert result.error >= 0.0
        assert len(result.points_inside) + len(result.points_outside) == 100

    def test_experiment_reproducibility(self):
        """Test that same parameters produce same results."""
        result1 = run_monte_carlo_experiment(n=1000, seed=42)
        result2 = run_monte_carlo_experiment(n=1000, seed=42)

        assert result1.n_total == result2.n_total
        assert result1.n_inside == result2.n_inside
        assert result1.pi_estimate == result2.pi_estimate
        assert result1.error == result2.error

    def test_experiment_different_seeds(self):
        """Test that different seeds produce different results."""
        result1 = run_monte_carlo_experiment(n=1000, seed=42)
        result2 = run_monte_carlo_experiment(n=1000, seed=123)

        # Results should be different (with high probability)
        assert result1.n_inside != result2.n_inside
        assert result1.pi_estimate != result2.pi_estimate

    def test_experiment_with_visualization_sample(self):
        """Test experiment with limited visualization sample."""
        result = run_monte_carlo_experiment(n=1000, seed=42, visualization_sample=100)

        assert result.n_total == 1000
        assert len(result.points_inside) <= 100
        assert len(result.points_outside) <= 100
        # Note: visualization_sample limits EACH category separately
        assert len(result.points_inside) <= 100
        assert len(result.points_outside) <= 100

    def test_experiment_zero_points_raises_error(self):
        """Test that zero points raises ValueError."""
        with pytest.raises(ValueError, match="Number of points must be positive"):
            run_monte_carlo_experiment(n=0, seed=42)

    def test_experiment_negative_points_raises_error(self):
        """Test that negative points raises ValueError."""
        with pytest.raises(ValueError, match="Number of points must be positive"):
            run_monte_carlo_experiment(n=-10, seed=42)

    def test_large_experiment_convergence(self):
        """Test that large experiments converge closer to π."""
        result = run_monte_carlo_experiment(n=100000, seed=42)

        # With 100,000 points, error should typically be < 0.1
        assert result.error < 0.1
        assert 2.5 < result.pi_estimate < 3.5


class TestCreateScatterPlot:
    """Test scatter plot creation."""

    def test_create_scatter_plot_basic(self):
        """Test basic scatter plot creation."""
        result = run_monte_carlo_experiment(n=100, seed=42)
        figure = create_scatter_plot(result)

        assert figure is not None
        assert len(figure.axes) == 1

        ax = figure.axes[0]
        assert ax.get_xlim() == (-0.1, 1.1)
        assert ax.get_ylim() == (-0.1, 1.1)
        assert ax.get_aspect() == 1.0  # 'equal' aspect ratio returns 1.0

        # Clean up
        figure.clear()

    def test_create_scatter_plot_with_empty_result(self):
        """Test scatter plot creation with minimal result."""
        result = ExperimentResult(
            n_total=0,
            n_inside=0,
            pi_estimate=0.0,
            error=np.pi,
            points_inside=[],
            points_outside=[],
        )

        figure = create_scatter_plot(result)
        assert figure is not None

        # Clean up
        figure.clear()


class TestCalculatePiStatistics:
    """Test π statistics calculation."""

    def test_calculate_statistics_single_result(self):
        """Test statistics calculation with single result."""
        result = run_monte_carlo_experiment(n=1000, seed=42)
        stats = calculate_pi_statistics([result])

        assert stats["mean_estimate"] == result.pi_estimate
        assert stats["std_estimate"] == 0.0
        assert stats["min_estimate"] == result.pi_estimate
        assert stats["max_estimate"] == result.pi_estimate
        assert stats["mean_error"] == result.error
        assert stats["theoretical_pi"] == pytest.approx(np.pi)

    def test_calculate_statistics_multiple_results(self):
        """Test statistics calculation with multiple results."""
        results = [
            run_monte_carlo_experiment(n=1000, seed=42),
            run_monte_carlo_experiment(n=1000, seed=123),
            run_monte_carlo_experiment(n=1000, seed=456),
        ]

        stats = calculate_pi_statistics(results)

        estimates = [r.pi_estimate for r in results]
        assert stats["mean_estimate"] == pytest.approx(np.mean(estimates))
        assert stats["std_estimate"] == pytest.approx(np.std(estimates))
        assert stats["min_estimate"] == min(estimates)
        assert stats["max_estimate"] == max(estimates)
        assert stats["theoretical_pi"] == pytest.approx(np.pi)

    def test_calculate_statistics_empty_list_raises_error(self):
        """Test that empty results list raises ValueError."""
        with pytest.raises(ValueError, match="Results list cannot be empty"):
            calculate_pi_statistics([])


class TestSaveScatterPlotPdf:
    """Test PDF saving functionality."""

    def test_save_scatter_plot_pdf(self):
        """Test saving scatter plot as PDF."""
        result = run_monte_carlo_experiment(n=100, seed=42)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_plot.pdf"
            save_scatter_plot_pdf(result, output_path, "test-commit-hash")

            assert output_path.exists()
            assert output_path.stat().st_size > 0


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_config_valid_yaml(self):
        """Test loading valid YAML configuration."""
        config_data = {
            "experiment": {"n": 1000, "seed": 42},
            "output": {"directory": "outputs"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)
            assert config == config_data
        finally:
            config_path.unlink()

    def test_load_config_nonexistent_file(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent.yaml"))

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            config_path.unlink()


class TestFindConfigFiles:
    """Test configuration file discovery."""

    def test_find_config_files_existing_directory(self):
        """Test finding config files in existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create test files
            (config_dir / "config1.yaml").touch()
            (config_dir / "config2.yml").touch()
            (config_dir / "not_config.txt").touch()

            config_files = find_config_files(config_dir)

            assert len(config_files) == 2
            assert all(f.suffix in [".yaml", ".yml"] for f in config_files)
            assert config_files == sorted(config_files)  # Should be sorted

    def test_find_config_files_nonexistent_directory(self):
        """Test finding config files in non-existent directory."""
        config_files = find_config_files(Path("nonexistent_directory"))
        assert config_files == []

    def test_find_config_files_empty_directory(self):
        """Test finding config files in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_files = find_config_files(config_dir)
            assert config_files == []


class TestGetGitCommitHash:
    """Test git commit hash retrieval."""

    def test_get_git_commit_hash_mocked(self):
        """Test git commit hash retrieval with mocked subprocess."""
        with patch("subprocess.check_output") as mock_check_output:
            mock_check_output.side_effect = [
                b"abc123def456\n",  # git rev-parse HEAD
                b"",  # git status --porcelain (clean)
            ]

            commit_hash = get_git_commit_hash()
            assert commit_hash == "abc123def456"

    def test_get_git_commit_hash_dirty_working_dir(self):
        """Test git commit hash with dirty working directory."""
        with patch("subprocess.check_output") as mock_check_output:
            mock_check_output.side_effect = [
                b"abc123def456\n",  # git rev-parse HEAD
                b" M some_file.py\n",  # git status --porcelain (dirty)
            ]

            commit_hash = get_git_commit_hash()
            assert commit_hash == "abc123def456-dirty"

    def test_get_git_commit_hash_not_git_repo(self):
        """Test git commit hash when not in git repository."""
        with patch("subprocess.check_output") as mock_check_output:
            mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")

            commit_hash = get_git_commit_hash()
            assert commit_hash == "not-a-git-repo"


class TestRunSingleExperiment:
    """Test single experiment execution."""

    def test_run_single_experiment_basic(self):
        """Test running single experiment with basic configuration."""
        config = {
            "experiment": {"n": 1000, "seed": 42},
            "output": {
                "save_plot": False,  # Don't save plot in test
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = run_single_experiment(config, output_dir)

            assert result.n_total == 1000
            assert isinstance(result.pi_estimate, float)
            assert result.error >= 0.0

    def test_run_single_experiment_with_plot_saving(self):
        """Test running single experiment with plot saving."""
        config = {"experiment": {"n": 100, "seed": 42}, "output": {"save_plot": True}}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch(
                "monte_carlo_pi_experiment.get_git_commit_hash",
                return_value="test-commit",
            ):
                result = run_single_experiment(config, output_dir)

                assert result.n_total == 100

                # Check that PDF was created
                pdf_files = list(output_dir.glob("*.pdf"))
                assert len(pdf_files) == 1
                assert "test-commit" in pdf_files[0].name


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_single_experiment(self):
        """Test complete workflow for single experiment."""
        config = {
            "experiment": {"n": 1000, "seed": 42, "visualization_sample": 500},
            "output": {"directory": "outputs", "save_plot": True},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Run experiment
            result = run_monte_carlo_experiment(
                config["experiment"]["n"],
                config["experiment"]["seed"],
                config["experiment"]["visualization_sample"],
            )

            # Verify results
            assert result.n_total == 1000
            assert len(result.points_inside) <= 500
            assert len(result.points_outside) <= 500
            assert 2.0 < result.pi_estimate < 4.0

            # Create and save visualization
            with patch(
                "monte_carlo_pi_experiment.get_git_commit_hash",
                return_value="integration-test",
            ):
                save_scatter_plot_pdf(
                    result,
                    output_dir / "integration_test.pdf",
                    "integration-test",
                )

                assert (output_dir / "integration_test.pdf").exists()

    def test_batch_experiment_workflow(self):
        """Test batch experiment workflow."""
        configs = [
            {"experiment": {"n": 100, "seed": 42}, "output": {"save_plot": False}},
            {"experiment": {"n": 200, "seed": 123}, "output": {"save_plot": False}},
            {"experiment": {"n": 300, "seed": 456}, "output": {"save_plot": False}},
        ]

        results = []
        for config in configs:
            result = run_monte_carlo_experiment(
                config["experiment"]["n"],
                config["experiment"]["seed"],
            )
            results.append(result)

        # Calculate batch statistics
        stats = calculate_pi_statistics(results)

        assert len(results) == 3
        assert stats["mean_estimate"] > 0
        assert stats["std_estimate"] >= 0
        assert stats["theoretical_pi"] == pytest.approx(np.pi)

        # Verify different experiments produced different results
        estimates = [r.pi_estimate for r in results]
        assert len(set(estimates)) > 1  # Should have different estimates
