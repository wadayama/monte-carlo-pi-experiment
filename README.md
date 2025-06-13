# Monte Carlo π Estimation Experiment

## Overview

This project implements a Monte Carlo method for estimating the value of π (pi) using random sampling. The experiment generates random points within a unit square and determines what fraction fall within the inscribed unit circle to estimate π.

## Theoretical Background

The Monte Carlo method for π estimation is based on the geometric relationship between a unit circle and its circumscribing square:

- Unit circle area: π/4
- Unit square area: 1
- Ratio: π/4

By randomly sampling points in the unit square [0,1] × [0,1] and counting how many fall within the unit circle (x² + y² ≤ 1), we can estimate π as:

```
π ≈ 4 × (points inside circle) / (total points)
```

## File Structure

```
monte-carlo-pi-experiment/
├── pyproject.toml                          # Project configuration & dependencies
├── uv.lock                                 # Locked dependencies for reproducibility
├── monte_carlo_pi_experiment.py            # Main experiment module
├── config/
│   ├── experiment.yaml                     # Default experiment configuration
│   └── experiments/                        # Multiple experiment variants
│       ├── small_experiment.yaml           # Quick test (1,000 points)
│       ├── medium_experiment.yaml          # Standard precision (100,000 points)
│       ├── large_experiment.yaml           # High precision (1,000,000 points)
│       └── precision_comparison.yaml       # Ultra-high precision (10,000,000 points)
├── tests/
│   ├── __init__.py
│   └── test_monte_carlo_pi_experiment.py   # Comprehensive test suite
├── outputs/                                # Generated results (PDFs with commit hashes)
├── CLAUDE.md                               # Claude Code development guidelines
└── README.md                               # This file
```

## Environment Setup

### Prerequisites
- Python 3.11 or higher
- `uv` package manager

### Installation

1. **Clone or navigate to the project directory**

2. **Create and activate virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv sync --frozen
   ```

## Execution Methods

### Single Experiment

Run a single experiment using the default configuration:
```bash
uv run python monte_carlo_pi_experiment.py
```

Run with a specific configuration:
```bash
uv run python monte_carlo_pi_experiment.py --config config/experiments/small_experiment.yaml
```

### Batch Experiments

Run all experiments in the `config/experiments/` directory:
```bash
uv run python monte_carlo_pi_experiment.py --batch
```

### Command Line Options

```bash
uv run python monte_carlo_pi_experiment.py --help
```

Available options:
- `--config PATH`: Specify configuration file (default: `config/experiment.yaml`)
- `--batch`: Run all configurations in `config/experiments/` directory
- `--output-dir PATH`: Set output directory (default: `outputs`)
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Example Execution

```bash
# Quick test with 1,000 points
uv run python monte_carlo_pi_experiment.py --config config/experiments/small_experiment.yaml

# High precision with 1,000,000 points
uv run python monte_carlo_pi_experiment.py --config config/experiments/large_experiment.yaml

# Run all predefined experiments
uv run python monte_carlo_pi_experiment.py --batch
```

## Configuration

Experiments are configured using YAML files with the following structure:

```yaml
experiment:
  n: 100000                    # Number of random points
  seed: 42                     # Random seed for reproducibility
  visualization_sample: 5000   # Max points to store for visualization

output:
  directory: "outputs"         # Output directory
  save_plot: true             # Whether to save scatter plot
```

### Predefined Configurations

- **small_experiment.yaml**: 1,000 points - Quick testing
- **medium_experiment.yaml**: 100,000 points - Standard precision
- **large_experiment.yaml**: 1,000,000 points - High precision
- **precision_comparison.yaml**: 10,000,000 points - Ultra-high precision

## Test Execution

Run the complete test suite:
```bash
uv run pytest
```

Run specific test categories:
```bash
uv run pytest tests/test_monte_carlo_pi_experiment.py::TestRunMonteCarloExperiment
uv run pytest -v  # Verbose output
```

## Code Quality Checks

### Automated Test Scripts

**Complete Quality Check (recommended):**
```bash
./run_tests.sh
```
Runs all quality checks with detailed output: formatting, linting, type checking, tests, coverage analysis, and functional verification.

**Quick Development Check:**
```bash
./quick_test.sh
```
Runs essential checks only for rapid development iteration: formatting, linting, and tests.

### Manual Commands

**Format Code:**
```bash
uv run ruff format .
```

**Lint Code:**
```bash
uv run ruff check .
```

**Type Checking:**
```bash
uv run pyright
```

**Run All Manual Checks:**
```bash
uv run ruff format . && uv run ruff check . && uv run pyright && uv run pytest
```

## Program Structure

### Core Functions

- **`generate_random_point(rng)`**: Generate random point in unit square
- **`check_inside_circle(point)`**: Determine if point is inside unit circle
- **`run_monte_carlo_experiment(n, seed, visualization_sample)`**: Execute Monte Carlo sampling
- **`create_scatter_plot(result)`**: Create visualization of sampling points
- **`calculate_pi_statistics(results)`**: Compute statistical measures for multiple runs

### Key Features

- **Reproducibility**: All experiments use fixed random seeds
- **Visualization**: Scatter plots show points inside/outside the unit circle
- **Git Integration**: Output files include git commit hashes for tracking
- **Batch Processing**: Run multiple experiment configurations simultaneously
- **Comprehensive Testing**: Unit tests, integration tests, and edge case handling

## Expected Results

The accuracy of π estimation improves with the number of sample points:

| Points      | Typical Error | Execution Time |
|-------------|---------------|----------------|
| 1,000       | ~0.01-0.1     | < 1 second     |
| 100,000     | ~0.001-0.01   | < 1 second     |
| 1,000,000   | ~0.0001-0.001 | ~2 seconds     |
| 10,000,000  | ~0.00001-0.0001| ~20 seconds   |

## Output Files

Results are saved as PDF files in the `outputs/` directory with naming convention:
```
monte_carlo_pi_n{points}_seed{seed}_{git_commit_hash}.pdf
```

Each PDF contains:
- Scatter plot of sampled points (red: inside circle, blue: outside)
- Unit circle and square boundaries
- Experiment results and statistics
- Git commit hash for reproducibility

## Reproducibility

This project ensures reproducibility through:

1. **Fixed random seeds** in configuration files
2. **Locked dependencies** via `uv.lock`
3. **Git commit tracking** in all output files
4. **Comprehensive configuration management**

To reproduce any experiment, use the same configuration file and git commit:
```bash
git checkout <commit_hash>
uv sync --frozen
uv run python monte_carlo_pi_experiment.py --config <config_file>
```

## Development Guidelines

This project follows strict scientific computing standards:

- **English-only**: All code, comments, and outputs in English
- **Type hints**: Comprehensive type annotations
- **Pure functions**: Separation of calculation logic from I/O
- **Comprehensive testing**: Unit and integration tests
- **Documentation**: NumPy-style docstrings

See `CLAUDE.md` for detailed development guidelines.