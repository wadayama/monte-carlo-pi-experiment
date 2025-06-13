# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a Monte Carlo method for calculating π (pi) as a practice exercise for experimental program development using Claude Code. The implementation should follow the structure and coding patterns from the `test_proj6` reference project.

## Development Environment

This project uses Python with `uv` for package management and virtual environment handling.

### Common Commands

**Environment Setup:**
```bash
# Create virtual environment
uv venv

# Activate virtual environment  
source .venv/bin/activate

# Install dependencies
uv sync --frozen

# Add new packages
uv add numpy pandas          # runtime dependencies
uv add -d pytest ruff pyright  # development dependencies
```

**Code Quality and Testing:**
```bash
# Automated test scripts (recommended)
./run_tests.sh          # Complete quality check with detailed output
./quick_test.sh         # Quick check for development iteration

# Manual commands
uv run ruff format .    # Format code
uv run ruff check .     # Lint code
uv run pyright          # Type checking
uv run pytest          # Run tests

# All manual checks combined (use when "test" is requested)
uv run ruff format . && uv run ruff check . && uv run pyright && uv run pytest
```

**Git Workflow:**
```bash
# Create private GitHub repository (do this first)
gh repo create --private

# Commit with conventional prefixes
git commit -m "feat: implement monte carlo pi calculation"
git commit -m "fix: correct boundary condition handling"
git commit -m "docs: update README with usage examples"
git commit -m "test: add edge case tests for calculation"
```

## Development Guidelines

### Code Standards
- **Languages**: Code, comments, docstrings, and all outputs must be in English
- **Type Hints**: Add type hints to all functions and variables
- **Docstrings**: Use NumPy style docstrings for all public functions/classes
- **Function Design**: Keep functions small (20-30 lines), follow single responsibility principle
- **Pure Functions**: Separate calculation logic from I/O operations for better testability

### Project Structure
Follow this recommended directory structure:
```
project-root/
├── pyproject.toml          # Project configuration & dependencies
├── uv.lock                 # Lock file for reproducible builds
├── README.md               # Project documentation
├── config/                 # Experiment parameter configuration (YAML files)
├── src/                    # Source code modules
├── tests/                  # Test code
├── outputs/                # Experiment results with timestamped directories
└── notebooks/              # Exploratory analysis (if needed)
```

### Configuration Management
- Store experiment parameters in YAML files (not hardcoded)
- Include random seeds in configuration for reproducibility
- Save configuration snapshots with experiment outputs

### Reproducibility Requirements
- Include git commit hash in experiment outputs
- Use consistent directory structure for results
- All experiments should be reproducible via `uv sync --frozen`

## Implementation Requirements

The project should implement Monte Carlo π calculation following the patterns established in `test_proj6`:
- Similar file structure and organization
- Consistent coding patterns and style
- Equivalent test coverage and design
- Same configuration management approach
- Matching documentation structure

Reference the detailed Python development guidelines in `python_dev.md` for specific coding standards and best practices.