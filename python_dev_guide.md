# Python Development Guidelines for Scientific Computing

## 1. Introduction

### 1.1. Purpose of These Guidelines
This document provides development guidelines to enhance the quality and efficiency of scientific computing conducted in this project. We prioritize three fundamental pillars: **Correctness**, **Readability**, and **Reproducibility**.

### 1.2. Basic Principles
* **Accuracy > Speed**: In scientific computing, result accuracy takes priority over computational speed.
* **For future self and others**: Code is not just written once but will be read many times later.
* **Always reproducible by anyone**: Aim for a state where the same experiment can be immediately reproduced in the same environment even 10 years later.

---

## 2. Development Environment Setup and Management

**A consistent development environment is the first step toward reproducibility. Use `uv` to build and manage independent environments for each project.**

### 2.1. Virtual Environment
Execute the following commands in the project root directory to create a virtual environment.
```bash
# Create virtual environment
uv venv

# Activate virtual environment (Windows/Linux/macOS common)
source .venv/bin/activate
```

### 2.2. Package Management
Unify package management with `uv` and avoid concurrent use of `pip` or `conda`. Manage dependencies centrally with `pyproject.toml`. To ensure reproducibility, specify the Python version required by the project in pyproject.toml.

* **Adding packages**:
    ```bash
    # Add packages required for execution
    uv add numpy pandas
    
    # Add packages needed only during development (tests, formatters, etc.)
    uv add -d pytest ruff-lsp
    ```
* **Environment synchronization**:
    With `pyproject.toml` and `uv.lock` files, anyone can reproduce the same environment with the following command.
    ```bash
    uv sync --frozen
    ```
* **Tool execution**:
    Execute installed tools using `uv run`.
    ```bash
    uv run pytest
    uv run ruff format .
    ```

---

## 3. Code Quality Standards

### 3.1. Quality Assurance through Static Analysis
**Maximize the use of automatic tool checks before human review.**

* **Formatting**: Use `ruff format` to unify code style.
* **Linting**: Use `ruff check` to detect potential bugs and deprecated practices.
* **Type Checking**:
    * Add **Type Hints** to code as much as possible.
    * Use `pyright` to statically verify type hint consistency.
    
**※ We recommend consolidating these tool settings in `pyproject.toml` and sharing them with the team.**

### 3.2. Ensuring Accuracy through Testing
**The principle is to write tests for all functions containing logic, not just "important functions".**

* **Test Framework**: Use `pytest`.
* **Testing Principles**:
    * **Small and focused**: Each test function should verify only one thing.
    * **Boundary values**: Prepare test cases for not only normal cases but also abnormal cases and boundary values.
    * **Side effect isolation**: Design processes containing "side effects" such as random number generation, current time acquisition, and file I/O separated from the core logic. This makes testing the logic portion easier.

### 3.3. Improving Readability
* **Documentation Strings (Docstring)**:
    * Write Docstrings for all public modules, functions, classes, and methods.
    * **Numpy style is recommended.** This clarifies arguments, return values, and processing content.
    * **Write Docstrings in English.** Considering international collaboration and future use, English description is the standard.
* **Comments**: Use comments to explain "why" that code is necessary and the intention behind complex logic. Comments explaining "what" that can be understood from reading the code are unnecessary.
    * **Comments should also be written in English.** This assumes team development and use in international environments.
* **Naming Conventions**: **Follow PEP 8 and use clear variable and function names.**
    * **Use English for variable and function names.** Avoid romaji naming and use appropriate English words.

### 3.4. Internationalization and Usability
* **Program Output Language**:
    * **Write console output, log messages, and error messages in English.**
    * Program execution results and progress displays should also be in English as standard.
    * Naming: variable names, function names, class names (※ Romaji notation is not allowed)
* **Graphs and Plots**:
    * **Write axis labels, titles, and legends in English.**
    * Charts created with matplotlib etc. should also use English notation as standard, considering international use.
* **File Output**:
    * Unify CSV headers, PDF titles, etc. in English.

### 3.5. Function Design Principles

### 3.5.1. Division into Small Functions
From the perspective of maintainability and readability, keep each function as small as possible.
* Single Responsibility Principle: One function should have only one responsibility
* Function length: Generally aim for within 20-30 lines, designed to fit on screen
* Decomposition of complex processing: Divide complex algorithms into functions by meaningful units

#### Bad: Large and complex function
```python
def analyze_data(data_path: str) -> dict:
    """Analyze data from file (too complex)."""
    # Data loading (10 lines)
    # Data preprocessing (15 lines)  
    # Statistical calculation (20 lines)
    # Visualization (15 lines)
    # Result saving (10 lines)
    pass  # Total 70 lines of complex processing
```
#### Good: Divided into small functions
```python
def load_data(data_path: str) -> pd.DataFrame:
    """Load data from specified path."""
    return pd.read_csv(data_path)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the input data."""
    # Preprocessing logic
    return cleaned_data

def calculate_statistics(data: pd.DataFrame) -> dict:
    """Calculate basic statistical measures."""
    # Statistical calculation logic
    return statistics

def create_visualization(data: pd.DataFrame, stats: dict) -> plt.Figure:
    """Create visualization plots."""
    # Visualization logic
    return figure

def save_results(stats: dict, figure: plt.Figure, output_dir: Path) -> None:
    """Save analysis results to files."""
    # Saving logic
    pass

def analyze_data(data_path: str, output_dir: Path) -> dict:
    """Analyze data with clear workflow."""
    data = load_data(data_path)
    cleaned_data = preprocess_data(data)
    stats = calculate_statistics(cleaned_data)
    figure = create_visualization(cleaned_data, stats)
    save_results(stats, figure, output_dir)
    return stats
```

### 3.5.2 Referential Transparency and Side Effect Separation
Clearly separate computational logic (pure functions) from I/O processing (functions with side effects).
#### Characteristics of pure functions with referential transparency
* Same output for same input: Function execution results depend only on input
* No side effects: Do not perform global variable changes, file operations, network communications, etc.
* Easy to test: Clear relationship between input and output, no mocks needed

### 3.5.3. Function Design Best Practices

* Function names start with verbs: calculate_, process_, validate_, etc.
* Argument order: Important arguments first, optional arguments later
* Default arguments: Do not use mutable objects (lists, dictionaries) as default values
* Use type hints: Clarify input/output types to make function contracts clear

### 3.5.4 Avoid Using Global Variables as Much as Possible

To increase referentially transparent parts, minimize the use of global variables.
Global variables make function behavior unpredictable, make testing difficult, and can cause race conditions during parallel processing.
However, global variables such as physical constants or immutable constants within programs may be used to improve code readability.

## 4. Ensuring Reproducibility

### 4.1. Version Control
* **Manage all code and configuration files with Git.**
* First, create a private repository on Github using the gh command.
* Commit changes in meaningful units with clear messages using prefixes like `feat:`, `fix:`, `docs:`, `test:`.
* Local checks before commit: Automatically run uv run ruff format ., uv run ruff check ., uv run pyright, uv run pytest and fix any issues.
* When instructed to "test", execute the following including tests: uv run ruff format ., uv run ruff check ., uv run pyright, uv run pytest

### 4.2. Experiment Parameter Management
* Key experimental parameters such as learning rates, data paths, and iteration counts should not be hardcoded directly in code but **managed separately in YAML files.**
* **When using random numbers, also record the seed value in the YAML file** and read it in code to set it, ensuring result reproducibility.

### 4.3. Data Management
* **Manage input and output data with a clear directory structure.** Below is an example.
    ```
    project-root/
    ├── pyproject.toml          # Project settings and dependencies
    ├── uv.lock                 # Lock file
    ├── README.md               # Project description
    ├── .gitignore              # Git exclusion settings
    ├── config/                 # Experiment parameter settings
    │   ├── default.yaml
    │   └── experiment_*.yaml
    ├── data/
    │   ├── raw/                # Original data (read-only)
    │   ├── interim/            # Intermediate processing data
    │   └── processed/          # Final input data
    ├── src/
    │   ├── __init__.py
    │   ├── data/               # Data processing modules
    │   ├── models/             # Model definitions
    │   ├── experiments/        # Experiment scripts
    │   └── utils/              # Utility functions
    ├── tests/                  # Test code
    │   ├── __init__.py
    │   ├── test_data/
    │   ├── test_models/
    │   └── test_utils/
    ├── outputs/                # Experiment result output
    │   ├── 2025-06-12_experiment-A/
    │   │   ├── config.yaml     # Copy of used configuration
    │   │   ├── results.csv
    │   │   ├── figures/
    │   │   └── logs/
    │   └── ...
    └── notebooks/              # For exploratory data analysis
        ├── 01_data_exploration.ipynb
        └── 02_result_analysis.ipynb
    ```

* We recommend including the Git commit hash at the time of experiment execution in experiment result output files and generated log files.
This allows unique identification of which code version generated that result.
With the commit hash, you can return to the code at that commit time using the git checkout command.
Below is reference code.
```python
import subprocess
import logging

def get_git_commit_hash() -> str:
    """
    Get the commit hash of the current Git repository.
    If there are uncommitted changes, append '-dirty' to the end of the hash.
    """
    try:
        # Get the latest commit hash with git rev-parse HEAD command
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip().decode('utf-8')

        # Check for uncommitted changes with git status --porcelain command
        # If there are changes, there will be output; if clean, no output
        status = subprocess.check_output(
            ['git', 'status', '--porcelain']
        ).strip().decode('utf-8')

        # If there are uncommitted changes, add '-dirty' to indicate incomplete reproducibility
        if status:
            return f"{commit_hash}-dirty"
        else:
            return commit_hash

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a Git repository or git command not found
        return "not-a-git-repo"

# --- Usage ---

# 1. Get commit hash at experiment start
commit_id = get_git_commit_hash()

# 2. Output to log
logging.info(f"Experiment running on git commit: {commit_id}")
if commit_id.endswith('-dirty'):
    logging.warning("Working directory has uncommitted changes. Reproducibility might be compromised.")
```

### 4.4. Reproducing Experiment Environment
* Enable experiment environment reproduction by executing uv sync --frozen.
* This command references the uv.lock file to install necessary packages.

---

## 5. Documentation

### 5.1. Creating README.md
**Always create a `README.md` for projects and organize information so third parties can easily understand and execute.**

#### 5.1.1. Content to Include in README.md
1. **Project Overview**
   * Concisely explain the purpose of the experiment/program
   * Include background and theoretical basis if applicable

2. **File Structure**
   * Explain the role of major files in the project
   * Visually show directory structure

3. **Environment Setup Procedure**
   * Prerequisites (Python version, required tools)
   * Environment setup procedure using `uv sync`
   * Include specific command examples

4. **Execution Method**
   * Main program execution commands
   * Execution examples and expected output examples
   * Parameter setting methods

5. **Test Execution Method**
   * Test execution commands using `pytest`
   * Explanation of test content and purpose

6. **Code Quality Checks**
   * How to execute `ruff format`, `ruff check`, `pyright`
   * Quality assurance procedures during development

#### 5.1.2. Notes for Creating README.md
* **Save in UTF-8 encoding**: Always save in UTF-8 encoding when including Japanese
* **Use markdown notation**: Properly use code blocks, lists, and headings
* **Present specific examples**: Show actual commands and output examples rather than abstract explanations
* **Continue updates**: Update README along with project changes

#### 5.1.3. Recommended Structure
```markdown
# Project Name

## Overview
## Experiment Content (if applicable)
## File Structure
## Environment Setup
### Prerequisites
### Setup Procedure
## Execution Method
### Basic Execution
### Execution Examples
## Parameter Settings (if applicable)
## Test Execution
## Code Quality Checks
## Program Structure (if complex)
## Theoretical Background (for scientific computing)
## Notes
## License
```

### 5.2. Encoding Notes
* **Always use UTF-8 encoding**: Especially for files containing Japanese, check editor settings
* **UTF-8 without BOM recommended**: BOM-included files can cause issues with some tools
* **Verification after creation**: Recommend verifying encoding with `file` command
  ```bash
  file README.md  # Should display "Unicode text, UTF-8 text"
  ```
---