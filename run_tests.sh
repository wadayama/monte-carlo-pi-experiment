#!/bin/bash

# Monte Carlo Pi Experiment - Test and Quality Check Script
# This script runs all quality checks and tests for the project

set -e  # Exit on any error

echo "🔧 Monte Carlo Pi Experiment - Quality Check & Test Suite"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BLUE}📋 $1${NC}"
    echo "----------------------------------------"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ ! -f ".venv/bin/activate" ]]; then
    print_warning "No virtual environment detected. Make sure to run 'uv venv && source .venv/bin/activate'"
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install uv first."
    exit 1
fi

# Install dependencies if needed
print_section "Checking Dependencies"
echo "Installing/updating dependencies..."
uv sync --frozen
print_success "Dependencies are up to date"

# 1. Code Formatting
print_section "Code Formatting (ruff format)"
if uv run ruff format .; then
    print_success "Code formatting completed"
else
    print_error "Code formatting failed"
    exit 1
fi

# 2. Linting
print_section "Code Linting (ruff check)"
if uv run ruff check .; then
    print_success "Linting passed"
else
    print_error "Linting failed"
    exit 1
fi

# 3. Type Checking
print_section "Type Checking (pyright)"
if uv run pyright; then
    print_success "Type checking passed"
else
    print_error "Type checking failed"
    exit 1
fi

# 4. Unit Tests
print_section "Running Tests (pytest)"
if uv run pytest -v; then
    print_success "All tests passed"
else
    print_error "Tests failed"
    exit 1
fi

# 5. Test Coverage (optional)
print_section "Test Coverage Analysis"
if uv run pytest --cov=monte_carlo_pi_experiment --cov-report=term-missing 2>/dev/null; then
    print_success "Coverage analysis completed"
else
    print_warning "Coverage analysis not available (pytest-cov not installed)"
fi

# 6. Quick Functional Test
print_section "Functional Test"
echo "Running small experiment to verify functionality..."
if uv run python monte_carlo_pi_experiment.py --config config/experiments/small_experiment.yaml > /dev/null 2>&1; then
    print_success "Functional test passed"
else
    print_error "Functional test failed"
    exit 1
fi

# Summary
echo ""
echo "🎉 All Quality Checks Completed Successfully!"
echo "============================================="
echo ""
print_success "✅ Code formatting"
print_success "✅ Linting"
print_success "✅ Type checking"
print_success "✅ Unit tests"
print_success "✅ Functional test"
echo ""
echo "Your code is ready for production! 🚀"