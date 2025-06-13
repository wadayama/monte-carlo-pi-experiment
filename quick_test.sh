#!/bin/bash

# Quick Test Script - Essential checks only
# For rapid development iteration

set -e

echo "⚡ Quick Quality Check"
echo "===================="

# Essential checks only
echo "🔧 Formatting..."
uv run ruff format . > /dev/null

echo "🔍 Linting..."
uv run ruff check .

echo "🧪 Testing..."
uv run pytest -q

echo "✅ Quick checks completed!"