#!/bin/bash
# Setup script for AI Battle framework documentation

set -e  # Exit on error

echo "Setting up documentation environment for AI Battle framework..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "docs_venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv docs_venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source docs_venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r docs/requirements.txt

# Create necessary directories
echo "Creating documentation directories..."
mkdir -p docs/api/core
mkdir -p docs/api/configuration
mkdir -p docs/api/model_clients
mkdir -p docs/api/file_handling
mkdir -p docs/api/metrics
mkdir -p docs/api/arbiter
mkdir -p docs/api/utilities
mkdir -p docs/api/examples

# Generate documentation
echo "Generating documentation..."
python docs/generate_docs.py

# Build documentation with MkDocs
echo "Building documentation with MkDocs..."
mkdocs build

echo "Documentation setup complete!"
echo "To view the documentation, run: mkdocs serve"
echo "Then open http://localhost:8000 in your browser."