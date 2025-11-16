#!/bin/bash
set -e

echo "Setting up Python environment for graniteR using UV..."

if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

VENV_PATH="${VENV_PATH:-.venv}"

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
    else
        echo "Using existing virtual environment"
        source "$VENV_PATH/bin/activate"
        exit 0
    fi
fi

echo "Creating virtual environment..."
uv venv "$VENV_PATH"

source "$VENV_PATH/bin/activate"

echo "Installing Python dependencies with UV..."
REQUIREMENTS_FILE="inst/python/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    uv pip install -r "$REQUIREMENTS_FILE"
else
    echo "Requirements file not found at $REQUIREMENTS_FILE"
    echo "Installing core dependencies..."
    uv pip install transformers torch datasets numpy
fi

echo ""
echo "Python environment setup complete!"
echo "Virtual environment location: $VENV_PATH"
echo ""
echo "To use this environment in R, add to your .Rprofile or script:"
echo "  Sys.setenv(RETICULATE_PYTHON = \"$(pwd)/$VENV_PATH/bin/python\")"
echo ""
echo "Or activate it in your shell:"
echo "  source $VENV_PATH/bin/activate"
