#!/bin/bash
set -e

echo "Setting up Python environment for graniteR using UV..."

if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # UV can install to different locations, check common ones
    if [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Verify UV is now available
    if ! command -v uv &> /dev/null; then
        echo "ERROR: UV installation completed but command not found."
        echo "Please add UV to your PATH manually:"
        echo "  export PATH=\"\$HOME/.cargo/bin:\$PATH\""
        echo "  or"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        exit 1
    fi

    echo "UV successfully installed and configured!"
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
