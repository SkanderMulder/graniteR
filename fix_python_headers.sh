#!/bin/bash

set -e

echo "Installing Python development headers for CUDA compilation..."

# Install Python development headers
sudo apt-get update
sudo apt-get install -y python3.12-dev python3-dev build-essential

echo ""
echo "Python headers installed successfully!"
echo "You can now run your training code."
