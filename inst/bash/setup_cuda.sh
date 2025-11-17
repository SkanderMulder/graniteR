#!/bin/bash

set -e

echo "Installing CUDA toolkit on WSL2..."

# Install the CUDA keyring
echo "Step 1: Installing CUDA keyring..."
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list
echo "Step 2: Updating package list..."
sudo apt-get update

# Install CUDA toolkit
echo "Step 3: Installing CUDA toolkit 12.6..."
sudo apt-get install -y cuda-toolkit-12-6

# Add CUDA to PATH if not already present
echo "Step 4: Configuring PATH..."
if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

echo ""
echo "CUDA installation complete!"
echo "Run 'source ~/.bashrc' or restart your terminal, then verify with 'nvcc --version'"
