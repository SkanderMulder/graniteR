#!/bin/bash

set -e

echo "Fixing CUDA setup for WSL2..."

# Remove old CUDA stub libraries that conflict with Windows driver
echo "Step 1: Removing old CUDA libraries..."
sudo apt-get remove --purge -y 'cuda-*' 'nvidia-*' || true
sudo apt-get autoremove -y

# Clean up old library symlinks
sudo rm -f /usr/lib/x86_64-linux-gnu/libcuda.so*
sudo rm -f /usr/lib/x86_64-linux-gnu/libnvidia*.so*

# Install CUDA toolkit without driver components (WSL2 uses Windows driver)
echo "Step 2: Installing CUDA keyring..."
sudo dpkg -i cuda-keyring_1.1-1_all.deb

echo "Step 3: Updating package list..."
sudo apt-get update

echo "Step 4: Installing CUDA toolkit (no driver)..."
sudo apt-get install -y cuda-toolkit-12-6

# Set up environment variables for WSL2
echo "Step 5: Configuring environment..."
cat >> ~/.bashrc << 'EOL'

# CUDA configuration for WSL2
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
EOL

echo ""
echo "CUDA fix complete!"
echo "Run 'source ~/.bashrc' or restart your terminal"
echo "Then test with: python3 -c 'import torch; print(torch.cuda.is_available())'"
