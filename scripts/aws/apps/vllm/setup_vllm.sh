#!/bin/bash
# Script to install vLLM directly on an EC2 instance using pip
# This avoids the need for Docker, which has a large image size

set -e  # Exit immediately if a command exits with a non-zero status

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip setuptools wheel vllm flashinfer-python==0.2.2

# Verify installation
echo "Verifying vLLM installation..."
if python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" &> /dev/null; then
    echo "vLLM installed successfully!"
else
    echo "ERROR: vLLM installation failed!"
    exit 1
fi

echo "Installation complete!"
