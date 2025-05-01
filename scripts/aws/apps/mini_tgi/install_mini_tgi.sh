#!/bin/bash
# Script to install dependencies for mini-TGI
set -e

echo "Installing mini-TGI dependencies..."
python3 -m pip install torch transformers fastapi uvicorn accelerate psutil setproctitle 'jinja2>=3.1.0'

echo "Mini-TGI dependencies installed successfully!"
