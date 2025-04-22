#!/bin/bash
# Script to set up vLLM as a systemd service
# This script creates a systemd service file, enables it, and starts the service

set -e  # Exit immediately if a command exits with a non-zero status

# Default values
MODEL_ID=${1:-"tiiuae/Falcon3-10B-Instruct"}
PORT=${2:-8000}
API_KEY=${3:-""}

echo "Setting up vLLM service with model: $MODEL_ID"

# Create directory for vLLM service
sudo mkdir -p /opt/vllm

# Create a script to run vLLM
cat << EOF | sudo tee /opt/vllm/run_vllm.sh
#!/bin/bash
# Script to run vLLM server
export MODEL="$MODEL_ID"
export VLLM_API_KEY="$API_KEY"

# Detect GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)

echo "Detected $GPU_COUNT GPU(s)"

python3 -m vllm.entrypoints.openai.api_server \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --model \$MODEL \\
    --api-key \$VLLM_API_KEY \\
    --tensor-parallel-size \$GPU_COUNT \\
    --enable-chunked-prefill \\
    --max-num-batched-tokens 2048 \\
    --distributed-executor-backend mp \\
    --gpu-memory-utilization 0.95
EOF

# Make the script executable
sudo chmod +x /opt/vllm/run_vllm.sh

# Create systemd service file
cat << EOF | sudo tee /etc/systemd/system/vllm.service
[Unit]
Description=vLLM API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/vllm
ExecStart=/opt/vllm/run_vllm.sh
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable vllm.service

# Start the service
sudo systemctl start vllm.service

# Check service status
echo "Checking vLLM service status..."
sudo systemctl status vllm.service --no-pager

echo "vLLM service setup complete!"
echo "Service is running on port $PORT"
echo "To check logs: journalctl -u vllm.service -f"
