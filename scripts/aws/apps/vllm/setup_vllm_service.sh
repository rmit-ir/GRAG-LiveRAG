#!/bin/bash
# Script to set up vLLM as a systemd service
# This script creates a systemd service file, enables it, and starts the service

set -e  # Exit immediately if a command exits with a non-zero status

# Read all parameters from environment variables with fallbacks
MODEL_ID=${MODEL_ID:-"tiiuae/falcon3-10b-instruct"}
PORT=${PORT:-8000}
API_KEY=${API_KEY:-""}

# Additional parameters with defaults
TENSOR_PARALLEL=${TENSOR_PARALLEL:-0}  # 0 means auto-detect GPU count
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-"true"}

echo "Setting up vLLM service with model: $MODEL_ID"
echo "Additional parameters:"
echo "  PORT: $PORT"
echo "  TENSOR_PARALLEL: $TENSOR_PARALLEL"
echo "  MAX_NUM_BATCHED_TOKENS: $MAX_NUM_BATCHED_TOKENS"
echo "  GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"
echo "  ENABLE_CHUNKED_PREFILL: $ENABLE_CHUNKED_PREFILL"

# Create directory for vLLM service
sudo mkdir -p /opt/vllm

# Create a script to run vLLM
cat << EOF | sudo tee /opt/vllm/run_vllm.sh
#!/bin/bash
# Script to run vLLM server

# Detect GPU count if TENSOR_PARALLEL is set to 0 (auto)
if [ "$TENSOR_PARALLEL" -eq 0 ]; then
    GPU_COUNT=\$(nvidia-smi -L | wc -l)
    TENSOR_PARALLEL_SIZE=\$GPU_COUNT
    echo "Auto-detected \$GPU_COUNT GPU(s), using tensor parallel size: \$TENSOR_PARALLEL_SIZE"
else
    TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL
    echo "Using specified tensor parallel size: \$TENSOR_PARALLEL_SIZE"
fi

# Determine if chunked prefill should be enabled
CHUNKED_PREFILL_ARG=""
if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    CHUNKED_PREFILL_ARG="--enable-chunked-prefill"
fi

python3 -m vllm.entrypoints.openai.api_server \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --model $MODEL_ID \\
    --api-key $API_KEY \\
    --tensor-parallel-size \$TENSOR_PARALLEL_SIZE \\
    \$CHUNKED_PREFILL_ARG \\
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \\
    --distributed-executor-backend mp \\
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
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
