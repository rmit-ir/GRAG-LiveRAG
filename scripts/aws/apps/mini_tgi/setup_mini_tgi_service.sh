#!/bin/bash
# Script to set up mini-TGI as a systemd service
set -e

# Read all parameters from environment variables with fallbacks
PORT=${PORT:-8000}
API_KEY=${API_KEY:-""}
PROGRAM_FILE=${PROGRAM_FILE:-"/tmp/llm_server.py"}
MODEL_ID=${MODEL_ID:-"tiiuae/falcon3-10b-instruct"}

# Additional parameters with defaults
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-64}

echo "Setting up mini-TGI service with the following parameters:"
echo "  PORT: $PORT"
echo "  MODEL_ID: $MODEL_ID"
echo "  PROGRAM_FILE: $PROGRAM_FILE"
echo "  MAX_BATCH_SIZE: $MAX_BATCH_SIZE"
echo "  NUM_WORKERS: $NUM_WORKERS"

# Create directory for mini-TGI service
sudo mkdir -p /opt/mini-tgi

# Copy the program file
sudo cp $PROGRAM_FILE /opt/mini-tgi/llm_server.py
sudo chmod +x /opt/mini-tgi/llm_server.py

# Create a script to run mini-TGI
cat << EOF | sudo tee /opt/mini-tgi/run_mini_tgi.sh
#!/bin/bash
# Script to run mini-TGI server
export API_KEY="$API_KEY"

cd /opt/mini-tgi
python3 llm_server.py --port $PORT --model "$MODEL_ID" --batch-size $MAX_BATCH_SIZE
EOF

# Make the script executable
sudo chmod +x /opt/mini-tgi/run_mini_tgi.sh

# Create systemd service file
cat << EOF | sudo tee /etc/systemd/system/mini-tgi.service
[Unit]
Description=Mini-TGI API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mini-tgi
ExecStart=/opt/mini-tgi/run_mini_tgi.sh
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable mini-tgi.service

# Start the service
sudo systemctl start mini-tgi.service

# Check service status
echo "Checking mini-TGI service status..."
sudo systemctl status mini-tgi.service --no-pager

echo "Mini-TGI service setup complete!"
echo "Service is running on port $PORT with model $MODEL"
echo "To check logs: journalctl -u mini-tgi.service -f"
