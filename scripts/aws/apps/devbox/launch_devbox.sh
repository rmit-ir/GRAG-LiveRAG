#!/bin/bash
# Launch script for devbox EC2 app
# This script starts FileBrowser with Docker and disables authentication

set -e

echo "Starting FileBrowser with Docker..."

# Stop any existing FileBrowser container
sudo docker stop filebrowser 2>/dev/null || true
sudo docker rm filebrowser 2>/dev/null || true

# Start FileBrowser with Docker
sudo docker run -d \
    --name filebrowser \
    --restart unless-stopped \
    -v /home/ubuntu:/srv \
    -v /opt/filebrowser/database.db:/database.db \
    -v /opt/filebrowser/config/filebrowser.json:/.filebrowser.json \
    -u $(id -u):$(id -g) \
    -p 8000:8080 \
    filebrowser/filebrowser:v2.32.0

# Wait for FileBrowser to start
sleep 2
sudo docker logs filebrowser

echo "No authentication is required to access the file browser."
