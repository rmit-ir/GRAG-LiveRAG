#!/bin/bash
# Setup script for devbox EC2 app
# This script sets up FileBrowser with Docker

set -e

echo "Setting up FileBrowser..."

# Create directories and files for FileBrowser
sudo mkdir -p /opt/filebrowser/config
# Create an empty database file (not a directory)
sudo touch /opt/filebrowser/database.db

# Create a default configuration file
cat << EOF | sudo tee /opt/filebrowser/config/filebrowser.json
{
  "port": 8080,
  "baseURL": "",
  "address": "",
  "log": "stdout",
  "database": "/database.db",
  "root": "/srv"
}
EOF

# Pull the FileBrowser Docker image
echo "Pulling FileBrowser Docker image..."
sudo docker pull filebrowser/filebrowser:v2.32.0

# Create a temporary container to initialize the database with a default user
echo "Initializing FileBrowser database with default user..."
sudo docker run --rm -v /opt/filebrowser/database.db:/database.db filebrowser/filebrowser:v2.32.0 -d /database.db config init
sudo docker run --rm -v /opt/filebrowser/database.db:/database.db filebrowser/filebrowser:v2.32.0 -d /database.db users add admin admin --perm.admin

# Set authentication method to noauth
echo "Setting authentication method to noauth..."
sudo docker run --rm -v /opt/filebrowser/database.db:/database.db filebrowser/filebrowser:v2.32.0 -d /database.db config set --auth.method=noauth

# FileBrowser setup is complete

echo "FileBrowser setup completed successfully!"
