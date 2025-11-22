#!/bin/bash

echo "ML Pipeline Deployment Script"
echo "=============================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
docker build -t ml-api:latest .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

echo "Docker image built successfully"
echo ""

# Start containers
echo "Starting containers..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "Error: Failed to start containers"
    exit 1
fi

echo "Containers started successfully"
echo ""

# Wait for API to be ready
echo "Waiting for API to be ready..."
sleep 10

# Health check
response=$(curl -s http://localhost:5000/health)
if [ $? -eq 0 ]; then
    echo "API is healthy!"
    echo "Response: $response"
else
    echo "Warning: API health check failed"
fi

echo ""
echo "=============================="
echo "Deployment Complete!"
echo "API available at: http://localhost:5000"
echo "=============================="