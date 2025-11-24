#!/bin/bash

echo "Cleaning up..."
echo "================"
echo ""

# Stop containers
echo "Stopping Docker containers..."
docker-compose down

# Remove uploaded files
echo "Removing uploaded files..."
rm -rf data/uploads/*
rm -rf data/train/*

# Keep .gitkeep files
touch data/uploads/.gitkeep
touch data/train/.gitkeep

# Remove test files
echo "Removing test files..."
rm -f test_image.png

echo ""
echo "Cleanup complete!"