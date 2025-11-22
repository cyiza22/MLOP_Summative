#!/bin/bash

echo "Testing ML API Endpoints"
echo "========================"
echo ""

API_URL="http://localhost:5000"

# Test health endpoint
echo "1. Testing Health Endpoint..."
curl -X GET $API_URL/health
echo -e "\n"

# Test metrics endpoint
echo "2. Testing Metrics Endpoint..."
curl -X GET $API_URL/metrics
echo -e "\n"

# Test prediction endpoint with sample image
echo "3. Testing Prediction Endpoint..."
# Create a test image (requires Python)
python3 << EOF
from PIL import Image
import numpy as np

# Create random test image
img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save('test_image.png')
print("Test image created")
EOF

# Send prediction request
curl -X POST -F "image=@test_image.png" $API_URL/predict
echo -e "\n"

# Clean up
rm -f test_image.png

echo ""
echo "========================"
echo "API Testing Complete!"
echo "========================"
