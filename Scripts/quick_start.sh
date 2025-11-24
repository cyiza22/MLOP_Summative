#!/bin/bash

echo " Quick Start - ML Pipeline"
echo "============================"
echo ""

# Step 1: Setup
echo "Step 1: Setting up environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

# Step 2: Train model (if not exists)
if [ ! -f "models/cifar10_classifier.h5" ]; then
    echo ""
    echo " Model not found. Please train the model first:"
    echo "   bash scripts/train_model.sh"
    echo ""
    echo "Or run the notebook manually:"
    echo "   jupyter notebook notebook/ml_pipeline.ipynb"
    exit 1
fi

# Step 3: Start API
echo ""
echo "Step 2: Starting API..."
python app.py &
API_PID=$!

sleep 5

# Step 4: Test
echo ""
echo "Step 3: Testing API..."
curl http://localhost:5000/health

echo ""
echo ""
echo "============================"
echo " Quick Start Complete!"
echo " API running at: http://localhost:5000"
echo " To stop: kill $API_PID"
echo "============================"

