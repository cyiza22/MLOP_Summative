#!/bin/bash

echo "🎓 Training ML Model"
echo "===================="
echo ""

# Check if notebook exists
if [ ! -f "notebook/ml_pipeline.ipynb" ]; then
    echo "Notebook not found: notebook/ml_pipeline.ipynb"
    exit 1
fi

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "Installing jupyter..."
    pip install jupyter
fi

echo "Starting Jupyter Notebook..."
echo "Please run all cells in the notebook to train the model"
echo ""
echo "The notebook will:"
echo "  1. Load CIFAR-10 dataset"
echo "  2. Perform EDA and create visualizations"
echo "  3. Train the model with transfer learning"
echo "  4. Evaluate and save the model"
echo ""
jupyter notebook notebook/ml_pipeline.ipynb