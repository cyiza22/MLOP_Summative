#!/bin/bash

echo "Setting up on Google Cloud Platform"
echo "===================================="
echo ""

PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="ml-api"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com

# Build and push image
echo "Building and pushing Docker image..."
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --max-instances 10

echo ""
echo "Deployment to GCP complete!"
