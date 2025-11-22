#!/bin/bash

echo "Setting up on AWS"
echo "================="
echo ""

REGION="us-east-1"
CLUSTER_NAME="ml-api-cluster"
SERVICE_NAME="ml-api-service"

# Create ECS cluster
echo "Creating ECS cluster..."
aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION

# Build and push to ECR
echo "Building and pushing to ECR..."
aws ecr create-repository --repository-name $SERVICE_NAME --region $REGION

# Get ECR URI
ECR_URI=$(aws ecr describe-repositories --repository-names $SERVICE_NAME --region $REGION --query 'repositories[0].repositoryUri' --output text)

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

# Build and push
docker build -t $SERVICE_NAME:latest .
docker tag $SERVICE_NAME:latest $ECR_URI:latest
docker push $ECR_URI:latest

echo ""
echo "Image pushed to ECR: $ECR_URI:latest"
echo "Next: Create ECS task definition and service using AWS Console"