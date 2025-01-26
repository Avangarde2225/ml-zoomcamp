# AWS Deployment Guide

This guide explains how to deploy the Food Detection application to AWS App Runner.

## Prerequisites

1. Install AWS CLI and configure credentials:
```bash
aws configure  # Enter your AWS Access Key ID and Secret Access Key
```

2. Install Docker on your local machine.

## Deployment Steps

### 1. Build and Test Docker Image Locally

```bash
# Navigate to the project directory
cd food_detection_cpu

# Build the Docker image
docker build -t food-detection-app .

# Test locally
docker run -p 7860:7860 food-detection-app
```

### 2. Push to Amazon ECR

```bash
# Get your AWS account ID
aws sts get-caller-identity --query Account --output text

# Create ECR repository
aws ecr create-repository --repository-name food-detection-app

# Login to ECR
aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com

# Tag and push image
docker tag food-detection-app:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/food-detection-app:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/food-detection-app:latest
```

### 3. Deploy to AWS App Runner

1. Go to AWS Console > App Runner
2. Click "Create service"
3. Choose "Container registry" as source
4. Select your ECR repository and image
5. Configure service:
   - Service name: food-detection-service
   - Port: 7860
   - CPU: 1 vCPU
   - Memory: 2 GB
   - Instance role: None required
6. Click "Create & deploy"

## Monitoring and Maintenance

1. Monitor your application:
   - AWS Console > App Runner > Your service > Metrics
   - Check CPU usage, memory usage, and request counts

2. Update application:
   - Build new Docker image
   - Push to ECR with new tag
   - Deploy new version in App Runner console

## Cost Management

AWS App Runner charges based on:
- Compute resources (vCPU and memory)
- Provisioned concurrency
- Data transfer

Tips to minimize costs:
1. Use Auto-scaling to scale down during low traffic
2. Monitor usage and adjust resources accordingly
3. Consider using reserved instances for consistent workloads

## Troubleshooting

1. Check service logs in App Runner console
2. Verify security group settings
3. Check that port 7860 is exposed correctly
4. Ensure environment variables are set properly

## Security Considerations

1. Use IAM roles and policies
2. Keep AWS credentials secure
3. Regularly update dependencies
4. Monitor for unusual activity 