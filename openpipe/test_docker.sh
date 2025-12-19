#!/bin/bash
# Docker Test Guide for OpenPipe Setup
# Run this script to test the OpenPipe Docker environment

set -e  # Exit on error

echo "🚀 OpenPipe Docker Test Guide"
echo "=============================="
echo ""

# Step 1: Check Docker
echo "Step 1: Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    echo "   Install from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker is installed: $(docker --version)"

# Step 2: Check if Docker is running
echo ""
echo "Step 2: Checking if Docker daemon is running..."
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running"
    echo ""
    echo "📋 To start Docker:"
    echo "   1. Open Docker Desktop application"
    echo "   2. Wait for Docker to start (whale icon in menu bar)"
    echo "   3. Run this script again"
    exit 1
fi

echo "✅ Docker daemon is running"

# Step 3: Setup environment
echo ""
echo "Step 3: Setting up environment..."
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "⚠️  .env file not found, copying from .env.example"
    cp .env.example .env
    echo ""
    echo "📝 IMPORTANT: Edit openpipe/.env and add your WANDB_API_KEY"
    echo "   Get your key from: https://wandb.ai/authorize"
    echo ""
    read -p "Press Enter after you've added your WANDB_API_KEY to .env..."
fi

echo "✅ Environment file ready"

# Step 4: Build Docker image
echo ""
echo "Step 4: Building Docker image..."
echo "   This may take 5-10 minutes on first run..."
docker compose build

echo "✅ Docker image built successfully"

# Step 5: Test with minimal training
echo ""
echo "Step 5: Running quick test (2 training steps)..."
echo "   This will:"
echo "   - Load the financial dataset"
echo "   - Initialize the model"
echo "   - Run 2 training steps"
echo "   - Log to W&B"
echo ""

docker compose run --rm openpipe-training python3 -c "
import asyncio
import os
from finetune_job import finetune_job

# Quick test with minimal steps
result = asyncio.run(finetune_job(
    dataset_url='virattt/financial-qa-10K',
    model_name='OpenPipe/Qwen3-14B-Instruct',
    user_id='test-user',
    num_steps=2,  # Just 2 steps for testing
    rollouts_per_step=4  # Fewer rollouts for speed
))

print('\n✅ Test completed successfully!')
print(f'Model: {result[\"model_name\"]}')
print(f'W&B URL: {result[\"wandb_url\"]}')
"

echo ""
echo "🎉 Docker test completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Check W&B dashboard: https://wandb.ai/finsureai"
echo "   2. Run full training: docker compose up"
echo "   3. Monitor logs: docker compose logs -f"
echo ""
