# OpenPipe Training Environment
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy OpenPipe training scripts
COPY . /workspace

# Note: reward.py will be mounted from ../modal/reward.py via docker-compose volumes

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python3", "finetune_job.py"]
