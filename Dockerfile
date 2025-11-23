# Use a lightweight Python base image instead of the heavy PyTorch one
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# ffmpeg: for audio processing
# git: for installing python packages from git
# build-essential: for compiling some python extensions
# portaudio19-dev: for sounddevice
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# 1. Install PyTorch with CUDA 12.4 support FIRST
# This avoids downloading the huge default CPU wheels or reinstalling later
# We use the official PyTorch wheel index for CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Install CUDA libraries for faster-whisper / ctranslate2
# These are needed because we aren't using the nvidia/cuda base image
RUN pip install nvidia-cudnn-cu12==9.* nvidia-cublas-cu12

# 3. Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Create directories for data persistence
RUN mkdir -p /app/data /app/volumes /app/backups

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_PATH=/app/data
ENV VOLUMES_PATH=/app/volumes

# Set LD_LIBRARY_PATH to include the pip-installed NVIDIA libraries
# This is crucial for ctranslate2/faster-whisper to find cuDNN/cuBLAS
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

# Run the application
CMD ["python", "-m", "app.main"]