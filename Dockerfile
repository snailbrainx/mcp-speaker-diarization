FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install cuDNN 9 via pip for faster-whisper compatibility
RUN pip install nvidia-cudnn-cu12==9.* nvidia-cublas-cu12

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/

# Create directories for data persistence
RUN mkdir -p /app/data /app/volumes

# Expose port
# 8000 for FastAPI + Gradio
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_PATH=/app/data
ENV VOLUMES_PATH=/app/volumes

# Set LD_LIBRARY_PATH for cuDNN 9 (required by faster-whisper/ctranslate2)
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

# Run the application
CMD ["python", "-m", "app.main"]
