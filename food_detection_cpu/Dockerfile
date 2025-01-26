FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV HOSTNAME="0.0.0.0"

# Install PyTorch CPU version first
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir ultralytics==8.0.227 gradio==3.50.2 Pillow==10.2.0 opencv-python-headless==4.9.0.80

# Copy application files
COPY app.py .
COPY runs/detect/train9/weights/best.pt runs/detect/train9/weights/best.pt
COPY examples examples/

# Create docs/images directory
RUN mkdir -p docs/images

# Command to run the application
CMD ["python", "app.py"] 