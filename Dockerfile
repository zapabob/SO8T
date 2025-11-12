# SO8T Safe Agent Dockerfile
# Multi-stage build for production deployment

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libxft-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libnetcdf-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Stage 2: Development image
FROM base as development

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data models checkpoints dist logs eval_results

# Set permissions
RUN chmod +x scripts/*.py

# Expose ports
EXPOSE 8000 8080

# Default command for development
CMD ["python", "-m", "inference.agent_runtime", "--config", "configs/inference_config.yaml"]

# Stage 3: Production image
FROM base as production

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install only production dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1+cu121 \
    torchvision==0.15.2+cu121 \
    torchaudio==2.0.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    transformers>=4.35.0 \
    tokenizers>=0.14.0 \
    accelerate>=0.24.0 \
    peft>=0.6.0 \
    bitsandbytes>=0.41.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    tqdm>=4.65.0 \
    pyyaml>=6.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    pydantic>=2.0.0 \
    requests>=2.31.0 \
    psutil>=5.9.0

# Copy source code
COPY models/ ./models/
COPY training/ ./training/
COPY inference/ ./inference/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY docs/ ./docs/

# Create necessary directories
RUN mkdir -p data models checkpoints dist logs eval_results

# Set permissions
RUN chmod +x scripts/*.py

# Create non-root user
RUN useradd -m -u 1000 so8t && \
    chown -R so8t:so8t /app

# Switch to non-root user
USER so8t

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["python", "-m", "inference.agent_runtime", "--config", "configs/inference_config.yaml"]

# Stage 4: Training image
FROM base as training

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install all dependencies including development tools
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data models checkpoints dist logs eval_results

# Set permissions
RUN chmod +x scripts/*.py

# Expose ports
EXPOSE 8000 8080 6006

# Default command for training
CMD ["python", "-m", "training.train_qlora", "--config", "configs/training_config.yaml"]

# Stage 5: Evaluation image
FROM base as evaluation

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install evaluation dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1+cu121 \
    torchvision==0.15.2+cu121 \
    torchaudio==2.0.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    transformers>=4.35.0 \
    tokenizers>=0.14.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    tqdm>=4.65.0 \
    pyyaml>=6.0 \
    psutil>=5.9.0

# Copy source code
COPY models/ ./models/
COPY eval/ ./eval/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p data models checkpoints dist logs eval_results

# Set permissions
RUN chmod +x scripts/*.py

# Default command for evaluation
CMD ["python", "-m", "eval.eval_safety", "--config", "configs/evaluation_config.yaml"]
