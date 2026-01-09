# Heat Signature Zero - Development Container
# Designed to match G4dn.2xlarge environment (8 vCPUs, 32GB RAM, Linux)
# With Claude Code + Ralph for experiment iteration

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    curl \
    wget \
    nodejs \
    npm \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Install Ralph orchestrator (use uv for psutil)
RUN git clone https://github.com/mikeyobrien/ralph-orchestrator.git /opt/ralph \
    && uv pip install --system psutil

# Set up workspace
WORKDIR /workspace

# Copy project files (use .dockerignore to exclude unnecessary files)
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/
COPY docs/ ./docs/
COPY CLAUDE.md ./

# Create necessary directories
RUN mkdir -p results mlruns notebooks

# Install Python dependencies with uv
RUN uv sync

# Environment variables for optimal performance
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORM_NAME=cuda
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# Default to 7 workers (matching G4dn.2xlarge with 8 vCPUs - 1 for system)
ENV N_WORKERS=7

# Expose MLflow UI port
EXPOSE 5000

# Default command
CMD ["/bin/bash"]
