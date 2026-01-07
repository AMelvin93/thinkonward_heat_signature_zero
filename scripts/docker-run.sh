#!/bin/bash
# Run experiments in the Heat Signature Zero Docker container

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "heat-signature-dev"; then
    echo "Starting container..."
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d heat-signature
    elif docker compose version &> /dev/null 2>&1; then
        docker compose up -d heat-signature
    else
        echo "Error: docker-compose not found"
        exit 1
    fi
    sleep 2
fi

# If no arguments, open interactive shell
if [ $# -eq 0 ]; then
    echo "========================================"
    echo "Heat Signature Zero - Interactive Shell"
    echo "========================================"
    echo ""
    echo "Quick commands:"
    echo "  uv run python experiments/timestep_subsampling/run.py --workers 7 --shuffle"
    echo "  uv run mlflow ui --host 0.0.0.0"
    echo "  claude  # Start Claude Code"
    echo ""
    docker exec -it heat-signature-dev bash
else
    # Run the provided command
    docker exec -it heat-signature-dev "$@"
fi
