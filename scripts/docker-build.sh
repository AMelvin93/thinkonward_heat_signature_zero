#!/bin/bash
# Build the Heat Signature Zero Docker image

set -e

echo "========================================"
echo "Building Heat Signature Zero Container"
echo "========================================"

# Navigate to project root
cd "$(dirname "$0")/.."

# Build with docker-compose (recommended)
if command -v docker-compose &> /dev/null; then
    docker-compose build
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    docker compose build
else
    # Fallback to plain docker build
    docker build -t heat-signature-zero:latest .
fi

echo ""
echo "Build complete!"
echo ""
echo "To run the container:"
echo "  docker-compose up -d"
echo "  docker exec -it heat-signature-dev bash"
echo ""
echo "Or use the run script:"
echo "  ./scripts/docker-run.sh"
