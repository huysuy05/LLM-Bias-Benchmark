# ...existing code...
#!/bin/bash
set -euo pipefail

# ----------------------------
# Bash script to build and run
# the Docker container for your
# Jupyter notebook project
# ----------------------------

IMAGE_NAME="bias-llms"

echo "=============================="
echo "Building Docker image: $IMAGE_NAME ..."
echo "=============================="
docker build -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
    echo "Docker build failed. Exiting."
    exit 1
fi

echo "=============================="
echo "Running Docker container from image: $IMAGE_NAME ..."
echo "=============================="

# Run the Docker container with proper line continuations (no trailing chars after backslash)
docker run -it \
    -p 8888:8888 \
    -v "$(pwd)":/app \
    "$IMAGE_NAME"
# ...existing code...