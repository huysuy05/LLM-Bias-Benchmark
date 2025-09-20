
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# Ensure pip and torch installed
RUN pip install --upgrade pip && \
    pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install python deps (ensure jupyterlab & papermill installed)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir jupyterlab papermill notebook

WORKDIR /app
COPY . /app

EXPOSE 8888

# Use python -m jupyter to avoid missing entrypoint issues
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
