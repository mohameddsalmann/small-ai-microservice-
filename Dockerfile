# Beard classification microservice - RunPod Serverless
# Build: docker build -t beard-classifier .
# Run locally (FastAPI): docker run -p 8000:8000 beard-classifier api
# RunPod runs: CMD ["python", "-u", "runpod_handler/handler.py"]
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV headless (no libgl1-mesa-glx - use libgl1 on Bookworm+ or skip for headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and RunPod handler
COPY app/ ./app/
COPY runpod_handler/ ./runpod_handler/
COPY templates/ ./templates/

# RunPod serverless: run handler
CMD ["python", "-u", "runpod_handler/handler.py"]
