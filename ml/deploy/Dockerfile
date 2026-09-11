FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000 \
    PYTHONPATH=/app

WORKDIR /app

# Install system libraries needed by OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install deployment requirements
COPY ml/deploy/requirements-deploy.txt /app/requirements-deploy.txt
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy ML code and models
COPY ml /app/ml

EXPOSE 10000

# Run FastAPI with uvicorn, dynamically binding to $PORT (required by Render/Cloud Run/Railway)
CMD ["sh", "-c", "uvicorn ml.deploy.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
