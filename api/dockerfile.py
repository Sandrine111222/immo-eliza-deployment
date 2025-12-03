# Dockerfile
FROM python:3.11-slim

# Install system deps for scikit-learn/pandas if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app

# Expose port (Render will provide $PORT env variable)
EXPOSE 8000

# Use PORT environment variable if present (Render sets $PORT)
ENV PORT=8000

# Start uvicorn; note: use host 0.0.0.0 so it's reachable
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
