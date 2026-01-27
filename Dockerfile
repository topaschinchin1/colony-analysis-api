FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY colony_analysis_api.py .

EXPOSE 5000

# Single worker, increased timeout, optimized for 512MB RAM
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "180", "--workers", "1", "--worker-class", "sync", "--max-requests", "50", "--max-requests-jitter", "10", "colony_analysis_api:app"]
