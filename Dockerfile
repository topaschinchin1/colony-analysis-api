FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY colony_analysis_api.py .

EXPOSE 5000

# Increased timeout and single worker for memory efficiency
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "180", "--workers", "1", "--worker-class", "sync", "colony_analysis_api:app"]
