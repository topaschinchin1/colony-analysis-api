FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "180", "--workers", "1", "--worker-class", "sync", "--max-requests", "50", "--max-requests-jitter", "10", "app:app"]
