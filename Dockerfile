# Serves the read-only forecasting/replenishment API (api/main.py).
#
# This does NOT need the full project -- just the API code, the
# retail_forecasting package it imports from (src/), and the small
# pre-computed data files it reads (dashboard/data/). Training scripts,
# the Streamlit dashboard, notebooks, and raw data are intentionally left
# out of the image.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across rebuilds that
# only change application code.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code and the small data files it serves.
COPY api/ ./api/
COPY src/ ./src/
COPY dashboard/data/ ./dashboard/data/

# AWS Elastic Beanstalk's single-container Docker platform expects the
# application to listen on port 8080 by default.
EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
