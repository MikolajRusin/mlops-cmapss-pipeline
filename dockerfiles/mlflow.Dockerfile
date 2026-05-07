FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mlflow==3.11.1 boto3 psycopg2-binary
EXPOSE 5000