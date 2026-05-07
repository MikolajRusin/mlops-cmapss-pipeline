FROM python:3.10-slim as builder

WORKDIR /opt/dagster/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /opt/dagster/app/wheels -r requirements.txt

# === Final Stage ===

FROM python:3.10-slim

WORKDIR /opt/dagster/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY --from=builder /opt/dagster/app/wheels /wheels
COPY --from=builder /opt/dagster/app/requirements.txt .

COPY src/ ./src/

RUN pip install --no-cache-dir /wheels/*

EXPOSE 4000

CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "src.cmapss_pipeline.definitions"]