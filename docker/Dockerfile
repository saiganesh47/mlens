FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /root/.local /root/.local

COPY mlens/   ./mlens/
COPY api/     ./api/
COPY setup.py .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
