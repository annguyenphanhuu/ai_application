FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt


FROM python:3.11-slim-bookworm AS runtime

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SMARTSHOP_UPLOAD_DIR=/app/data/uploads/catalog

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system smartshop \
    && useradd --system --gid smartshop --home-dir /app smartshop

COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN mkdir -p /app/data/uploads/catalog \
    && chown -R smartshop:smartshop /app

USER smartshop

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
