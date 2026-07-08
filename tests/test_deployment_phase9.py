from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dockerfile_uses_multistage_build_and_healthcheck():
    dockerfile = read("Dockerfile")

    assert "FROM python:3.11-slim-bookworm AS builder" in dockerfile
    assert "FROM python:3.11-slim-bookworm AS full-builder" in dockerfile
    assert "FROM python:3.11-slim-bookworm AS runtime" in dockerfile
    assert "FROM runtime AS full-runtime" in dockerfile
    assert "FROM full-runtime AS vector-indexer" in dockerfile
    assert "COPY requirements-api.txt ." in dockerfile
    assert "pip install -r requirements-api.txt" in dockerfile
    assert "COPY requirements.txt ." in dockerfile
    assert "COPY --from=builder /opt/venv /opt/venv" in dockerfile
    assert "COPY --from=full-builder /opt/venv /opt/venv" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/health/ready" in dockerfile
    assert "USER smartshop" in dockerfile


def test_docker_compose_wires_api_dependencies():
    compose = read("docker-compose.yml")

    assert "smartshop-api:latest" in compose
    assert "REDIS_HOST: redis" in compose
    assert "SMARTSHOP_ENV:" in compose
    assert "QDRANT_HOST: qdrant" in compose
    assert "SMARTSHOP_EMBEDDING_BACKEND: ${SMARTSHOP_EMBEDDING_BACKEND:-hashing}" in compose
    assert "KAFKA_BOOTSTRAP_SERVERS: kafka:9092" in compose
    assert "redis:7-alpine" in compose
    assert "qdrant/qdrant" in compose
    assert "apache/kafka:3.7.2" in compose


def test_api_runtime_uses_lightweight_hashing_encoder():
    requirements_api = read("requirements-api.txt")
    compose = read("docker-compose.yml")
    deployment = read("k8s/api-deployment.yaml")
    roadmap = read("ai_engineer_roadmap_project.md")

    assert "sentence-transformers" not in [
        line.strip()
        for line in requirements_api.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "SMARTSHOP_API_BUILD_TARGET:-runtime" in compose
    assert "SMARTSHOP_EMBEDDING_BACKEND:-hashing" in compose
    assert "name: SMARTSHOP_EMBEDDING_BACKEND" in deployment
    assert "value: hashing" in deployment
    assert "HashingTextEncoder" in roadmap
    assert "khong bang embedding that" in roadmap


def test_compose_defines_optional_vector_indexer():
    compose = read("docker-compose.yml")
    roadmap = read("ai_engineer_roadmap_project.md")

    assert "vector-indexer:" in compose
    assert 'profiles: ["indexer"]' in compose
    assert "target: vector-indexer" in compose
    assert "python" in compose
    assert "src.vector_store" in compose
    assert "${SMARTSHOP_VECTOR_INPUT_PATH:-/app/data/processed/products_processed}" in compose
    assert "./data/processed:/app/data/processed:ro" in compose
    assert "docker compose --profile indexer run --rm vector-indexer" in roadmap
    assert "redis-cli FLUSHDB" in roadmap


def test_phase10_documents_monitoring_runtime_requirements():
    env_example = read(".env.example")
    prometheus = read("monitoring/prometheus.yml")
    roadmap = read("ai_engineer_roadmap_project.md")

    assert "LANGFUSE_PUBLIC_KEY=pk-lf-..." in env_example
    assert "LANGFUSE_SECRET_KEY=sk-lf-..." in env_example
    assert "metrics_path: /metrics" in prometheus
    assert 'targets: ["api:8000"]' in prometheus
    assert "No data" in roadmap
    assert "docker compose up --build -d" in roadmap


def test_kubernetes_manifests_define_api_service_and_hpa():
    deployment = read("k8s/api-deployment.yaml")
    service = read("k8s/api-service.yaml")
    hpa = read("k8s/api-hpa.yaml")
    kustomization = read("k8s/kustomization.yaml")

    assert "kind: Deployment" in deployment
    assert "name: smartshop-api" in deployment
    assert "readinessProbe:" in deployment
    assert "path: /health/ready" in deployment
    assert "livenessProbe:" in deployment
    assert "name: REDIS_HOST" in deployment
    assert "name: SMARTSHOP_ENV" in deployment
    assert "name: QDRANT_HOST" in deployment
    assert "kind: Service" in service
    assert "type: LoadBalancer" in service
    assert "kind: HorizontalPodAutoscaler" in hpa
    assert "maxReplicas: 6" in hpa
    assert "kafka.yaml" in kustomization
