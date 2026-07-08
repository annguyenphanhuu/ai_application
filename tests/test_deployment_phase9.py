from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dockerfile_uses_multistage_build_and_healthcheck():
    dockerfile = read("Dockerfile")

    assert "FROM python:3.11-slim-bookworm AS builder" in dockerfile
    assert "FROM python:3.11-slim-bookworm AS runtime" in dockerfile
    assert "COPY requirements-api.txt ." in dockerfile
    assert "pip install -r requirements-api.txt" in dockerfile
    assert "COPY --from=builder /opt/venv /opt/venv" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert '"/health"' in dockerfile or "/health" in dockerfile
    assert "USER smartshop" in dockerfile


def test_docker_compose_wires_api_dependencies():
    compose = read("docker-compose.yml")

    assert "smartshop-api:latest" in compose
    assert "REDIS_HOST: redis" in compose
    assert "QDRANT_HOST: qdrant" in compose
    assert "SMARTSHOP_EMBEDDING_BACKEND: hashing" in compose
    assert "KAFKA_BOOTSTRAP_SERVERS: kafka:9092" in compose
    assert "redis:7-alpine" in compose
    assert "qdrant/qdrant" in compose
    assert "apache/kafka:3.7.2" in compose


def test_kubernetes_manifests_define_api_service_and_hpa():
    deployment = read("k8s/api-deployment.yaml")
    service = read("k8s/api-service.yaml")
    hpa = read("k8s/api-hpa.yaml")
    kustomization = read("k8s/kustomization.yaml")

    assert "kind: Deployment" in deployment
    assert "name: smartshop-api" in deployment
    assert "readinessProbe:" in deployment
    assert "livenessProbe:" in deployment
    assert "name: REDIS_HOST" in deployment
    assert "name: QDRANT_HOST" in deployment
    assert "kind: Service" in service
    assert "type: LoadBalancer" in service
    assert "kind: HorizontalPodAutoscaler" in hpa
    assert "maxReplicas: 6" in hpa
    assert "kafka.yaml" in kustomization
