# SmartShop AI Platform

Production-oriented learning project for a modern AI application pipeline:
data ingestion, Spark ETL, MLflow training, Qdrant semantic search, Redis
cache/rate limit/session state, Kafka clickstream events, FastAPI, Docker,
Kubernetes, Prometheus/Grafana, and Langfuse tracing.

The long roadmap lives in [ai_engineer_roadmap_project.md](ai_engineer_roadmap_project.md).
This README is the operational runbook: use it when you want to run, verify,
or explain the project end to end.

## Current Status

- Unit and local integration-style tests: `python -m pytest`
- API entrypoint: `src.main:app`
- Docker Compose stack: API, Redis, Qdrant, Kafka, Prometheus, Grafana
- Default local API image uses `SMARTSHOP_EMBEDDING_BACKEND=hashing` so it can
  boot quickly without PyTorch or sentence-transformers.
- Real semantic search quality requires indexing and serving with the same
  `sentence-transformers` backend.

## Architecture Map

| Concern | Tool | Main files |
| --- | --- | --- |
| Raw data ingest | Hugging Face / requests | `jobs/amazon_reviews_2023.py` |
| Batch ETL | Spark / Delta-capable pipeline | `jobs/spark_etl.py`, `dags/etl_scheduler.py` |
| Training and registry | scikit-learn / MLflow | `src/train.py`, `src/model_registry.py` |
| Vector search | Qdrant | `src/vector_store.py` |
| Cache, rate limit, state | Redis | `src/cache_service.py` |
| Clickstream | Kafka | `src/streaming.py` |
| Agent workflow | LangChain/LangGraph-style tool flow | `src/agent.py` |
| API | FastAPI | `src/main.py` |
| Observability | Prometheus, Grafana, Langfuse | `src/monitoring.py`, `monitoring/` |
| Deployment | Docker Compose, Kubernetes | `Dockerfile`, `docker-compose.yml`, `k8s/` |

## Local Development

Use a dedicated environment. Conda is the simplest option on Windows:

```powershell
cd D:\ANNGUYEN\Project\AI_Application
conda env create -f environment.yml
conda activate smartshop-ai
python -m pytest
```

If the environment already exists:

```powershell
conda activate smartshop-ai
conda env update -f environment.yml --prune
python -m pytest
```

Spark local mode on Windows can need `HADOOP_HOME` and `winutils.exe` for
Parquet/Delta writes. If that gets noisy, run ETL from WSL as documented in the
roadmap, or use Docker/WSL for the Spark step.

## Production Slice: End-to-End Flow

The production slice is the smallest useful loop that proves the tools work
together:

1. Materialize raw product and review data.
2. Run Spark ETL into `data/processed/amazon_reviews_2023_flow_smoke`.
3. Train and register a baseline model in MLflow.
4. Promote the candidate model to the `champion` alias.
5. Start Docker Compose services.
6. Index products into Qdrant.
7. Smoke test API health, metrics, and optional authenticated endpoints.

You can run the scripted version:

```powershell
.\scripts\e2e_local.ps1
```

For a quicker verification that avoids data download/training:

```powershell
.\scripts\smoke_test.ps1
```

## Manual Commands

### 1. Download a Small Dataset

```powershell
python -m jobs.amazon_reviews_2023 `
  --categories All_Beauty `
  --max-products 5000 `
  --max-reviews 20000 `
  --output-dir data/raw/amazon_reviews_2023
```

### 2. Run Spark ETL

```powershell
python -m jobs.spark_etl `
  --input-products data/raw/amazon_reviews_2023/combined/meta.jsonl `
  --input-reviews data/raw/amazon_reviews_2023/combined/reviews.jsonl `
  --output-path data/processed/amazon_reviews_2023_flow_smoke `
  --output-format parquet `
  --master "local[*]"
```

### 3. Train and Register the Model

```powershell
python -m src.train `
  --input-path data/processed/amazon_reviews_2023_flow_smoke `
  --tracking-uri sqlite:///mlflow.db `
  --experiment-name SmartShop_Rating_Classification `
  --model-name SmartShopRatingClassifier `
  --register-model `
  --registry-alias candidate
```

Promote the selected candidate:

```powershell
python -m src.model_registry promote `
  --tracking-uri sqlite:///mlflow.db `
  --model-name SmartShopRatingClassifier `
  --source-alias candidate `
  --alias champion
```

### 4. Start the Runtime Stack

```powershell
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### 5. Index Qdrant

The default Docker path uses the lightweight hashing encoder:

```powershell
docker compose --profile indexer run --rm vector-indexer
docker compose exec -T redis redis-cli FLUSHDB
```

For better semantic quality, make both API and indexer use
`sentence-transformers` and rebuild the full runtime:

```powershell
$env:SMARTSHOP_API_BUILD_TARGET = "full-runtime"
$env:SMARTSHOP_EMBEDDING_BACKEND = "sentence-transformers"
$env:SMARTSHOP_INDEX_EMBEDDING_BACKEND = "sentence-transformers"
docker compose up --build -d api qdrant redis
docker compose --profile indexer run --rm vector-indexer
docker compose exec -T redis redis-cli FLUSHDB
```

### 6. Call the API

Create a local development token:

```powershell
$TOKEN = docker compose exec -T api python -c "from src.main import create_access_token; print(create_access_token('admin-user'))"
```

Call search:

```powershell
curl -H "Authorization: Bearer $TOKEN" "http://localhost:8000/search?query=noise%20cancelling%20headphones&top_k=5"
```

Call chat stream:

```powershell
curl -N -H "Authorization: Bearer $TOKEN" "http://localhost:8000/chat/stream?message=show%20me%20wireless%20headphones&session_id=demo"
```

Record a clickstream event:

```powershell
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" `
  -d '{"user_id":"demo-user","product_id":"P01","session_id":"demo"}' `
  http://localhost:8000/events/click
```

## Observability

When the Docker Compose stack is running:

- FastAPI metrics: <http://localhost:8000/metrics>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> with `admin/admin` by default

To enable Langfuse, put real keys in `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

If keys are missing, Langfuse tracing disables itself and the API keeps running.

## Security and Production Notes

- Do not use `dev-smartshop-secret` outside local development.
- In production set `SMARTSHOP_ENV=prod` and configure JWT issuer validation via
  `SMARTSHOP_JWKS_URL` or `SMARTSHOP_JWT_PUBLIC_KEY`.
- Keep CORS explicit with `SMARTSHOP_CORS_ORIGINS`.
- Store secrets in a real secret manager or Kubernetes `Secret`, not in Git.
- Use persistent storage for Qdrant, Kafka, Redis, Prometheus, and Grafana.
- Treat the current Kubernetes manifests as local learning manifests. Production
  should add Ingress/TLS, resource tuning, network policy, pod disruption
  budgets, and managed Kafka or a Kafka operator.
- The API can trigger ETL, but a serious deployment should run ETL/training in a
  worker image or orchestrated job rather than inside the API container.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `/health/ready` returns 503 | Redis is unavailable or not healthy. Run `docker compose ps redis`. |
| `/search` returns empty results | Qdrant collection is empty. Run the vector indexer and flush Redis cache. |
| Search quality is weak | API/indexer are probably using hashing. Switch both to `sentence-transformers` and re-index. |
| Spark fails on Windows writing Parquet | Configure `HADOOP_HOME`/`winutils.exe`, or run ETL from WSL. |
| Grafana shows no data | Generate API traffic after `docker compose up`; Prometheus scrapes every 10 seconds. |
| Kafka publish is unavailable | The API accepts click events even when Kafka is down. Check `docker compose logs kafka`. |

## Useful Verification Commands

```powershell
python -m pytest
docker compose config
docker build --target runtime -t smartshop-api:ci .
python -m src.monitoring status
```

