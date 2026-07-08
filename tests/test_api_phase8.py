import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.agent import AgentConfig, ProductSearchTool, SmartShopAgent
from src.cache_service import HumanApprovalRequest, RateLimitResult, RedisConfig
from src.main import APIConfig, create_access_token, create_app


class FakeCacheService:
    def __init__(
        self,
        cached_results=None,
        rate_limit_allowed=True,
        redis_available=True,
    ):
        self.config = RedisConfig(rate_limit=5, rate_window_seconds=60)
        self.cached_results = cached_results
        self.rate_limit_allowed = rate_limit_allowed
        self.redis_available = redis_available
        self.cache_gets = []
        self.cache_sets = []
        self.session_appends = []
        self.session_messages = []
        self.approvals = {}
        self.pending_approvals = []
        self.rate_checks = []

    def get_cached_search(self, query, filters=None, top_k=5):
        self.cache_gets.append(
            {"query": query, "filters": filters or {}, "top_k": top_k}
        )
        return self.cached_results

    def set_cached_search(
        self,
        query,
        results,
        filters=None,
        top_k=5,
        ttl_seconds=None,
    ):
        self.cache_sets.append(
            {
                "query": query,
                "results": results,
                "filters": filters or {},
                "top_k": top_k,
                "ttl_seconds": ttl_seconds,
            }
        )
        return "testshop:search:cached"

    def append_session_message(
        self,
        session_id,
        role,
        content,
        metadata=None,
        max_messages=50,
        ttl_seconds=None,
    ):
        self.session_appends.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }
        )
        return SimpleNamespace(role=role, content=content)

    def get_session_messages(self, session_id, limit=None):
        return list(self.session_messages)

    def create_human_approval_request(
        self,
        session_id,
        message,
        history=None,
        reason="",
        metadata=None,
        ttl_seconds=None,
    ):
        request = HumanApprovalRequest(
            request_id=f"A{len(self.approvals) + 1:03d}",
            session_id=session_id,
            message=message,
            history=list(history or []),
            reason=reason,
            metadata=metadata or {},
        )
        self.approvals[request.request_id] = request
        self.pending_approvals.append(request.request_id)
        return request

    def get_human_approval_request(self, request_id):
        return self.approvals.get(request_id)

    def list_pending_human_approvals(self, limit=20):
        return [
            self.approvals[request_id]
            for request_id in self.pending_approvals[:limit]
            if self.approvals[request_id].status == "pending"
        ]

    def resolve_human_approval_request(
        self,
        request_id,
        approved,
        reviewer,
        note=None,
        ttl_seconds=None,
    ):
        request = self.approvals.get(request_id)
        if request is None:
            raise KeyError(request_id)
        resolved = HumanApprovalRequest(
            request_id=request.request_id,
            session_id=request.session_id,
            message=request.message,
            history=request.history,
            reason=request.reason,
            status="approved" if approved else "rejected",
            created_at=request.created_at,
            resolved_at="2026-07-08T00:00:00+00:00",
            resolved_by=reviewer,
            note=note,
            metadata=request.metadata,
        )
        self.approvals[request_id] = resolved
        self.pending_approvals = [
            pending for pending in self.pending_approvals if pending != request_id
        ]
        return resolved

    def check_rate_limit(self, identity, limit=None, window_seconds=None):
        self.rate_checks.append(identity)
        return RateLimitResult(
            allowed=self.rate_limit_allowed,
            limit=5,
            remaining=4 if self.rate_limit_allowed else 0,
            reset_after_seconds=60,
            key=f"rate:{identity}",
        )

    def ping(self):
        return self.redis_available


class FakeSearchService:
    def __init__(self):
        self.searches = []
        self.initialized = False

    def init_collection(self):
        self.initialized = True

    def search_products(
        self,
        query,
        filters=None,
        top_k=5,
        category_filter=None,
    ):
        self.searches.append(
            {
                "query": query,
                "filters": filters,
                "top_k": top_k,
                "category_filter": category_filter,
            }
        )
        return [
            {
                "product_id": "P01",
                "title": "Wireless Headphones",
                "score": 0.91,
            }
        ]


class FakeETLJobRunner:
    def __init__(self, status="completed"):
        self.status = status
        self.runs = []

    def run(self, input_path, manifest_path, config):
        self.runs.append(
            {
                "input_path": input_path,
                "manifest_path": manifest_path,
                "output_format": config.etl_output_format,
            }
        )
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        result = {"status": self.status, "output_path": "data/processed/test"}
        manifest.update({"status": f"etl_{self.status}", "etl": result})
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        return result


class FakeKafkaProducer:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.clicks = []

    def log_click(
        self,
        user_id,
        product_id,
        session_id=None,
        query=None,
        metadata=None,
    ):
        if self.should_fail:
            raise OSError("broker unavailable")
        event = {
            "user_id": user_id,
            "product_id": product_id,
            "event_type": "product_click",
            "schema_version": "1",
            "metadata": metadata or {},
        }
        if session_id is not None:
            event["session_id"] = session_id
        if query is not None:
            event["query"] = query
        self.clicks.append(event)
        return event


def auth_headers(secret="test-secret", subject="user-1"):
    token = create_access_token(subject=subject, secret=secret)
    return {"Authorization": f"Bearer {token}"}


def build_client(
    cache=None,
    search=None,
    agent=None,
    upload_dir=None,
    etl_runner=None,
    kafka_producer=None,
):
    config = APIConfig(
        environment="test",
        jwt_secret="test-secret",
        upload_dir=str(upload_dir or "data/uploads/test"),
    )
    app = create_app(
        cache_service=cache or FakeCacheService(),
        search_service=search or FakeSearchService(),
        agent=agent,
        config=config,
        etl_runner=etl_runner or FakeETLJobRunner(),
        kafka_producer=kafka_producer,
    )
    return TestClient(app)


def test_health_is_public():
    client = build_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_checks_redis():
    client = build_client(cache=FakeCacheService(redis_available=True))

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "redis": "ok"}


def test_readiness_fails_when_redis_is_unavailable():
    client = build_client(cache=FakeCacheService(redis_available=False))

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["redis"] == "unavailable"


def test_search_requires_bearer_token():
    client = build_client()

    response = client.get("/search", params={"query": "headphones"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Bearer token is required."


def test_cors_allows_configured_frontend_origin():
    config = APIConfig(
        environment="test",
        jwt_secret="test-secret",
        cors_allowed_origins=("https://frontend.smartshop.test",),
    )
    app = create_app(
        cache_service=FakeCacheService(),
        search_service=FakeSearchService(),
        config=config,
        etl_runner=FakeETLJobRunner(),
    )
    client = TestClient(app)

    response = client.options(
        "/search",
        headers={
            "Origin": "https://frontend.smartshop.test",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "https://frontend.smartshop.test"
    )
    assert "Authorization" in response.headers["access-control-allow-headers"]


def test_prod_auth_rejects_default_dev_secret_without_issuer():
    config = APIConfig(environment="prod")
    app = create_app(
        cache_service=FakeCacheService(),
        search_service=FakeSearchService(),
        config=config,
        etl_runner=FakeETLJobRunner(),
    )
    client = TestClient(app)

    response = client.get(
        "/search",
        params={"query": "headphones"},
        headers=auth_headers(secret="dev-smartshop-secret"),
    )

    assert response.status_code == 401
    assert "JWT issuer/JWKS validation" in response.json()["detail"]


def test_auth_accepts_rs256_token_from_configured_issuer():
    jwt = pytest.importorskip("jwt")
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = (
        private_key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    token = jwt.encode(
        {
            "sub": "issuer-user",
            "iss": "https://issuer.smartshop.test",
            "aud": "smartshop-api",
            "scope": "catalog:read",
            "exp": 4102444800,
        },
        private_key,
        algorithm="RS256",
    )
    config = APIConfig(
        environment="prod",
        jwt_public_key=public_key,
        jwt_issuer="https://issuer.smartshop.test",
        jwt_audience="smartshop-api",
    )
    app = create_app(
        cache_service=FakeCacheService(cached_results=[]),
        search_service=FakeSearchService(),
        config=config,
        etl_runner=FakeETLJobRunner(),
    )
    client = TestClient(app)

    response = client.get(
        "/search",
        params={"query": "headphones"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200


def test_search_returns_cached_results_without_vector_call():
    cache = FakeCacheService(
        cached_results=[{"product_id": "P01", "title": "Cached Headphones"}]
    )
    search = FakeSearchService()
    client = build_client(cache=cache, search=search)

    response = client.get(
        "/search",
        params={"query": "Headphones", "category": "Electronics", "top_k": 3},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "cache"
    assert payload["results"][0]["title"] == "Cached Headphones"
    assert cache.cache_gets == [
        {
            "query": "Headphones",
            "filters": {"category": "Electronics"},
            "top_k": 3,
        }
    ]
    assert search.searches == []


def test_search_cache_miss_calls_vector_search_and_caches_results():
    cache = FakeCacheService(cached_results=None)
    search = FakeSearchService()
    client = build_client(cache=cache, search=search)

    response = client.get(
        "/search",
        params={"query": "office chair", "brand": "ErgoFlex", "top_k": 2},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "vector_store"
    assert payload["results"][0]["product_id"] == "P01"
    assert search.searches[0]["query"] == "office chair"
    assert search.searches[0]["filters"].brand == "ErgoFlex"
    assert search.searches[0]["top_k"] == 2
    assert cache.cache_sets[0]["filters"] == {"brand": "ErgoFlex"}


def test_search_dependency_error_returns_service_unavailable(monkeypatch):
    cache = FakeCacheService(cached_results=None)
    config = APIConfig(jwt_secret="test-secret")
    app = create_app(cache_service=cache, search_service=None, config=config)

    class BrokenVectorSearchService:
        def __init__(self):
            raise RuntimeError("embedding package is missing")

    monkeypatch.setattr("src.main.VectorSearchService", BrokenVectorSearchService)
    client = TestClient(app)

    response = client.get(
        "/search",
        params={"query": "headphones"},
        headers=auth_headers(),
    )

    assert response.status_code == 503
    assert "embedding package is missing" in response.json()["detail"]


def test_rate_limit_blocks_protected_endpoint():
    cache = FakeCacheService(rate_limit_allowed=False)
    client = build_client(cache=cache)

    response = client.get(
        "/search",
        params={"query": "headphones"},
        headers=auth_headers(),
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"


def test_chat_streams_agent_response_and_persists_session():
    cache = FakeCacheService()
    search = FakeSearchService()
    agent = SmartShopAgent(
        config=AgentConfig(top_k=1),
        search_tool=ProductSearchTool(search),
    )
    client = build_client(cache=cache, search=search, agent=agent)

    response = client.post(
        "/chat",
        json={"message": "what is your shipping policy?", "session_id": "S01"},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert "event: start" in response.text
    assert "event: done" in response.text
    assert "3-5 business days" in response.text
    assert [row["role"] for row in cache.session_appends] == ["user", "assistant"]
    assert cache.session_appends[1]["metadata"]["action"] == "answer_policy"


def test_default_chat_policy_does_not_initialize_vector_search():
    cache = FakeCacheService()
    config = APIConfig(jwt_secret="test-secret")
    app = create_app(cache_service=cache, search_service=None, config=config)
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"message": "what is your shipping policy?", "session_id": "S01"},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert "3-5 business days" in response.text
    assert app.state.search_service is None


def test_chat_human_review_creates_approval_request():
    cache = FakeCacheService()
    agent = SmartShopAgent(
        config=AgentConfig(use_llm_routing="never"),
        search_tool=ProductSearchTool(FakeSearchService()),
    )
    client = build_client(cache=cache, agent=agent)

    response = client.post(
        "/chat",
        json={"message": "I want a refund for a damaged order", "session_id": "S01"},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert '"approval_request_id": "A001"' in response.text
    assert cache.pending_approvals == ["A001"]
    assert cache.session_appends[1]["metadata"]["approval_request_id"] == "A001"


def test_approval_queue_can_be_listed_and_resolved():
    cache = FakeCacheService()
    approval = cache.create_human_approval_request(
        "S01",
        "refund damaged order",
        reason="damaged order",
    )
    client = build_client(cache=cache)

    listed = client.get("/approvals/pending", headers=auth_headers())
    resolved = client.post(
        f"/approvals/{approval.request_id}/resolve",
        json={"approved": True, "note": "ok"},
        headers=auth_headers(subject="reviewer-1"),
    )

    assert listed.status_code == 200
    assert listed.json()["requests"][0]["request_id"] == approval.request_id
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "approved"
    assert resolved.json()["resolved_by"] == "reviewer-1"
    assert cache.pending_approvals == []


def test_default_search_initializes_vector_collection(monkeypatch):
    cache = FakeCacheService(cached_results=None)
    search = FakeSearchService()
    config = APIConfig(jwt_secret="test-secret")
    app = create_app(cache_service=cache, search_service=None, config=config)

    monkeypatch.setattr("src.main.VectorSearchService", lambda: search)
    client = TestClient(app)

    response = client.get(
        "/search",
        params={"query": "headphones"},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert search.initialized is True
    assert response.json()["results"][0]["product_id"] == "P01"


def test_click_event_endpoint_publishes_to_kafka_producer():
    kafka = FakeKafkaProducer()
    client = build_client(kafka_producer=kafka)

    response = client.post(
        "/events/click",
        json={
            "user_id": "U01",
            "product_id": "P01",
            "session_id": "S01",
            "query": "headphones",
            "metadata": {"page": "search"},
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "published"
    assert payload["kafka_available"] is True
    assert payload["event"]["product_id"] == "P01"
    assert kafka.clicks[0]["metadata"] == {"page": "search"}


def test_click_event_endpoint_accepts_event_when_kafka_publish_fails():
    kafka = FakeKafkaProducer(should_fail=True)
    client = build_client(kafka_producer=kafka)

    response = client.post(
        "/events/click",
        json={"user_id": "U01", "product_id": "P01"},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted_no_broker"
    assert payload["kafka_available"] is False
    assert payload["event"]["product_id"] == "P01"


def test_upload_accepts_csv_and_saves_file(tmp_path):
    cache = FakeCacheService()
    etl_runner = FakeETLJobRunner()
    client = build_client(cache=cache, upload_dir=tmp_path, etl_runner=etl_runner)

    response = client.post(
        "/upload",
        files={"file": ("catalog.csv", b"product_id,title\nP01,Headphones\n")},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "catalog.csv"
    assert payload["status"] == "queued_for_etl"
    assert Path(payload["saved_path"]).exists()
    assert Path(payload["manifest_path"]).exists()
    assert etl_runner.runs[0]["input_path"] == payload["saved_path"]
    manifest = json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "etl_completed"


def test_upload_rejects_non_csv_file(tmp_path):
    client = build_client(upload_dir=tmp_path)

    response = client.post(
        "/upload",
        files={"file": ("catalog.json", b"{}")},
        headers=auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV catalog uploads are supported."
