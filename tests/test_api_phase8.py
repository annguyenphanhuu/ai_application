from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.agent import AgentConfig, ProductSearchTool, SmartShopAgent
from src.cache_service import RateLimitResult, RedisConfig
from src.main import APIConfig, create_access_token, create_app


class FakeCacheService:
    def __init__(self, cached_results=None, rate_limit_allowed=True):
        self.config = RedisConfig(rate_limit=5, rate_window_seconds=60)
        self.cached_results = cached_results
        self.rate_limit_allowed = rate_limit_allowed
        self.cache_gets = []
        self.cache_sets = []
        self.session_appends = []
        self.session_messages = []
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

    def check_rate_limit(self, identity, limit=None, window_seconds=None):
        self.rate_checks.append(identity)
        return RateLimitResult(
            allowed=self.rate_limit_allowed,
            limit=5,
            remaining=4 if self.rate_limit_allowed else 0,
            reset_after_seconds=60,
            key=f"rate:{identity}",
        )


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


def auth_headers(secret="test-secret", subject="user-1"):
    token = create_access_token(subject=subject, secret=secret)
    return {"Authorization": f"Bearer {token}"}


def build_client(cache=None, search=None, agent=None, upload_dir=None):
    config = APIConfig(
        jwt_secret="test-secret",
        upload_dir=str(upload_dir or "data/uploads/test"),
    )
    app = create_app(
        cache_service=cache or FakeCacheService(),
        search_service=search or FakeSearchService(),
        agent=agent,
        config=config,
    )
    return TestClient(app)


def test_health_is_public():
    client = build_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_requires_bearer_token():
    client = build_client()

    response = client.get("/search", params={"query": "headphones"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Bearer token is required."


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


def test_upload_accepts_csv_and_saves_file(tmp_path):
    cache = FakeCacheService()
    client = build_client(cache=cache, upload_dir=tmp_path)

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


def test_upload_rejects_non_csv_file(tmp_path):
    client = build_client(upload_dir=tmp_path)

    response = client.post(
        "/upload",
        files={"file": ("catalog.json", b"{}")},
        headers=auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV catalog uploads are supported."
