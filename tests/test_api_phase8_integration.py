import os
import time
import uuid

import pytest

from src.cache_service import RedisConfig, RedisService
from src.vector_store import VectorSearchService, VectorStoreConfig


pytestmark = pytest.mark.integration


def _require_container_tests():
    if os.getenv("SMARTSHOP_RUN_CONTAINER_TESTS") != "1":
        pytest.skip(
            "Set SMARTSHOP_RUN_CONTAINER_TESTS=1 to run Docker integration tests."
        )


def test_redis_service_uses_real_redis_container():
    _require_container_tests()
    redis_container = pytest.importorskip("testcontainers.redis")

    with redis_container.RedisContainer("redis:7-alpine") as container:
        service = RedisService(
            RedisConfig(
                host=container.get_container_host_ip(),
                port=int(container.get_exposed_port(6379)),
                key_prefix=f"smartshop:it:{uuid.uuid4().hex[:8]}",
                rate_limit=2,
                rate_window_seconds=60,
            )
        )

        assert service.ping() is True
        service.set_cached_search("headphones", [{"product_id": "P01"}], top_k=1)
        assert service.get_cached_search("headphones", top_k=1) == [
            {"product_id": "P01"}
        ]
        first = service.check_rate_limit("user:integration")
        second = service.check_rate_limit("user:integration")
        third = service.check_rate_limit("user:integration")
        assert first.allowed is True
        assert second.allowed is True
        assert third.allowed is False


def test_qdrant_vector_search_uses_real_qdrant_container():
    _require_container_tests()
    container_module = pytest.importorskip("testcontainers.core.container")

    collection = f"products_it_{uuid.uuid4().hex[:8]}"
    with container_module.DockerContainer("qdrant/qdrant:v1.10.1").with_exposed_ports(
        6333
    ) as container:
        config = VectorStoreConfig(
            collection_name=collection,
            embedding_backend="hashing",
            vector_size=32,
            qdrant_host=container.get_container_host_ip(),
            qdrant_port=int(container.get_exposed_port(6333)),
        )
        service = VectorSearchService(config=config)

        deadline = time.time() + 30
        while True:
            try:
                service.init_collection()
                break
            except Exception:  # noqa: BLE001
                if time.time() >= deadline:
                    raise
                time.sleep(1)

        indexed = service.index_products(
            [
                {
                    "product_id": "P01",
                    "title": "Wireless Headphones",
                    "description": "Noise cancelling audio",
                    "brand": "SoundWave",
                    "category": "Electronics",
                    "price": 150.0,
                }
            ]
        )
        results = service.search_products("noise cancelling headphones", top_k=1)

        assert indexed == 1
        assert results[0]["product_id"] == "P01"
