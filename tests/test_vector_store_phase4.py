from types import SimpleNamespace

import pandas as pd
import pytest

from src.vector_store import (
    HashingTextEncoder,
    ProductSearchFilters,
    VectorSearchService,
    VectorStoreConfig,
    build_product_text,
    infer_input_format,
    prepare_products_for_indexing,
)


class FakeEncoder:
    def __init__(self):
        self.encoded_texts = []

    def encode(self, texts):
        self.encoded_texts.extend(texts)
        return [[float(index), 0.5, 1.0] for index, _text in enumerate(texts)]


class FakeBackend:
    def __init__(self):
        self.initialized = []
        self.upserts = []
        self.searches = []

    def init_collection(self, collection_name, vector_size):
        self.initialized.append((collection_name, vector_size))

    def upsert_products(self, collection_name, vectors, payloads):
        self.upserts.append((collection_name, list(vectors), payloads))

    def search(self, collection_name, query_vector, limit, query_filter=None):
        self.searches.append(
            SimpleNamespace(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
            )
        )
        return [
            {
                "score": 0.91,
                "product_id": "P01",
                "title": "Wireless Headphones",
            }
        ]


def sample_products():
    return pd.DataFrame(
        {
            "product_id": ["P01", "P02"],
            "title": ["Wireless Headphones", "Office Chair"],
            "description": [
                "Noise cancelling audio with long battery life",
                "Ergonomic support for home office",
            ],
            "brand": ["SoundWave", "ErgoFlex"],
            "category": ["Electronics", "Furniture"],
            "price": [150.0, 200.0],
            "price_tier": ["Mid-range", "Premium"],
            "avg_rating": [4.5, 4.0],
            "review_count": [25, 10],
        }
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("products.csv", "csv"),
        ("products.jsonl", "json"),
        ("products.json", "json"),
        ("data/processed/amazon_reviews_2023_flow_smoke", "parquet"),
    ],
)
def test_infer_input_format(path, expected):
    assert infer_input_format(path) == expected


def test_build_product_text_joins_searchable_fields():
    text = build_product_text(
        {
            "title": " Wireless Headphones ",
            "description": "Noise cancelling",
            "brand": "SoundWave",
            "category": "Electronics",
        }
    )

    assert text == "Wireless Headphones Noise cancelling SoundWave Electronics"


def test_prepare_products_for_indexing_builds_payloads_and_texts():
    texts, payloads = prepare_products_for_indexing(sample_products())

    assert len(texts) == 2
    assert "Noise cancelling" in texts[0]
    assert payloads[0]["product_id"] == "P01"
    assert payloads[0]["price"] == 150.0
    assert payloads[0]["category"] == "Electronics"


def test_prepare_products_for_indexing_requires_core_columns():
    with pytest.raises(ValueError, match="missing columns"):
        prepare_products_for_indexing(pd.DataFrame({"product_id": ["P01"]}))


def test_vector_search_service_initializes_and_indexes_dataframe():
    encoder = FakeEncoder()
    backend = FakeBackend()
    service = VectorSearchService(
        config=VectorStoreConfig(collection_name="test-products", vector_size=3),
        encoder=encoder,
        backend=backend,
    )

    service.init_collection()
    indexed_count = service.index_dataframe(sample_products())

    assert indexed_count == 2
    assert backend.initialized == [("test-products", 3)]
    collection_name, vectors, payloads = backend.upserts[0]
    assert collection_name == "test-products"
    assert vectors == [[0.0, 0.5, 1.0], [1.0, 0.5, 1.0]]
    assert payloads[1]["product_id"] == "P02"
    assert "Wireless Headphones" in encoder.encoded_texts[0]


def test_vector_search_service_searches_with_encoded_query():
    encoder = FakeEncoder()
    backend = FakeBackend()
    service = VectorSearchService(
        config=VectorStoreConfig(collection_name="test-products", vector_size=3),
        encoder=encoder,
        backend=backend,
    )

    results = service.search_products(
        "noise cancelling headphones",
        filters=ProductSearchFilters(),
        top_k=3,
    )

    assert results[0]["product_id"] == "P01"
    assert encoder.encoded_texts == ["noise cancelling headphones"]
    assert backend.searches[0].collection_name == "test-products"
    assert backend.searches[0].query_vector == [0.0, 0.5, 1.0]
    assert backend.searches[0].limit == 3
    assert backend.searches[0].query_filter is None


def test_vector_search_service_rejects_empty_query():
    service = VectorSearchService(
        config=VectorStoreConfig(collection_name="test-products", vector_size=3),
        encoder=FakeEncoder(),
        backend=FakeBackend(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        service.search_products("   ")


def test_hashing_text_encoder_is_deterministic_and_normalized():
    encoder = HashingTextEncoder(vector_size=16)

    first = encoder.encode(["Wireless headphones"])[0]
    second = encoder.encode(["Wireless headphones"])[0]

    assert first == second
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_vector_search_service_can_use_hashing_backend():
    backend = FakeBackend()
    service = VectorSearchService(
        config=VectorStoreConfig(
            collection_name="test-products",
            embedding_backend="hashing",
            vector_size=16,
        ),
        backend=backend,
    )

    results = service.search_products("headphones", top_k=2)

    assert results[0]["product_id"] == "P01"
    assert len(backend.searches[0].query_vector) == 16
    assert backend.searches[0].limit == 2


def test_vector_store_config_can_be_loaded_from_env(monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "qdrant")
    monkeypatch.setenv("QDRANT_PORT", "6334")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "catalog")
    monkeypatch.setenv("SMARTSHOP_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("SMARTSHOP_VECTOR_SIZE", "768")

    config = VectorStoreConfig.from_env()

    assert config.qdrant_host == "qdrant"
    assert config.qdrant_port == 6334
    assert config.collection_name == "catalog"
    assert config.embedding_backend == "hashing"
    assert config.vector_size == 768
