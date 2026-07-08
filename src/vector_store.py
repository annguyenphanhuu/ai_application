"""Phase 4 vector indexing and semantic search for SmartShop products."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import pandas as pd


DEFAULT_COLLECTION_NAME = "products"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_SIZE = 384
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_INPUT_PATH = "data/processed/products_processed"
DEFAULT_EMBEDDING_BACKEND = "sentence-transformers"
TEXT_COLUMNS = ("title", "description", "brand", "category")


class EmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts into dense vectors."""


class VectorBackend(Protocol):
    def init_collection(self, collection_name: str, vector_size: int) -> None:
        """Create or verify a vector collection."""

    def upsert_products(
        self,
        collection_name: str,
        vectors: Sequence[Sequence[float]],
        payloads: list[dict],
    ) -> None:
        """Upsert embedded products and metadata."""

    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int,
        query_filter: Any | None = None,
    ) -> list[dict]:
        """Search the vector store and return normalized hit dictionaries."""


@dataclass(frozen=True)
class ProductSearchFilters:
    category: str | None = None
    brand: str | None = None
    min_price: float | None = None
    max_price: float | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            value is not None
            for value in (self.category, self.brand, self.min_price, self.max_price)
        )


@dataclass(frozen=True)
class VectorStoreConfig:
    input_path: str = DEFAULT_INPUT_PATH
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_backend: str = DEFAULT_EMBEDDING_BACKEND
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    vector_size: int = DEFAULT_VECTOR_SIZE
    qdrant_host: str = DEFAULT_QDRANT_HOST
    qdrant_port: int = DEFAULT_QDRANT_PORT

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        return cls(
            input_path=os.getenv("SMARTSHOP_VECTOR_INPUT_PATH", DEFAULT_INPUT_PATH),
            collection_name=os.getenv(
                "QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME
            ),
            embedding_backend=os.getenv(
                "SMARTSHOP_EMBEDDING_BACKEND", DEFAULT_EMBEDDING_BACKEND
            ),
            embedding_model=os.getenv(
                "SMARTSHOP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
            vector_size=int(
                os.getenv("SMARTSHOP_VECTOR_SIZE", str(DEFAULT_VECTOR_SIZE))
            ),
            qdrant_host=os.getenv(
                "QDRANT_HOST", os.getenv("SMARTSHOP_QDRANT_HOST", DEFAULT_QDRANT_HOST)
            ),
            qdrant_port=int(
                os.getenv(
                    "QDRANT_PORT",
                    os.getenv("SMARTSHOP_QDRANT_PORT", str(DEFAULT_QDRANT_PORT)),
                )
            ),
        )


def infer_input_format(path: str) -> str:
    clean_path = path.rstrip("/").lower()
    if clean_path.endswith(".csv"):
        return "csv"
    if clean_path.endswith(".json") or clean_path.endswith(".jsonl"):
        return "json"
    return "parquet"


def load_products(path: str) -> pd.DataFrame:
    data_format = infer_input_format(path)
    if data_format == "csv":
        return pd.read_csv(path)
    if data_format == "json":
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


def build_product_text(product: dict) -> str:
    parts = [product.get(column) for column in TEXT_COLUMNS]
    text = " ".join(
        str(part).strip() for part in parts if pd.notna(part) and str(part).strip()
    )
    return " ".join(text.split())


def normalize_product_payload(product: dict) -> dict:
    payload = {
        "product_id": str(product.get("product_id", "")),
        "title": product.get("title", ""),
        "description": product.get("description", ""),
        "brand": product.get("brand", ""),
        "category": product.get("category", ""),
        "price": product.get("price"),
        "price_tier": product.get("price_tier", ""),
        "avg_rating": product.get("avg_rating"),
        "review_count": product.get("review_count"),
    }
    return {key: (None if pd.isna(value) else value) for key, value in payload.items()}


def prepare_products_for_indexing(
    products: pd.DataFrame,
) -> tuple[list[str], list[dict]]:
    required_columns = {"product_id", "title", "description"}
    missing_columns = sorted(required_columns - set(products.columns))
    if missing_columns:
        raise ValueError(f"Product data is missing columns: {missing_columns}")

    clean_products = products.dropna(subset=["product_id"]).copy()
    if clean_products.empty:
        raise ValueError("Product data has no rows with product_id.")

    payloads = [
        normalize_product_payload(row) for row in clean_products.to_dict("records")
    ]
    texts = [build_product_text(payload) for payload in payloads]
    return texts, payloads


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for Phase 4 embeddings. "
                "Install it with `pip install -r requirements.txt` or update the Conda env."
            ) from exc

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(list(texts), normalize_embeddings=True)
        return embeddings.tolist()


class HashingTextEncoder:
    """Small dependency-free encoder for local Docker demos.

    This is not a replacement for sentence-transformers quality. It gives the
    API image a deterministic vector representation so Qdrant search can run
    without pulling a large PyTorch stack.
    """

    def __init__(self, vector_size: int = DEFAULT_VECTOR_SIZE):
        if vector_size < 1:
            raise ValueError("vector_size must be greater than zero.")
        self.vector_size = vector_size

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_size
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.vector_size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def build_encoder(config: VectorStoreConfig) -> EmbeddingModel:
    backend = config.embedding_backend.strip().lower()
    if backend in {"hashing", "local-hashing", "hash"}:
        return HashingTextEncoder(vector_size=config.vector_size)
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEncoder(config.embedding_model)
    raise ValueError(
        "Unsupported embedding backend. Use 'sentence-transformers' or 'hashing'."
    )


class QdrantVectorBackend:
    def __init__(
        self, host: str = DEFAULT_QDRANT_HOST, port: int = DEFAULT_QDRANT_PORT
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for Phase 4 vector search. "
                "Install it with `pip install -r requirements.txt` or update the Conda env."
            ) from exc

        self.client = QdrantClient(host=host, port=port)

    def init_collection(self, collection_name: str, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(collection_name):
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_products(
        self,
        collection_name: str,
        vectors: Sequence[Sequence[float]],
        payloads: list[dict],
    ) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, payload["product_id"])),
                vector=list(vector),
                payload=payload,
            )
            for vector, payload in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int,
        query_filter: Any | None = None,
    ) -> list[dict]:
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=list(query_vector),
                query_filter=query_filter,
                limit=limit,
            )
            hits = response.points
        else:
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=list(query_vector),
                query_filter=query_filter,
                limit=limit,
            )
        return [
            {
                "score": float(getattr(hit, "score", 0.0)),
                **(getattr(hit, "payload", {}) or {}),
            }
            for hit in hits
        ]


def build_qdrant_filter(filters: ProductSearchFilters) -> Any | None:
    if filters.is_empty:
        return None

    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

    conditions = []
    if filters.category:
        conditions.append(
            FieldCondition(key="category", match=MatchValue(value=filters.category))
        )
    if filters.brand:
        conditions.append(
            FieldCondition(key="brand", match=MatchValue(value=filters.brand))
        )
    if filters.min_price is not None or filters.max_price is not None:
        conditions.append(
            FieldCondition(
                key="price",
                range=Range(gte=filters.min_price, lte=filters.max_price),
            )
        )
    return Filter(must=conditions)


class VectorSearchService:
    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        encoder: EmbeddingModel | None = None,
        backend: VectorBackend | None = None,
    ):
        self.config = config or VectorStoreConfig.from_env()
        self.encoder = encoder or build_encoder(self.config)
        self.backend = backend or QdrantVectorBackend(
            host=self.config.qdrant_host, port=self.config.qdrant_port
        )

    def init_collection(self) -> None:
        self.backend.init_collection(
            self.config.collection_name, self.config.vector_size
        )

    def index_dataframe(self, products: pd.DataFrame) -> int:
        texts, payloads = prepare_products_for_indexing(products)
        vectors = self.encoder.encode(texts)
        self.backend.upsert_products(self.config.collection_name, vectors, payloads)
        return len(payloads)

    def index_products(self, products: Sequence[dict]) -> int:
        return self.index_dataframe(pd.DataFrame(products))

    def index_from_path(self, path: str | None = None) -> int:
        return self.index_dataframe(load_products(path or self.config.input_path))

    def search_products(
        self,
        query: str,
        filters: ProductSearchFilters | None = None,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Search query must not be empty.")

        filters = filters or ProductSearchFilters(category=category_filter)
        query_vector = self.encoder.encode([query])[0]
        query_filter = build_qdrant_filter(filters)
        return self.backend.search(
            self.config.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartShop Phase 4 semantic search.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--collection-name", default=DEFAULT_COLLECTION_NAME
        )
        command_parser.add_argument(
            "--embedding-model", default=DEFAULT_EMBEDDING_MODEL
        )
        command_parser.add_argument(
            "--embedding-backend",
            default=DEFAULT_EMBEDDING_BACKEND,
            choices=["sentence-transformers", "hashing"],
        )
        command_parser.add_argument(
            "--vector-size", type=int, default=DEFAULT_VECTOR_SIZE
        )
        command_parser.add_argument("--qdrant-host", default=DEFAULT_QDRANT_HOST)
        command_parser.add_argument(
            "--qdrant-port", type=int, default=DEFAULT_QDRANT_PORT
        )

    index_parser = subparsers.add_parser(
        "index", help="Index processed products in Qdrant."
    )
    add_common_options(index_parser)
    index_parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)

    search_parser = subparsers.add_parser(
        "search", help="Search products by natural language."
    )
    add_common_options(search_parser)
    search_parser.add_argument("query")
    search_parser.add_argument("--category")
    search_parser.add_argument("--brand")
    search_parser.add_argument("--min-price", type=float)
    search_parser.add_argument("--max-price", type=float)
    search_parser.add_argument("--top-k", type=int, default=5)

    return parser


def service_from_args(args: argparse.Namespace) -> VectorSearchService:
    return VectorSearchService(
        VectorStoreConfig(
            input_path=getattr(args, "input_path", DEFAULT_INPUT_PATH),
            collection_name=args.collection_name,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            vector_size=args.vector_size,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = service_from_args(args)
    service.init_collection()

    if args.command == "index":
        indexed_count = service.index_from_path(args.input_path)
        print(f"Indexed {indexed_count} products into '{args.collection_name}'.")
        return 0

    if args.command == "search":
        filters = ProductSearchFilters(
            category=args.category,
            brand=args.brand,
            min_price=args.min_price,
            max_price=args.max_price,
        )
        results = service.search_products(args.query, filters=filters, top_k=args.top_k)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
