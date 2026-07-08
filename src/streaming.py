"""Phase 5 real-time clickstream ingestion for SmartShop.

Enhancements:
- schema_version field on every event (v1) for forward compatibility.
- KafkaClickEventProducer: retry with exponential back-off, dead-letter topic on
  repeated failure.
- RedisHotProductsStore: pipeline-based atomic writes, error logging.
- ClickEventConsumer: per-message error handling with dead-letter forwarding.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "user-clicks"
DEFAULT_DEAD_LETTER_TOPIC = "user-clicks-dlq"
DEFAULT_GROUP_ID = "smartshop-click-consumer"
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_HOT_PRODUCTS_KEY = "hot_products"
SCHEMA_VERSION = "1"

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class EventProducer(Protocol):
    def send(self, topic: str, value: dict) -> Any:
        """Send a structured event to a broker topic."""

    def flush(self) -> None:
        """Flush buffered events."""


class HotProductsStore(Protocol):
    def record_click(self, product_id: str, event: dict) -> float:
        """Record one product click and return the current score."""

    def top_products(self, limit: int = 10) -> list[dict]:
        """Return top products ordered by click score."""


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClickEvent:
    user_id: str
    product_id: str
    event_type: str = "product_click"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    session_id: str | None = None
    query: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION  # ← NEW: schema versioning

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("user_id must not be empty.")
        if not self.product_id.strip():
            raise ValueError("product_id must not be empty.")
        if not self.event_type.strip():
            raise ValueError("event_type must not be empty.")

    def to_dict(self) -> dict:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def from_payload(cls, payload: dict) -> "ClickEvent":
        return cls(
            user_id=str(payload.get("user_id", "")),
            product_id=str(payload.get("product_id", "")),
            event_type=str(payload.get("event_type", "product_click")),
            timestamp=str(
                payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
            ),
            session_id=payload.get("session_id"),
            query=payload.get("query"),
            metadata=dict(payload.get("metadata") or {}),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamingConfig:
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS
    topic: str = DEFAULT_TOPIC
    dead_letter_topic: str = DEFAULT_DEAD_LETTER_TOPIC
    group_id: str = DEFAULT_GROUP_ID
    redis_host: str = DEFAULT_REDIS_HOST
    redis_port: int = DEFAULT_REDIS_PORT
    redis_db: int = 0
    hot_products_key: str = DEFAULT_HOT_PRODUCTS_KEY
    producer_max_retries: int = 3
    producer_retry_backoff_ms: int = 200

    @classmethod
    def from_env(cls) -> "StreamingConfig":
        return cls(
            bootstrap_servers=os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS
            ),
            topic=os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC),
            dead_letter_topic=os.getenv(
                "KAFKA_DEAD_LETTER_TOPIC", DEFAULT_DEAD_LETTER_TOPIC
            ),
            group_id=os.getenv("KAFKA_GROUP_ID", DEFAULT_GROUP_ID),
            redis_host=os.getenv("REDIS_HOST", DEFAULT_REDIS_HOST),
            redis_port=int(os.getenv("REDIS_PORT", str(DEFAULT_REDIS_PORT))),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            hot_products_key=os.getenv(
                "SMARTSHOP_HOT_PRODUCTS_KEY", DEFAULT_HOT_PRODUCTS_KEY
            ),
            producer_max_retries=int(os.getenv("KAFKA_PRODUCER_MAX_RETRIES", "3")),
            producer_retry_backoff_ms=int(
                os.getenv("KAFKA_PRODUCER_RETRY_BACKOFF_MS", "200")
            ),
        )


# ---------------------------------------------------------------------------
# Producer with retry + dead-letter
# ---------------------------------------------------------------------------


class KafkaClickEventProducer:
    """Kafka producer with exponential-backoff retry and dead-letter topic."""

    def __init__(
        self,
        bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
        topic: str = DEFAULT_TOPIC,
        dead_letter_topic: str = DEFAULT_DEAD_LETTER_TOPIC,
        max_retries: int = 3,
        retry_backoff_ms: int = 200,
        producer: EventProducer | None = None,
    ):
        self.topic = topic
        self.dead_letter_topic = dead_letter_topic
        self.max_retries = max_retries
        self.retry_backoff_ms = retry_backoff_ms
        self.producer = producer or self._build_producer(bootstrap_servers)

    @staticmethod
    def _build_producer(bootstrap_servers: str) -> EventProducer:
        try:
            from kafka import KafkaProducer
        except ImportError as exc:
            raise RuntimeError(
                "kafka-python is required. Install with `pip install -r requirements.txt`."
            ) from exc

        return KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            # Built-in Kafka producer retries for transient broker errors
            retries=3,
            retry_backoff_ms=100,
            acks="all",
        )

    def _send_with_retry(self, topic: str, event: dict) -> bool:
        """Try sending to *topic* up to max_retries times. Returns True on success."""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.producer.send(topic, value=event)
                self.producer.flush()
                return True
            except Exception as exc:  # noqa: BLE001
                wait_s = (self.retry_backoff_ms / 1000) * (2 ** (attempt - 1))
                logger.warning(
                    "Kafka send attempt %d/%d failed (topic=%s): %s. Retrying in %.2fs.",
                    attempt,
                    self.max_retries,
                    topic,
                    exc,
                    wait_s,
                )
                if attempt < self.max_retries:
                    time.sleep(wait_s)
        return False

    def log_click(
        self,
        user_id: str,
        product_id: str,
        session_id: str | None = None,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        event = ClickEvent(
            user_id=user_id,
            product_id=product_id,
            session_id=session_id,
            query=query,
            metadata=metadata or {},
        ).to_dict()

        success = self._send_with_retry(self.topic, event)
        if not success:
            # Annotate with failure reason and forward to dead-letter topic
            dlq_event = {**event, "_dlq_reason": "producer_max_retries_exceeded"}
            try:
                self.producer.send(self.dead_letter_topic, value=dlq_event)
                self.producer.flush()
                logger.error(
                    "Event forwarded to dead-letter topic %s after %d failed attempts.",
                    self.dead_letter_topic,
                    self.max_retries,
                )
            except Exception as dlq_exc:  # noqa: BLE001
                logger.critical(
                    "Dead-letter send also failed: %s. Event lost: %s",
                    dlq_exc,
                    event,
                )
        return event


# ---------------------------------------------------------------------------
# Redis hot-products store
# ---------------------------------------------------------------------------


class RedisHotProductsStore:
    def __init__(
        self,
        host: str = DEFAULT_REDIS_HOST,
        port: int = DEFAULT_REDIS_PORT,
        db: int = 0,
        key: str = DEFAULT_HOT_PRODUCTS_KEY,
        client: Any | None = None,
    ):
        self.key = key
        self.client = client or self._build_client(host, port, db)

    @staticmethod
    def _build_client(host: str, port: int, db: int) -> Any:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "redis is required. Install with `pip install -r requirements.txt`."
            ) from exc

        return redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

    def record_click(self, product_id: str, event: dict) -> float:
        try:
            pipe = self.client.pipeline()
            pipe.zincrby(self.key, 1, product_id)
            pipe.hset(f"product_click:last:{product_id}", mapping=event)
            results = pipe.execute()
            return float(results[0])
        except Exception as exc:  # noqa: BLE001
            logger.error("Redis record_click failed for %s: %s", product_id, exc)
            return 0.0

    def top_products(self, limit: int = 10) -> list[dict]:
        try:
            rows = self.client.zrevrange(self.key, 0, limit - 1, withscores=True)
            return [
                {"product_id": product_id, "click_count": int(score)}
                for product_id, score in rows
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("Redis top_products failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Consumer with per-message error handling + DLQ forwarding
# ---------------------------------------------------------------------------


class ClickEventConsumer:
    def __init__(
        self,
        store: HotProductsStore,
        topic: str = DEFAULT_TOPIC,
        bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
        group_id: str = DEFAULT_GROUP_ID,
        dead_letter_topic: str = DEFAULT_DEAD_LETTER_TOPIC,
        consumer: Iterable[Any] | None = None,
        dlq_producer: EventProducer | None = None,
    ):
        self.store = store
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.dead_letter_topic = dead_letter_topic
        self.consumer = consumer
        self._dlq_producer = dlq_producer

    def _build_consumer(self) -> Iterable[Any]:
        try:
            from kafka import KafkaConsumer
        except ImportError as exc:
            raise RuntimeError(
                "kafka-python is required. Install with `pip install -r requirements.txt`."
            ) from exc

        return KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers.split(","),
            group_id=self.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

    def _get_dlq_producer(self) -> EventProducer | None:
        if self._dlq_producer is not None:
            return self._dlq_producer
        try:
            from kafka import KafkaProducer

            self._dlq_producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            return self._dlq_producer
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create DLQ producer: %s", exc)
            return None

    @staticmethod
    def message_to_payload(message: Any) -> dict:
        if isinstance(message, dict):
            return message
        value = getattr(message, "value", message)
        if isinstance(value, bytes):
            return json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
        if isinstance(value, dict):
            return value
        raise TypeError(f"Unsupported message payload type: {type(value)!r}")

    def process_message(self, message: Any) -> dict:
        payload = self.message_to_payload(message)
        event = ClickEvent.from_payload(payload).to_dict()
        click_count = self.store.record_click(event["product_id"], event)
        return {"event": event, "click_count": click_count}

    def _send_to_dlq(self, raw_payload: Any, reason: str) -> None:
        producer = self._get_dlq_producer()
        if producer is None:
            return
        dlq_event = {
            "raw_payload": str(raw_payload),
            "_dlq_reason": reason,
            "_dlq_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            producer.send(self.dead_letter_topic, value=dlq_event)
            producer.flush()
            logger.warning("Sent message to DLQ (%s): %s", self.dead_letter_topic, reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("DLQ send failed: %s", exc)

    def consume_forever(self) -> None:
        consumer = self.consumer or self._build_consumer()
        for message in consumer:
            raw = getattr(message, "value", message)
            try:
                result = self.process_message(message)
                event = result["event"]
                # Validate schema version
                sv = event.get("schema_version", SCHEMA_VERSION)
                if sv != SCHEMA_VERSION:
                    logger.warning(
                        "Unexpected schema_version=%s for product_id=%s",
                        sv,
                        event.get("product_id"),
                    )
                logger.info(
                    "Recorded click user_id=%s product_id=%s click_count=%d",
                    event["user_id"],
                    event["product_id"],
                    int(result["click_count"]),
                )
            except (ValueError, TypeError, KeyError) as exc:
                logger.error("Malformed message, sending to DLQ: %s", exc)
                self._send_to_dlq(raw, f"parse_error:{exc}")
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error processing message, sending to DLQ: %s", exc)
                self._send_to_dlq(raw, f"unexpected_error:{exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartShop Phase 5 streaming.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_streaming_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--bootstrap-servers", default=DEFAULT_BOOTSTRAP_SERVERS)
        p.add_argument("--topic", default=DEFAULT_TOPIC)
        p.add_argument("--dead-letter-topic", default=DEFAULT_DEAD_LETTER_TOPIC)

    produce_parser = subparsers.add_parser("produce", help="Publish one click event.")
    add_streaming_options(produce_parser)
    produce_parser.add_argument("--user-id", required=True)
    produce_parser.add_argument("--product-id", required=True)
    produce_parser.add_argument("--session-id")
    produce_parser.add_argument("--query")
    produce_parser.add_argument("--max-retries", type=int, default=3)

    consume_parser = subparsers.add_parser("consume", help="Consume click events.")
    add_streaming_options(consume_parser)
    consume_parser.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    consume_parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST)
    consume_parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT)
    consume_parser.add_argument("--redis-db", type=int, default=0)
    consume_parser.add_argument("--hot-products-key", default=DEFAULT_HOT_PRODUCTS_KEY)

    top_parser = subparsers.add_parser("top", help="Show hot products from Redis.")
    top_parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST)
    top_parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT)
    top_parser.add_argument("--redis-db", type=int, default=0)
    top_parser.add_argument("--hot-products-key", default=DEFAULT_HOT_PRODUCTS_KEY)
    top_parser.add_argument("--limit", type=int, default=10)

    return parser


def redis_store_from_args(args: argparse.Namespace) -> RedisHotProductsStore:
    return RedisHotProductsStore(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        key=args.hot_products_key,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    if args.command == "produce":
        producer = KafkaClickEventProducer(
            bootstrap_servers=args.bootstrap_servers,
            topic=args.topic,
            dead_letter_topic=args.dead_letter_topic,
            max_retries=args.max_retries,
        )
        event = producer.log_click(
            user_id=args.user_id,
            product_id=args.product_id,
            session_id=args.session_id,
            query=args.query,
        )
        print(json.dumps(event, indent=2, ensure_ascii=False))
        return 0

    if args.command == "consume":
        store = redis_store_from_args(args)
        consumer = ClickEventConsumer(
            store=store,
            topic=args.topic,
            bootstrap_servers=args.bootstrap_servers,
            group_id=args.group_id,
            dead_letter_topic=args.dead_letter_topic,
        )
        consumer.consume_forever()
        return 0

    if args.command == "top":
        store = redis_store_from_args(args)
        print(json.dumps(store.top_products(args.limit), indent=2, ensure_ascii=False))
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
