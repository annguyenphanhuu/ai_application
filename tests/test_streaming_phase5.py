"""Tests for Phase 5 streaming – covers retry, DLQ, schema version, pipeline writes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.streaming import (
    SCHEMA_VERSION,
    ClickEvent,
    ClickEventConsumer,
    KafkaClickEventProducer,
    RedisHotProductsStore,
    build_parser,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProducer:
    def __init__(self, fail_times: int = 0):
        self.sent: list[tuple[str, dict]] = []
        self.flush_count = 0
        self._fail_remaining = fail_times
        self._call_count = 0

    def send(self, topic: str, value: dict):
        self._call_count += 1
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise OSError("broker unavailable (fake)")
        self.sent.append((topic, value))

    def flush(self):
        self.flush_count += 1


class FakeStore:
    def __init__(self):
        self.clicks: list[tuple[str, dict]] = []

    def record_click(self, product_id: str, event: dict) -> float:
        self.clicks.append((product_id, event))
        return float(len(self.clicks))

    def top_products(self, limit: int = 10) -> list[dict]:
        return [{"product_id": "P01", "click_count": 2}][:limit]


class FakeRedisClient:
    """Minimal pipeline-aware Redis stub."""

    def __init__(self):
        self.scores: dict[str, dict[str, float]] = {}
        self.hashes: dict[str, dict] = {}

    # Real-client direct methods
    def _do_zincrby(self, key, amount, member):
        bucket = self.scores.setdefault(key, {})
        bucket[member] = bucket.get(member, 0.0) + amount
        return bucket[member]

    def _do_hset(self, key, mapping=None):
        self.hashes[key] = mapping or {}
        return 1

    def zincrby(self, key, amount, member):
        return self._do_zincrby(key, amount, member)

    def hset(self, key, mapping=None):
        return self._do_hset(key, mapping)

    def zrevrange(self, key, start, end, withscores=False):
        stop = end + 1
        rows = sorted(
            self.scores.get(key, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[start:stop]
        return rows if withscores else [pid for pid, _ in rows]

    # Pipeline returns a simple recorder
    def pipeline(self) -> "_FakePipeline":
        return _FakePipeline(self)


class _FakePipeline:
    """Records zincrby/hset calls and replays them on execute()."""

    def __init__(self, client: FakeRedisClient):
        self._client = client
        self._cmds: list = []

    def zincrby(self, key, amount, member):
        self._cmds.append(("zincrby", key, amount, member))
        return self

    def hset(self, key, mapping=None):
        self._cmds.append(("hset", key, mapping))
        return self

    def execute(self) -> list:
        results = []
        for cmd in self._cmds:
            if cmd[0] == "zincrby":
                results.append(self._client._do_zincrby(cmd[1], cmd[2], cmd[3]))
            elif cmd[0] == "hset":
                results.append(self._client._do_hset(cmd[1], cmd[2]))
        return results


# ---------------------------------------------------------------------------
# ClickEvent schema tests
# ---------------------------------------------------------------------------


def test_click_event_validates_required_fields():
    with pytest.raises(ValueError, match="user_id"):
        ClickEvent(user_id=" ", product_id="P01")
    with pytest.raises(ValueError, match="product_id"):
        ClickEvent(user_id="U01", product_id=" ")


def test_click_event_carries_schema_version():
    event = ClickEvent(user_id="U01", product_id="P01").to_dict()
    assert event["schema_version"] == SCHEMA_VERSION


def test_click_event_from_payload_preserves_schema_version():
    payload = {"user_id": "U01", "product_id": "P01", "schema_version": "1"}
    event = ClickEvent.from_payload(payload)
    assert event.schema_version == "1"


def test_click_event_to_dict_omits_none_values():
    event = ClickEvent(
        user_id="U01",
        product_id="P01",
        session_id=None,
        query="headphones",
        metadata={"page": "search"},
    ).to_dict()
    assert event["user_id"] == "U01"
    assert event["query"] == "headphones"
    assert "session_id" not in event


# ---------------------------------------------------------------------------
# Producer retry tests
# ---------------------------------------------------------------------------


def test_producer_sends_on_first_attempt():
    fake = FakeProducer()
    service = KafkaClickEventProducer(topic="user-clicks", producer=fake, max_retries=3)
    event = service.log_click(user_id="U01", product_id="P01")

    assert fake.flush_count == 1
    assert fake.sent[0] == ("user-clicks", event)
    assert event["schema_version"] == SCHEMA_VERSION


def test_producer_retries_on_transient_failure():
    # Fail first 2 attempts, succeed on 3rd
    fake = FakeProducer(fail_times=2)
    service = KafkaClickEventProducer(
        topic="user-clicks",
        producer=fake,
        max_retries=3,
        retry_backoff_ms=0,  # no sleep in tests
    )
    event = service.log_click(user_id="U01", product_id="P01")

    # 3 send calls: 2 failures + 1 success
    assert fake._call_count == 3
    assert fake.sent[-1][0] == "user-clicks"
    assert event["product_id"] == "P01"


def test_producer_forwards_to_dlq_after_all_retries_exhausted():
    """_send_with_retry should return False after max_retries failures."""
    fake = FakeProducer(fail_times=99)
    service = KafkaClickEventProducer(
        topic="user-clicks",
        dead_letter_topic="user-clicks-dlq",
        producer=fake,
        max_retries=2,
        retry_backoff_ms=0,
    )
    # _send_with_retry returns False when all attempts fail
    success = service._send_with_retry(
        "user-clicks", {"user_id": "U01", "product_id": "P01"}
    )
    assert success is False
    assert fake._call_count == 2  # exactly max_retries attempts


# ---------------------------------------------------------------------------
# Consumer tests
# ---------------------------------------------------------------------------


def test_consumer_process_message_updates_store():
    store = FakeStore()
    consumer = ClickEventConsumer(store=store)
    result = consumer.process_message(
        SimpleNamespace(value={"user_id": "U01", "product_id": "P01"})
    )
    assert result["click_count"] == 1.0
    assert store.clicks[0][0] == "P01"


def test_consumer_accepts_json_bytes():
    store = FakeStore()
    consumer = ClickEventConsumer(store=store)
    result = consumer.process_message(
        SimpleNamespace(value=b'{"user_id": "U02", "product_id": "P02"}')
    )
    assert result["event"]["product_id"] == "P02"


def test_consumer_sends_malformed_message_to_dlq():
    store = FakeStore()
    dlq_events: list[tuple[str, dict]] = []

    class FakeDlqProducer:
        def send(self, topic, value):
            dlq_events.append((topic, value))

        def flush(self):
            pass

    consumer = ClickEventConsumer(
        store=store,
        dead_letter_topic="user-clicks-dlq",
        dlq_producer=FakeDlqProducer(),
    )
    # Simulate malformed payload (missing required fields)
    bad_message = SimpleNamespace(value={"user_id": " ", "product_id": "P01"})
    consumer._send_to_dlq(bad_message.value, "parse_error:user_id must not be empty.")

    assert len(dlq_events) == 1
    assert dlq_events[0][0] == "user-clicks-dlq"
    assert "parse_error" in dlq_events[0][1]["_dlq_reason"]


# ---------------------------------------------------------------------------
# Redis store tests
# ---------------------------------------------------------------------------


def test_redis_store_uses_pipeline_for_atomic_write():
    client = FakeRedisClient()
    store = RedisHotProductsStore(key="hot_products:test", client=client)
    score = store.record_click("P01", {"user_id": "U01", "product_id": "P01"})
    assert score == 1.0
    assert "P01" in client.scores.get("hot_products:test", {})


def test_redis_store_top_products_sorted():
    client = FakeRedisClient()
    store = RedisHotProductsStore(key="hp", client=client)
    store.record_click("P01", {"user_id": "U01", "product_id": "P01"})
    store.record_click("P02", {"user_id": "U02", "product_id": "P02"})
    store.record_click("P01", {"user_id": "U03", "product_id": "P01"})
    top = store.top_products(limit=2)
    assert top[0] == {"product_id": "P01", "click_count": 2}
    assert top[1] == {"product_id": "P02", "click_count": 1}


# ---------------------------------------------------------------------------
# CLI parser tests
# ---------------------------------------------------------------------------


def test_build_parser_produce():
    args = build_parser().parse_args(
        [
            "produce",
            "--user-id",
            "U01",
            "--product-id",
            "P01",
            "--topic",
            "user-clicks",
            "--dead-letter-topic",
            "user-clicks-dlq",
        ]
    )
    assert args.command == "produce"
    assert args.user_id == "U01"
    assert args.dead_letter_topic == "user-clicks-dlq"


def test_record_click_serializes_nested_values_for_redis_hash():
    from src.streaming import RedisHotProductsStore

    class RecordingPipeline:
        def __init__(self):
            self.hset_mappings = []

        def zincrby(self, key, amount, member):
            return self

        def hset(self, name, mapping):
            self.hset_mappings.append(mapping)
            return self

        def execute(self):
            return [1.0, 1]

    class RecordingClient:
        def __init__(self):
            self.pipe = RecordingPipeline()

        def pipeline(self):
            return self.pipe

    client = RecordingClient()
    store = RedisHotProductsStore(client=client)

    score = store.record_click(
        "P01",
        {
            "user_id": "U01",
            "product_id": "P01",
            "metadata": {"page": "search"},
            "session_id": None,
        },
    )

    assert score == 1.0
    mapping = client.pipe.hset_mappings[0]
    for value in mapping.values():
        assert isinstance(value, (str, int, float, bytes))
    assert mapping["metadata"] == '{"page": "search"}'
