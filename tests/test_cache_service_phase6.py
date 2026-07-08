import json

import pytest

from src.cache_service import (
    RedisConfig,
    RedisService,
    SessionMessage,
    build_parser,
    normalize_query,
)


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.operations = []

    def incr(self, key):
        self.operations.append(("incr", key))
        return self

    def ttl(self, key):
        self.operations.append(("ttl", key))
        return self

    def execute(self):
        results = []
        for operation, key in self.operations:
            if operation == "incr":
                results.append(self.client.incr(key))
            elif operation == "ttl":
                results.append(self.client.ttl(key))
        return results


class FakeRedisClient:
    def __init__(self):
        self.values = {}
        self.lists = {}
        self.expiries = {}
        self.deleted = []

    def get(self, name):
        return self.values.get(name)

    def setex(self, name, time, value):
        self.values[name] = value
        self.expiries[name] = time

    def delete(self, *names):
        deleted_count = 0
        for name in names:
            existed = name in self.values or name in self.lists
            self.values.pop(name, None)
            self.lists.pop(name, None)
            self.expiries.pop(name, None)
            self.deleted.append(name)
            deleted_count += int(existed)
        return deleted_count

    def rpush(self, name, *values):
        bucket = self.lists.setdefault(name, [])
        bucket.extend(values)
        return len(bucket)

    def lrange(self, name, start, end):
        rows = self.lists.get(name, [])
        if start < 0:
            start = max(len(rows) + start, 0)
        if end < 0:
            end = len(rows) + end
        return rows[start : end + 1]

    def ltrim(self, name, start, end):
        rows = self.lists.get(name, [])
        if start < 0:
            start = max(len(rows) + start, 0)
        if end < 0:
            end = len(rows) + end
        self.lists[name] = rows[start : end + 1]

    def expire(self, name, time):
        self.expiries[name] = time

    def incr(self, name):
        self.values[name] = str(int(self.values.get(name, "0")) + 1)
        return int(self.values[name])

    def ttl(self, name):
        return self.expiries.get(name, -1 if name in self.values else -2)

    def pipeline(self):
        return FakePipeline(self)


def build_service():
    return RedisService(
        config=RedisConfig(
            key_prefix="testshop",
            search_ttl_seconds=120,
            session_ttl_seconds=3600,
            rate_limit=2,
            rate_window_seconds=30,
        ),
        client=FakeRedisClient(),
    )


def test_normalize_query_collapses_whitespace_and_case():
    assert normalize_query("  Noise   Cancelling HEADPHONES ") == (
        "noise cancelling headphones"
    )

    with pytest.raises(ValueError, match="must not be empty"):
        normalize_query("   ")


def test_search_cache_round_trips_results_with_stable_key_and_ttl():
    service = build_service()
    results = [{"product_id": "P01", "score": 0.91}]
    filters = {"category": "Electronics", "max_price": 200}

    key = service.set_cached_search(
        "Noise Cancelling Headphones",
        results,
        filters=filters,
        top_k=3,
    )

    assert key == service.build_search_cache_key(
        " noise cancelling headphones ",
        filters={"max_price": 200, "category": "Electronics"},
        top_k=3,
    )
    assert service.client.expiries[key] == 120
    assert (
        service.get_cached_search(
            "noise cancelling headphones",
            filters=filters,
            top_k=3,
        )
        == results
    )


def test_search_cache_miss_returns_none():
    service = build_service()

    assert service.get_cached_search("office chair") is None


def test_session_messages_are_appended_trimmed_and_expire():
    service = build_service()

    service.append_session_message("S01", "user", "hello", max_messages=2)
    service.append_session_message("S01", "assistant", "hi there", max_messages=2)
    service.append_session_message("S01", "user", "show headphones", max_messages=2)

    key = service.key("session", "S01")
    messages = service.get_session_messages("S01")
    assert [message["content"] for message in messages] == [
        "hi there",
        "show headphones",
    ]
    assert service.client.expiries[key] == 3600


def test_session_limit_reads_latest_messages():
    service = build_service()
    service.append_session_message("S01", "user", "first")
    service.append_session_message("S01", "assistant", "second")
    service.append_session_message("S01", "user", "third")

    assert [row["content"] for row in service.get_session_messages("S01", limit=2)] == [
        "second",
        "third",
    ]


def test_clear_session_deletes_session_key():
    service = build_service()
    service.append_session_message("S01", "user", "hello")

    assert service.clear_session("S01") == 1
    assert service.get_session_messages("S01") == []


def test_session_message_validates_required_fields():
    with pytest.raises(ValueError, match="role"):
        SessionMessage(role=" ", content="hello")

    with pytest.raises(ValueError, match="content"):
        SessionMessage(role="user", content=" ")


def test_fixed_window_rate_limit_blocks_after_limit():
    service = build_service()

    first = service.check_rate_limit("user-1")
    second = service.check_rate_limit("user-1")
    third = service.check_rate_limit("user-1")

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0
    assert third.allowed is False
    assert third.remaining == 0
    assert third.reset_after_seconds == 30


def test_build_parser_parses_rate_check_command():
    args = build_parser().parse_args(
        [
            "rate-check",
            "--identity",
            "127.0.0.1",
            "--limit",
            "5",
            "--window-seconds",
            "10",
        ]
    )

    assert args.command == "rate-check"
    assert args.identity == "127.0.0.1"
    assert args.limit == 5
    assert args.window_seconds == 10


def test_redis_config_can_be_loaded_from_env(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "redis")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "2")
    monkeypatch.setenv("SMARTSHOP_REDIS_KEY_PREFIX", "prodshop")
    monkeypatch.setenv("SMARTSHOP_RATE_LIMIT", "100")

    config = RedisConfig.from_env()

    assert config.host == "redis"
    assert config.port == 6380
    assert config.db == 2
    assert config.key_prefix == "prodshop"
    assert config.rate_limit == 100


def test_cli_cache_payload_is_valid_json_shape():
    service = build_service()
    key = service.set_cached_search("headphones", json.loads('[{"product_id":"P01"}]'))

    assert key.startswith("testshop:search:")
