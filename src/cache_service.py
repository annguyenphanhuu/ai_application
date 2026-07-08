"""Phase 6 Redis cache, session memory, and rate limiting for SmartShop."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Protocol, Sequence


DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_KEY_PREFIX = "smartshop"
DEFAULT_SEARCH_TTL_SECONDS = 300
DEFAULT_SESSION_TTL_SECONDS = 24 * 60 * 60
DEFAULT_RATE_LIMIT = 60
DEFAULT_RATE_WINDOW_SECONDS = 60


class RedisClient(Protocol):
    def get(self, name: str) -> Any:
        """Return a value by key."""

    def setex(self, name: str, time: int, value: str) -> Any:
        """Set a value with TTL."""

    def delete(self, *names: str) -> Any:
        """Delete keys."""

    def rpush(self, name: str, *values: str) -> Any:
        """Append values to a list."""

    def lrange(self, name: str, start: int, end: int) -> list[Any]:
        """Read list values."""

    def ltrim(self, name: str, start: int, end: int) -> Any:
        """Trim list values."""

    def expire(self, name: str, time: int) -> Any:
        """Set key expiry."""

    def incr(self, name: str) -> int:
        """Increment a numeric key."""

    def ttl(self, name: str) -> int:
        """Return key TTL."""

    def pipeline(self) -> Any:
        """Return a Redis pipeline."""


@dataclass(frozen=True)
class RedisConfig:
    host: str = DEFAULT_REDIS_HOST
    port: int = DEFAULT_REDIS_PORT
    db: int = 0
    key_prefix: str = DEFAULT_KEY_PREFIX
    search_ttl_seconds: int = DEFAULT_SEARCH_TTL_SECONDS
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS
    rate_limit: int = DEFAULT_RATE_LIMIT
    rate_window_seconds: int = DEFAULT_RATE_WINDOW_SECONDS

    @classmethod
    def from_env(cls) -> "RedisConfig":
        return cls(
            host=os.getenv(
                "REDIS_HOST", os.getenv("SMARTSHOP_REDIS_HOST", DEFAULT_REDIS_HOST)
            ),
            port=int(
                os.getenv(
                    "REDIS_PORT",
                    os.getenv("SMARTSHOP_REDIS_PORT", str(DEFAULT_REDIS_PORT)),
                )
            ),
            db=int(os.getenv("REDIS_DB", os.getenv("SMARTSHOP_REDIS_DB", "0"))),
            key_prefix=os.getenv("SMARTSHOP_REDIS_KEY_PREFIX", DEFAULT_KEY_PREFIX),
            search_ttl_seconds=int(
                os.getenv(
                    "SMARTSHOP_SEARCH_TTL_SECONDS", str(DEFAULT_SEARCH_TTL_SECONDS)
                )
            ),
            session_ttl_seconds=int(
                os.getenv(
                    "SMARTSHOP_SESSION_TTL_SECONDS", str(DEFAULT_SESSION_TTL_SECONDS)
                )
            ),
            rate_limit=int(os.getenv("SMARTSHOP_RATE_LIMIT", str(DEFAULT_RATE_LIMIT))),
            rate_window_seconds=int(
                os.getenv(
                    "SMARTSHOP_RATE_WINDOW_SECONDS", str(DEFAULT_RATE_WINDOW_SECONDS)
                )
            ),
        )


@dataclass(frozen=True)
class SessionMessage:
    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("Session message role must not be empty.")
        if not self.content.strip():
            raise ValueError("Session message content must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SessionMessage":
        return cls(
            role=str(payload.get("role", "")),
            content=str(payload.get("content", "")),
            timestamp=str(
                payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_after_seconds: int
    key: str


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def normalize_query(query: str) -> str:
    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        raise ValueError("Search query must not be empty.")
    return normalized


class RedisService:
    def __init__(
        self,
        config: RedisConfig | None = None,
        client: RedisClient | None = None,
    ):
        self.config = config or RedisConfig.from_env()
        self.client = client or self._build_client(self.config)

    @staticmethod
    def _build_client(config: RedisConfig) -> RedisClient:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "redis is required for Phase 6 cache/session/rate limiting. "
                "Install it with `pip install -r requirements.txt` or update the Conda env."
            ) from exc

        return redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            decode_responses=True,
        )

    def key(self, *parts: str) -> str:
        clean_parts = [self.config.key_prefix.strip(":")]
        clean_parts.extend(str(part).strip(":") for part in parts if str(part))
        return ":".join(clean_parts)

    def build_search_cache_key(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> str:
        payload = {
            "query": normalize_query(query),
            "filters": filters or {},
            "top_k": top_k,
        }
        digest = sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:24]
        return self.key("search", digest)

    def get_cached_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict] | None:
        cached_value = self.client.get(
            self.build_search_cache_key(query, filters=filters, top_k=top_k)
        )
        if not cached_value:
            return None
        return json.loads(cached_value)

    def set_cached_search(
        self,
        query: str,
        results: list[dict],
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        ttl_seconds: int | None = None,
    ) -> str:
        key = self.build_search_cache_key(query, filters=filters, top_k=top_k)
        self.client.setex(
            key,
            ttl_seconds or self.config.search_ttl_seconds,
            json.dumps(results, ensure_ascii=False),
        )
        return key

    def append_session_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        max_messages: int = 50,
        ttl_seconds: int | None = None,
    ) -> SessionMessage:
        if not session_id.strip():
            raise ValueError("session_id must not be empty.")
        if max_messages < 1:
            raise ValueError("max_messages must be greater than zero.")

        message = SessionMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        key = self.key("session", session_id)
        self.client.rpush(key, json.dumps(message.to_dict(), ensure_ascii=False))
        self.client.ltrim(key, -max_messages, -1)
        self.client.expire(key, ttl_seconds or self.config.session_ttl_seconds)
        return message

    def get_session_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not session_id.strip():
            raise ValueError("session_id must not be empty.")
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than zero.")

        key = self.key("session", session_id)
        start = -limit if limit else 0
        rows = self.client.lrange(key, start, -1)
        return [json.loads(row) for row in rows]

    def clear_session(self, session_id: str) -> int:
        if not session_id.strip():
            raise ValueError("session_id must not be empty.")
        return int(self.client.delete(self.key("session", session_id)))

    def check_rate_limit(
        self,
        identity: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> RateLimitResult:
        if not identity.strip():
            raise ValueError("Rate limit identity must not be empty.")

        limit = limit or self.config.rate_limit
        window_seconds = window_seconds or self.config.rate_window_seconds
        if limit < 1:
            raise ValueError("limit must be greater than zero.")
        if window_seconds < 1:
            raise ValueError("window_seconds must be greater than zero.")

        key = self.key("rate", identity)
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        current_count, current_ttl = pipe.execute()

        if current_ttl == -1:
            self.client.expire(key, window_seconds)
            current_ttl = window_seconds
        elif current_ttl == -2:
            current_ttl = window_seconds

        remaining = max(limit - int(current_count), 0)
        return RateLimitResult(
            allowed=int(current_count) <= limit,
            limit=limit,
            remaining=remaining,
            reset_after_seconds=int(current_ttl),
            key=key,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartShop Phase 6 Redis utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_redis_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST)
        command_parser.add_argument(
            "--redis-port", type=int, default=DEFAULT_REDIS_PORT
        )
        command_parser.add_argument("--redis-db", type=int, default=0)
        command_parser.add_argument("--key-prefix", default=DEFAULT_KEY_PREFIX)

    cache_parser = subparsers.add_parser("cache-set", help="Cache search results JSON.")
    add_redis_options(cache_parser)
    cache_parser.add_argument("--query", required=True)
    cache_parser.add_argument("--results-json", required=True)
    cache_parser.add_argument(
        "--ttl-seconds", type=int, default=DEFAULT_SEARCH_TTL_SECONDS
    )

    read_cache_parser = subparsers.add_parser(
        "cache-get", help="Read cached search results."
    )
    add_redis_options(read_cache_parser)
    read_cache_parser.add_argument("--query", required=True)

    session_parser = subparsers.add_parser(
        "session-add", help="Append a session message."
    )
    add_redis_options(session_parser)
    session_parser.add_argument("--session-id", required=True)
    session_parser.add_argument("--role", required=True)
    session_parser.add_argument("--content", required=True)

    read_session_parser = subparsers.add_parser(
        "session-get", help="Read session messages."
    )
    add_redis_options(read_session_parser)
    read_session_parser.add_argument("--session-id", required=True)
    read_session_parser.add_argument("--limit", type=int)

    rate_parser = subparsers.add_parser(
        "rate-check", help="Check fixed-window rate limit."
    )
    add_redis_options(rate_parser)
    rate_parser.add_argument("--identity", required=True)
    rate_parser.add_argument("--limit", type=int, default=DEFAULT_RATE_LIMIT)
    rate_parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_RATE_WINDOW_SECONDS,
    )

    return parser


def service_from_args(args: argparse.Namespace) -> RedisService:
    return RedisService(
        RedisConfig(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            key_prefix=args.key_prefix,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = service_from_args(args)

    if args.command == "cache-set":
        key = service.set_cached_search(
            query=args.query,
            results=json.loads(args.results_json),
            ttl_seconds=args.ttl_seconds,
        )
        print(json.dumps({"key": key, "status": "cached"}, indent=2))
        return 0

    if args.command == "cache-get":
        print(
            json.dumps(
                service.get_cached_search(args.query),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "session-add":
        message = service.append_session_message(
            session_id=args.session_id,
            role=args.role,
            content=args.content,
        )
        print(json.dumps(message.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "session-get":
        print(
            json.dumps(
                service.get_session_messages(args.session_id, limit=args.limit),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "rate-check":
        result = service.check_rate_limit(
            identity=args.identity,
            limit=args.limit,
            window_seconds=args.window_seconds,
        )
        print(json.dumps(asdict(result), indent=2))
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
