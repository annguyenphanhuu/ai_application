"""Phase 6 Redis cache, session memory, and rate limiting for SmartShop."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
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
DEFAULT_RATE_LIMIT_ALGORITHM = "token_bucket"
DEV_ENVIRONMENTS = {"dev", "local", "test"}


def _memory_fallback_default() -> bool:
    override = os.getenv("SMARTSHOP_REDIS_ALLOW_MEMORY_FALLBACK")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    environment = (os.getenv("SMARTSHOP_ENV") or os.getenv("APP_ENV") or "dev").strip()
    return environment.lower() in DEV_ENVIRONMENTS


class RedisClient(Protocol):
    def get(self, name: str) -> Any:
        """Return a value by key."""

    def setex(self, name: str, time: int, value: str) -> Any:
        """Set a value with TTL."""

    def ping(self) -> Any:
        """Return Redis connectivity status."""

    def delete(self, *names: str) -> Any:
        """Delete keys."""

    def rpush(self, name: str, *values: str) -> Any:
        """Append values to a list."""

    def lrange(self, name: str, start: int, end: int) -> list[Any]:
        """Read list values."""

    def ltrim(self, name: str, start: int, end: int) -> Any:
        """Trim list values."""

    def lrem(self, name: str, count: int, value: str) -> Any:
        """Remove list values."""

    def expire(self, name: str, time: int) -> Any:
        """Set key expiry."""

    def incr(self, name: str) -> int:
        """Increment a numeric key."""

    def ttl(self, name: str) -> int:
        """Return key TTL."""

    def pipeline(self) -> Any:
        """Return a Redis pipeline."""

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        """Evaluate a Lua script."""


@dataclass(frozen=True)
class RedisConfig:
    host: str = DEFAULT_REDIS_HOST
    port: int = DEFAULT_REDIS_PORT
    db: int = 0
    password: str | None = None
    allow_memory_fallback: bool = True
    key_prefix: str = DEFAULT_KEY_PREFIX
    search_ttl_seconds: int = DEFAULT_SEARCH_TTL_SECONDS
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS
    rate_limit: int = DEFAULT_RATE_LIMIT
    rate_window_seconds: int = DEFAULT_RATE_WINDOW_SECONDS
    rate_limit_algorithm: str = DEFAULT_RATE_LIMIT_ALGORITHM

    @classmethod
    def from_env(cls) -> "RedisConfig":
        environment = os.getenv("SMARTSHOP_ENV") or os.getenv("APP_ENV") or "dev"
        key_prefix = os.getenv(
            "SMARTSHOP_REDIS_KEY_PREFIX",
            f"{DEFAULT_KEY_PREFIX}:{environment.strip().lower()}",
        )
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
            password=os.getenv(
                "REDIS_PASSWORD", os.getenv("SMARTSHOP_REDIS_PASSWORD", "")
            )
            or None,
            allow_memory_fallback=_memory_fallback_default(),
            key_prefix=key_prefix,
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
            rate_limit_algorithm=os.getenv(
                "SMARTSHOP_RATE_LIMIT_ALGORITHM", DEFAULT_RATE_LIMIT_ALGORITHM
            ).strip()
            or DEFAULT_RATE_LIMIT_ALGORITHM,
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
class HumanApprovalRequest:
    request_id: str
    session_id: str
    message: str
    history: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    status: str = "pending"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: str | None = None
    resolved_by: str | None = None
    note: str | None = None
    consumed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("approval request_id must not be empty.")
        if not self.session_id.strip():
            raise ValueError("approval session_id must not be empty.")
        if not self.message.strip():
            raise ValueError("approval message must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "HumanApprovalRequest":
        return cls(
            request_id=str(payload.get("request_id", "")),
            session_id=str(payload.get("session_id", "")),
            message=str(payload.get("message", "")),
            history=list(payload.get("history") or []),
            reason=str(payload.get("reason", "")),
            status=str(payload.get("status", "pending")),
            created_at=str(
                payload.get("created_at") or datetime.now(timezone.utc).isoformat()
            ),
            resolved_at=payload.get("resolved_at"),
            resolved_by=payload.get("resolved_by"),
            note=payload.get("note"),
            consumed_at=payload.get("consumed_at"),
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


class MemoryRedisClient:
    def __init__(self):
        self._data = {}
        self._ttls = {}
        self._lists = {}

    def ping(self):
        return True

    def get(self, name: str) -> Any:
        import time

        if name in self._ttls and self._ttls[name] < time.time():
            self.delete(name)
            return None
        return self._data.get(name)

    def setex(self, name: str, seconds: int, value: str) -> Any:
        import time

        self._data[name] = value
        self._ttls[name] = time.time() + seconds
        return True

    def delete(self, *names: str) -> Any:
        count = 0
        for name in names:
            if name in self._data:
                del self._data[name]
                count += 1
            if name in self._ttls:
                del self._ttls[name]
            if name in self._lists:
                del self._lists[name]
                count += 1
        return count

    def rpush(self, name: str, *values: str) -> Any:
        if name not in self._lists:
            self._lists[name] = []
        self._lists[name].extend(values)
        return len(self._lists[name])

    def lrange(self, name: str, start: int, end: int) -> list[Any]:
        lst = self._lists.get(name, [])
        if not lst:
            return []
        end_idx = len(lst) if end == -1 else end + 1
        start_idx = len(lst) + start if start < 0 else start
        return lst[start_idx:end_idx]

    def ltrim(self, name: str, start: int, end: int) -> Any:
        lst = self._lists.get(name, [])
        if not lst:
            return True
        start_idx = len(lst) + start if start < 0 else start
        end_idx = len(lst) + end + 1 if end < 0 else end + 1
        self._lists[name] = lst[start_idx:end_idx]
        return True

    def lrem(self, name: str, count: int, value: str) -> Any:
        lst = self._lists.get(name, [])
        if not lst:
            return 0
        removed = 0
        new_lst = []
        for item in lst:
            if item == value:
                removed += 1
            else:
                new_lst.append(item)
        self._lists[name] = new_lst
        return removed

    def expire(self, name: str, seconds: int) -> Any:
        import time

        self._ttls[name] = time.time() + seconds
        return True

    def incr(self, name: str) -> int:
        val = self._data.get(name, "0")
        try:
            new_val = int(val) + 1
        except ValueError:
            new_val = 1
        self._data[name] = str(new_val)
        return new_val

    def ttl(self, name: str) -> int:
        import time

        if name not in self._ttls:
            return -1
        remaining = int(self._ttls[name] - time.time())
        return max(remaining, 0)

    def pipeline(self) -> Any:
        return self

    def execute(self) -> list[Any]:
        return [1, 60]

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        return [1, 59, 1]


class RedisService:
    _TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local requested = tonumber(ARGV[5])

local tokens = capacity
local updated_at = now
local bucket = redis.call("GET", key)

if bucket then
    local data = cjson.decode(bucket)
    tokens = tonumber(data["tokens"]) or capacity
    updated_at = tonumber(data["updated_at"]) or now
    local elapsed = math.max(0, now - updated_at)
    tokens = math.min(capacity, tokens + (elapsed * refill_rate))
end

local allowed = 0
if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
end

local reset_after = 0
if allowed == 1 then
    reset_after = math.ceil((capacity - tokens) / refill_rate)
else
    reset_after = math.ceil((requested - tokens) / refill_rate)
end

redis.call(
    "SETEX",
    key,
    ttl,
    cjson.encode({tokens = tokens, updated_at = now})
)

return {allowed, math.floor(tokens), reset_after}
"""

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

        if config.host == ":memory:":
            import logging

            logging.getLogger(__name__).warning(
                "Using in-memory mock Redis client (:memory:)."
            )
            return MemoryRedisClient()

        client = redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
        try:
            client.ping()
            return client
        except Exception as exc:
            if not config.allow_memory_fallback:
                raise RuntimeError(
                    f"Cannot connect to Redis at {config.host}:{config.port} "
                    f"({exc}). In-memory fallback is disabled outside "
                    "dev/local/test environments; fix Redis connectivity or set "
                    "SMARTSHOP_REDIS_ALLOW_MEMORY_FALLBACK=true for demos."
                ) from exc

            import logging

            logging.getLogger(__name__).warning(
                f"Could not connect to Redis at {config.host}:{config.port} ({exc}). "
                "Falling back to in-memory mock Redis client."
            )
            return MemoryRedisClient()

    def key(self, *parts: str) -> str:
        clean_parts = [self.config.key_prefix.strip(":")]
        clean_parts.extend(str(part).strip(":") for part in parts if str(part))
        return ":".join(clean_parts)

    def ping(self) -> bool:
        return bool(self.client.ping())

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

    def create_human_approval_request(
        self,
        session_id: str,
        message: str,
        history: Sequence[dict[str, Any]] | None = None,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> HumanApprovalRequest:
        request = HumanApprovalRequest(
            request_id=uuid.uuid4().hex,
            session_id=session_id,
            message=message,
            history=list(history or []),
            reason=reason,
            metadata=metadata or {},
        )
        payload = json.dumps(request.to_dict(), ensure_ascii=False)
        ttl = ttl_seconds or self.config.session_ttl_seconds
        self.client.setex(self.key("approval", request.request_id), ttl, payload)
        self.client.rpush(self.key("approval", "pending"), request.request_id)
        self.client.expire(self.key("approval", "pending"), ttl)
        return request

    def get_human_approval_request(
        self,
        request_id: str,
    ) -> HumanApprovalRequest | None:
        if not request_id.strip():
            raise ValueError("approval request_id must not be empty.")
        payload = self.client.get(self.key("approval", request_id))
        if not payload:
            return None
        return HumanApprovalRequest.from_payload(json.loads(payload))

    def list_pending_human_approvals(
        self,
        limit: int = 20,
    ) -> list[HumanApprovalRequest]:
        if limit < 1:
            raise ValueError("limit must be greater than zero.")
        ids = self.client.lrange(self.key("approval", "pending"), 0, limit - 1)
        requests: list[HumanApprovalRequest] = []
        for request_id in ids:
            request = self.get_human_approval_request(str(request_id))
            if request is not None and request.status == "pending":
                requests.append(request)
        return requests

    def resolve_human_approval_request(
        self,
        request_id: str,
        approved: bool,
        reviewer: str,
        note: str | None = None,
        ttl_seconds: int | None = None,
    ) -> HumanApprovalRequest:
        request = self.get_human_approval_request(request_id)
        if request is None:
            raise KeyError(f"approval request not found: {request_id}")

        resolved = HumanApprovalRequest(
            request_id=request.request_id,
            session_id=request.session_id,
            message=request.message,
            history=request.history,
            reason=request.reason,
            status="approved" if approved else "rejected",
            created_at=request.created_at,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            resolved_by=reviewer,
            note=note,
            metadata=request.metadata,
        )
        ttl = ttl_seconds or self.config.session_ttl_seconds
        self.client.setex(
            self.key("approval", request_id),
            ttl,
            json.dumps(resolved.to_dict(), ensure_ascii=False),
        )
        self.client.lrem(self.key("approval", "pending"), 0, request_id)
        return resolved

    def consume_human_approval_request(
        self,
        request_id: str,
        session_id: str,
        message: str,
        ttl_seconds: int | None = None,
    ) -> HumanApprovalRequest | None:
        """Redeem an approved request exactly once.

        Returns the approval only when it is approved, belongs to *session_id*,
        was raised for *message*, and has not been redeemed before.  Any other
        case returns None so the caller keeps treating the turn as unapproved.

        Single-use is what stops a reviewer's one-time "yes" from being replayed
        by the client on every later turn.
        """
        if not request_id.strip():
            raise ValueError("approval request_id must not be empty.")

        request = self.get_human_approval_request(request_id)
        if request is None:
            return None
        if request.status != "approved" or request.consumed_at is not None:
            return None
        if request.session_id != session_id or request.message != message:
            return None

        consumed = replace(
            request,
            consumed_at=datetime.now(timezone.utc).isoformat(),
        )
        ttl = ttl_seconds or self.config.session_ttl_seconds
        self.client.setex(
            self.key("approval", request_id),
            ttl,
            json.dumps(consumed.to_dict(), ensure_ascii=False),
        )
        return consumed

    def check_rate_limit(
        self,
        identity: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> RateLimitResult:
        algorithm = self.config.rate_limit_algorithm.lower().replace("-", "_")
        if algorithm == "fixed_window":
            return self.check_fixed_window_rate_limit(identity, limit, window_seconds)
        if algorithm in {"token_bucket", "tokenbucket"}:
            return self.check_token_bucket_rate_limit(identity, limit, window_seconds)
        raise ValueError(f"Unsupported rate limit algorithm: {algorithm}")

    def check_fixed_window_rate_limit(
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

    def check_token_bucket_rate_limit(
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

        key = self.key("rate", "token_bucket", identity)
        refill_rate = limit / window_seconds
        ttl_seconds = max(window_seconds * 2, 1)
        allowed, remaining, reset_after = self.client.eval(
            self._TOKEN_BUCKET_SCRIPT,
            1,
            key,
            limit,
            refill_rate,
            time.time(),
            ttl_seconds,
            1,
        )
        return RateLimitResult(
            allowed=bool(int(allowed)),
            limit=limit,
            remaining=max(int(remaining), 0),
            reset_after_seconds=max(int(reset_after), 0),
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
        "rate-check", help="Check Redis-backed API rate limit."
    )
    add_redis_options(rate_parser)
    rate_parser.add_argument("--identity", required=True)
    rate_parser.add_argument("--limit", type=int, default=DEFAULT_RATE_LIMIT)
    rate_parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_RATE_WINDOW_SECONDS,
    )
    rate_parser.add_argument(
        "--algorithm",
        choices=["token_bucket", "fixed_window"],
        default=DEFAULT_RATE_LIMIT_ALGORITHM,
    )

    return parser


def service_from_args(args: argparse.Namespace) -> RedisService:
    return RedisService(
        RedisConfig(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            key_prefix=args.key_prefix,
            rate_limit_algorithm=getattr(
                args, "algorithm", DEFAULT_RATE_LIMIT_ALGORITHM
            ),
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
