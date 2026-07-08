"""Phase 8–10 FastAPI API layer for the SmartShop platform.

Phase 10 additions:
- Prometheus metrics via prometheus-fastapi-instrumentator (``/metrics``)
- Custom business counters / histograms in ``src.monitoring``
- Langfuse LLM observability (``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePath
from typing import Any, AsyncIterator, Protocol

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.agent import ProductSearchTool, SmartShopAgent
from src.cache_service import RateLimitResult, RedisConfig, RedisService
from src.monitoring import MonitoringService
from src.streaming import KafkaClickEventProducer, StreamingConfig
from src.vector_store import ProductSearchFilters, VectorSearchService


DEFAULT_API_TITLE = "SmartShop AI API Layer"
DEFAULT_UPLOAD_DIR = "data/uploads/catalog"
DEFAULT_JWT_SECRET = "dev-smartshop-secret"
PUBLIC_PATH_PREFIXES = ("/docs", "/redoc", "/openapi.json")
PUBLIC_PATHS = {"/health"}

bearer_scheme = HTTPBearer(auto_error=False)


class CacheService(Protocol):
    config: RedisConfig

    def get_cached_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict] | None:
        """Return cached search hits."""

    def set_cached_search(
        self,
        query: str,
        results: list[dict],
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        ttl_seconds: int | None = None,
    ) -> str:
        """Cache search hits."""

    def append_session_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        max_messages: int = 50,
        ttl_seconds: int | None = None,
    ) -> Any:
        """Append one chat message to session storage."""

    def get_session_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return session chat history."""

    def check_rate_limit(
        self,
        identity: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> RateLimitResult:
        """Check whether a caller can continue."""


class SearchService(Protocol):
    def search_products(
        self,
        query: str,
        filters: ProductSearchFilters | None = None,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict]:
        """Search products."""


@dataclass(frozen=True)
class APIConfig:
    title: str = DEFAULT_API_TITLE
    upload_dir: str = DEFAULT_UPLOAD_DIR
    jwt_secret: str = DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    chat_history_limit: int = 20
    chat_chunk_delay_seconds: float = 0.0

    @classmethod
    def from_env(cls) -> "APIConfig":
        return cls(
            upload_dir=os.getenv("SMARTSHOP_UPLOAD_DIR", DEFAULT_UPLOAD_DIR),
            jwt_secret=os.getenv("SMARTSHOP_JWT_SECRET", DEFAULT_JWT_SECRET),
            access_token_expire_minutes=int(
                os.getenv("SMARTSHOP_TOKEN_EXPIRE_MINUTES", "60")
            ),
        )


class TokenPayload(BaseModel):
    sub: str
    exp: int | None = None
    scopes: list[str] = Field(default_factory=list)


class CurrentUser(BaseModel):
    user_id: str
    scopes: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    source: str
    results: list[dict[str, Any]]
    filters: dict[str, Any]
    top_k: int


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    approved_by_human: bool = False


class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    bytes_received: int
    status: str


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def create_access_token(
    subject: str,
    secret: str = DEFAULT_JWT_SECRET,
    expires_delta: timedelta | None = None,
    scopes: list[str] | None = None,
) -> str:
    """Create a small HS256 JWT for local development and tests."""

    expires_delta = expires_delta or timedelta(minutes=60)
    now = datetime.now(timezone.utc)
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "exp": int((now + expires_delta).timestamp()),
        "scopes": scopes or [],
    }
    signing_input = ".".join(
        [
            _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        ]
    )
    signature = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def verify_access_token(token: str, secret: str) -> TokenPayload:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        ) from exc

    signing_input = f"{header_b64}.{payload_b64}"
    expected_signature = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    supplied_signature = _b64url_decode(signature_b64)
    if not hmac.compare_digest(expected_signature, supplied_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        )

    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        ) from exc

    if header.get("alg") != "HS256":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unsupported token algorithm.",
        )

    token_payload = TokenPayload(**payload)
    if token_payload.exp and token_payload.exp < int(
        datetime.now(timezone.utc).timestamp()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token has expired.",
        )
    if not token_payload.sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token subject is missing.",
        )
    return token_payload


def _is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or any(
        path.startswith(prefix) for prefix in PUBLIC_PATH_PREFIXES
    )


def _request_identity(request: Request, user: CurrentUser | None = None) -> str:
    if user:
        return f"user:{user.user_id}"
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "anonymous"


def _filters_to_dict(filters: ProductSearchFilters) -> dict[str, Any]:
    payload = asdict(filters)
    return {key: value for key, value in payload.items() if value is not None}


def _safe_upload_filename(filename: str | None) -> str:
    clean_name = PurePath(filename or "catalog.csv").name
    clean_name = re.sub(r"[^A-Za-z0-9._-]+", "_", clean_name).strip("._")
    return clean_name or "catalog.csv"


def _save_upload(target_path: str, filename: str, contents: bytes) -> str:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(contents)
    manifest_path = target_path.with_suffix(target_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "filename": filename,
                "saved_path": str(target_path),
                "bytes_received": len(contents),
                "status": "queued_for_etl",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(target_path)


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(
    cache_service: CacheService | None = None,
    search_service: SearchService | None = None,
    agent: SmartShopAgent | None = None,
    config: APIConfig | None = None,
    monitoring: MonitoringService | None = None,
    kafka_producer: KafkaClickEventProducer | None = None,
) -> FastAPI:
    config = config or APIConfig.from_env()
    monitoring = monitoring or MonitoringService.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):  # noqa: ARG001
        # ---- startup ----
        yield
        # ---- shutdown ----
        monitoring.flush()

    app = FastAPI(title=config.title, lifespan=lifespan)
    app.state.config = config
    app.state.cache_service = cache_service
    app.state.search_service = search_service
    app.state.agent = agent
    app.state.monitoring = monitoring
    app.state.kafka_producer = kafka_producer

    # Phase 10: attach Prometheus /metrics endpoint
    monitoring.instrument(app)

    route_source = globals().get("app")
    if route_source is not None:
        for route in route_source.routes:
            if isinstance(route, APIRoute):
                app.router.routes.append(route)

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        if not _is_public_path(request.url.path):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()


def get_config(request: Request) -> APIConfig:
    return request.app.state.config


def get_cache_service(request: Request) -> CacheService:
    if request.app.state.cache_service is None:
        request.app.state.cache_service = RedisService()
    return request.app.state.cache_service


def get_search_service(request: Request) -> SearchService:
    if request.app.state.search_service is None:
        try:
            request.app.state.search_service = VectorSearchService()
            request.app.state.search_service.init_collection()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Search service is unavailable: {exc}",
            ) from exc
    return request.app.state.search_service


def get_agent(request: Request) -> SmartShopAgent:
    if request.app.state.agent is None:
        request.app.state.agent = SmartShopAgent(search_tool=ProductSearchTool())
    return request.app.state.agent


def get_kafka_producer(request: Request) -> KafkaClickEventProducer | None:
    """Lazily initialise the Kafka producer from environment variables.

    Returns None (silently) when Kafka is unavailable so the API keeps running
    in environments without a broker (e.g. unit tests, local dev without Docker).
    """
    if request.app.state.kafka_producer is None:
        cfg = StreamingConfig.from_env()
        try:
            request.app.state.kafka_producer = KafkaClickEventProducer(
                bootstrap_servers=cfg.bootstrap_servers,
                topic=cfg.topic,
                dead_letter_topic=cfg.dead_letter_topic,
                max_retries=cfg.producer_max_retries,
                retry_backoff_ms=cfg.producer_retry_backoff_ms,
            )
        except RuntimeError:
            # kafka-python not installed or broker unreachable at startup
            pass
    return request.app.state.kafka_producer


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    config: APIConfig = Depends(get_config),
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is required.",
        )
    payload = verify_access_token(credentials.credentials, config.jwt_secret)
    return CurrentUser(user_id=payload.sub, scopes=payload.scopes)


def enforce_rate_limit(
    request: Request,
    cache_service: CacheService = Depends(get_cache_service),
    user: CurrentUser = Depends(get_current_user),
) -> RateLimitResult:
    result = cache_service.check_rate_limit(_request_identity(request, user))
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(result.reset_after_seconds)},
        )
    return result


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
async def search(
    query: str,
    category: str | None = None,
    brand: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    top_k: int = 5,
    cache_service: CacheService = Depends(get_cache_service),
    search_service: SearchService = Depends(get_search_service),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> SearchResponse:
    if top_k < 1 or top_k > 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="top_k must be between 1 and 50.",
        )
    filters = ProductSearchFilters(
        category=category,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
    )
    filter_payload = _filters_to_dict(filters)

    try:
        cached_results = cache_service.get_cached_search(
            query,
            filters=filter_payload,
            top_k=top_k,
        )
        if cached_results is not None:
            return SearchResponse(
                query=query,
                source="cache",
                results=cached_results,
                filters=filter_payload,
                top_k=top_k,
            )

        results = search_service.search_products(query, filters=filters, top_k=top_k)
        cache_service.set_cached_search(
            query,
            results,
            filters=filter_payload,
            top_k=top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search service is unavailable: {exc}",
        ) from exc

    return SearchResponse(
        query=query,
        source="vector_store",
        results=results,
        filters=filter_payload,
        top_k=top_k,
    )


async def stream_agent_response(
    request: ChatRequest,
    agent: SmartShopAgent,
    cache_service: CacheService,
    config: APIConfig,
) -> AsyncIterator[str]:
    history: list[dict[str, Any]] = []
    if request.session_id:
        history = cache_service.get_session_messages(
            request.session_id,
            limit=config.chat_history_limit,
        )
        cache_service.append_session_message(
            request.session_id,
            "user",
            request.message,
        )

    response = agent.handle_message(
        request.message,
        history=history,
        approved_by_human=request.approved_by_human,
    )
    if request.session_id:
        cache_service.append_session_message(
            request.session_id,
            "assistant",
            response.content,
            metadata={
                "action": response.action,
                "requires_human_review": response.requires_human_review,
            },
        )

    yield _sse_event("start", {"action": response.action})
    for chunk in response.content.split():
        yield _sse_event("chunk", {"content": chunk})
        if config.chat_chunk_delay_seconds > 0:
            await asyncio.sleep(config.chat_chunk_delay_seconds)
    yield _sse_event(
        "done",
        {
            "content": response.content,
            "action": response.action,
            "requires_human_review": response.requires_human_review,
            "tool_outputs": response.tool_outputs,
        },
    )


@app.post("/chat")
async def chat(
    request: ChatRequest,
    agent: SmartShopAgent = Depends(get_agent),
    cache_service: CacheService = Depends(get_cache_service),
    config: APIConfig = Depends(get_config),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> StreamingResponse:
    return StreamingResponse(
        stream_agent_response(request, agent, cache_service, config),
        media_type="text/event-stream",
    )


@app.get("/chat/stream")
async def chat_stream(
    message: str,
    session_id: str | None = None,
    approved_by_human: bool = False,
    agent: SmartShopAgent = Depends(get_agent),
    cache_service: CacheService = Depends(get_cache_service),
    config: APIConfig = Depends(get_config),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> StreamingResponse:
    request = ChatRequest(
        message=message,
        session_id=session_id,
        approved_by_human=approved_by_human,
    )
    return StreamingResponse(
        stream_agent_response(request, agent, cache_service, config),
        media_type="text/event-stream",
    )


@app.post("/upload", response_model=UploadResponse)
@app.post("/catalog/upload", response_model=UploadResponse)
async def upload_catalog(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: APIConfig = Depends(get_config),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> UploadResponse:
    filename = _safe_upload_filename(file.filename)
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV catalog uploads are supported.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded catalog file is empty.",
        )

    target_dir = Path(config.upload_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    preview_path = target_dir / f"{timestamp}_{filename}"
    background_tasks.add_task(_save_upload, str(preview_path), filename, contents)
    return UploadResponse(
        filename=filename,
        saved_path=str(preview_path),
        bytes_received=len(contents),
        status="queued_for_etl",
    )


# ---------------------------------------------------------------------------
# Phase 5 × Phase 8: Clickstream event endpoint (Kafka producer integration)
# ---------------------------------------------------------------------------


class ClickEventRequest(BaseModel):
    user_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    session_id: str | None = None
    query: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClickEventResponse(BaseModel):
    status: str
    event: dict[str, Any]
    kafka_available: bool


@app.post("/events/click", response_model=ClickEventResponse)
async def record_click_event(
    request: ClickEventRequest,
    kafka: KafkaClickEventProducer | None = Depends(get_kafka_producer),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> ClickEventResponse:
    """Record a product click event and publish it to Kafka.

    * If a Kafka broker is reachable the event is published to the
      ``user-clicks`` topic (with retry + dead-letter on failure).
    * If Kafka is not available the endpoint still returns 200 with
      ``kafka_available: false`` so the frontend is never blocked.
    """
    from src.streaming import ClickEvent

    event_payload = ClickEvent(
        user_id=request.user_id,
        product_id=request.product_id,
        session_id=request.session_id,
        query=request.query,
        metadata=request.metadata,
    ).to_dict()

    if kafka is not None:
        try:
            published = kafka.log_click(
                user_id=request.user_id,
                product_id=request.product_id,
                session_id=request.session_id,
                query=request.query,
                metadata=request.metadata,
            )
            return ClickEventResponse(
                status="published", event=published, kafka_available=True
            )
        except Exception as exc:  # noqa: BLE001
            # Log but do not propagate – API remains responsive
            import logging as _logging
            _logging.getLogger(__name__).error("Kafka publish error: %s", exc)

    return ClickEventResponse(
        status="accepted_no_broker", event=event_payload, kafka_available=False
    )

