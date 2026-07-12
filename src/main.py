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
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.agent import ProductSearchTool, SmartShopAgent
from src.cache_service import (
    HumanApprovalRequest,
    RateLimitResult,
    RedisConfig,
    RedisService,
)
from src.logging_config import configure_logging, request_id_var
from src.model_service import RatingModelService
from src.monitoring import MonitoringService
from src.streaming import KafkaClickEventProducer, StreamingConfig
from src.vector_store import ProductSearchFilters, VectorSearchService

DEFAULT_API_TITLE = "SmartShop AI API Layer"
DEFAULT_UPLOAD_DIR = "data/uploads/catalog"
DEFAULT_ETL_OUTPUT_DIR = "data/processed/catalog_uploads"
DEFAULT_JWT_SECRET = "dev-smartshop-secret"
DEFAULT_CORS_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
PUBLIC_PATH_PREFIXES = ("/docs", "/redoc", "/openapi.json")
PUBLIC_PATHS = {"/", "/health", "/health/ready"}
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

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

    def create_human_approval_request(
        self,
        session_id: str,
        message: str,
        history: list[dict[str, Any]] | None = None,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> HumanApprovalRequest:
        """Queue a request for human support approval."""

    def get_human_approval_request(
        self,
        request_id: str,
    ) -> HumanApprovalRequest | None:
        """Return one human approval request."""

    def list_pending_human_approvals(
        self,
        limit: int = 20,
    ) -> list[HumanApprovalRequest]:
        """Return pending human approval requests."""

    def resolve_human_approval_request(
        self,
        request_id: str,
        approved: bool,
        reviewer: str,
        note: str | None = None,
        ttl_seconds: int | None = None,
    ) -> HumanApprovalRequest:
        """Mark a human approval request approved or rejected."""

    def ping(self) -> bool:
        """Return Redis connectivity status."""


class SearchService(Protocol):
    def search_products(
        self,
        query: str,
        filters: ProductSearchFilters | None = None,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[dict]:
        """Search products."""


class ETLJobRunner(Protocol):
    def run(self, input_path: str, manifest_path: str, config: "APIConfig") -> dict:
        """Run the catalog ETL job for an uploaded file."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _split_env_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class APIConfig:
    title: str = DEFAULT_API_TITLE
    environment: str = "dev"
    upload_dir: str = DEFAULT_UPLOAD_DIR
    etl_output_dir: str = DEFAULT_ETL_OUTPUT_DIR
    etl_output_format: str = "parquet"
    etl_spark_master: str | None = None
    etl_command: str | None = None
    etl_timeout_seconds: int = 30 * 60
    cors_allowed_origins: tuple[str, ...] = DEFAULT_CORS_ALLOWED_ORIGINS
    cors_allow_credentials: bool = True
    jwt_secret: str = DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_algorithms: tuple[str, ...] = ("RS256",)
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_jwks_url: str | None = None
    jwt_public_key: str | None = None
    access_token_expire_minutes: int = 60
    chat_history_limit: int = 20
    chat_chunk_delay_seconds: float = 0.0
    readiness_check_qdrant: bool = False

    @classmethod
    def from_env(cls) -> "APIConfig":
        cors_origins = _split_env_list(os.getenv("SMARTSHOP_CORS_ORIGINS"))
        jwt_algorithms = _split_env_list(os.getenv("SMARTSHOP_JWT_ALGORITHMS"))
        environment = os.getenv("SMARTSHOP_ENV", "dev")
        return cls(
            environment=environment,
            upload_dir=os.getenv("SMARTSHOP_UPLOAD_DIR", DEFAULT_UPLOAD_DIR),
            etl_output_dir=os.getenv(
                "SMARTSHOP_ETL_OUTPUT_DIR", DEFAULT_ETL_OUTPUT_DIR
            ),
            etl_output_format=os.getenv("SMARTSHOP_ETL_OUTPUT_FORMAT", "parquet"),
            etl_spark_master=os.getenv("SMARTSHOP_ETL_SPARK_MASTER") or None,
            etl_command=os.getenv("SMARTSHOP_ETL_COMMAND") or None,
            etl_timeout_seconds=int(os.getenv("SMARTSHOP_ETL_TIMEOUT_SECONDS", "1800")),
            cors_allowed_origins=cors_origins or DEFAULT_CORS_ALLOWED_ORIGINS,
            cors_allow_credentials=_env_bool("SMARTSHOP_CORS_ALLOW_CREDENTIALS", True),
            jwt_secret=os.getenv("SMARTSHOP_JWT_SECRET", DEFAULT_JWT_SECRET),
            jwt_algorithms=jwt_algorithms or ("RS256",),
            jwt_issuer=os.getenv("SMARTSHOP_JWT_ISSUER") or None,
            jwt_audience=os.getenv("SMARTSHOP_JWT_AUDIENCE") or None,
            jwt_jwks_url=os.getenv("SMARTSHOP_JWKS_URL") or None,
            jwt_public_key=(
                os.getenv("SMARTSHOP_JWT_PUBLIC_KEY", "").replace("\\n", "\n") or None
            ),
            access_token_expire_minutes=int(
                os.getenv("SMARTSHOP_TOKEN_EXPIRE_MINUTES", "60")
            ),
            readiness_check_qdrant=_env_bool(
                "SMARTSHOP_READINESS_CHECK_QDRANT",
                environment.strip().lower() not in {"dev", "local", "test"},
            ),
        )


class TokenPayload(BaseModel):
    sub: str
    exp: int | None = None
    iss: str | None = None
    aud: str | list[str] | None = None
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
    manifest_path: str | None = None
    bytes_received: int
    status: str
    etl_status: str = "queued"


class HumanApprovalRequestPayload(BaseModel):
    request_id: str
    session_id: str
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""
    status: str
    created_at: str
    resolved_at: str | None = None
    resolved_by: str | None = None
    note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanApprovalListResponse(BaseModel):
    requests: list[HumanApprovalRequestPayload]


class ResolveHumanApprovalRequest(BaseModel):
    approved: bool
    reviewer: str | None = None
    note: str | None = None


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


def _normalize_token_payload(payload: dict[str, Any]) -> TokenPayload:
    scopes = payload.get("scopes", [])
    if not scopes and payload.get("scope"):
        scopes = str(payload["scope"]).split()
    return TokenPayload(**{**payload, "scopes": scopes})


def _verify_hs256_dev_access_token(token: str, secret: str) -> TokenPayload:
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

    token_payload = _normalize_token_payload(payload)
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


def _verify_issuer_access_token(token: str, config: APIConfig) -> TokenPayload:
    try:
        import jwt
    except ImportError as exc:
        raise RuntimeError(
            "PyJWT with crypto support is required for issuer/JWKS token validation. "
            "Install `pyjwt[crypto]` or use the dev HS256 fallback locally."
        ) from exc

    algorithms = list(config.jwt_algorithms or ("RS256",))
    try:
        if config.jwt_public_key:
            signing_key = config.jwt_public_key
        elif config.jwt_jwks_url:
            signing_key = (
                jwt.PyJWKClient(config.jwt_jwks_url).get_signing_key_from_jwt(token).key
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT issuer validation is not configured.",
            )

        payload = jwt.decode(
            token,
            signing_key,
            algorithms=algorithms,
            audience=config.jwt_audience,
            issuer=config.jwt_issuer,
            options={"verify_aud": config.jwt_audience is not None},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        ) from exc
    return _normalize_token_payload(payload)


def verify_access_token(token: str, config_or_secret: APIConfig | str) -> TokenPayload:
    if isinstance(config_or_secret, APIConfig):
        config = config_or_secret
    else:
        config = APIConfig(jwt_secret=config_or_secret)

    if config.jwt_jwks_url or config.jwt_public_key:
        return _verify_issuer_access_token(token, config)

    if (
        config.environment.strip().lower() not in {"dev", "local", "test"}
        and config.jwt_secret == DEFAULT_JWT_SECRET
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT issuer/JWKS validation must be configured outside dev.",
        )
    return _verify_hs256_dev_access_token(token, config.jwt_secret)


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


def _manifest_path_for(target_path: Path) -> Path:
    return target_path.with_suffix(target_path.suffix + ".manifest.json")


def _write_upload_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_upload_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _format_etl_command(command: str, values: dict[str, str]) -> list[str]:
    formatted = command.format(**values)
    return shlex.split(formatted, posix=os.name != "nt")


def _default_etl_command(
    input_path: str, output_path: str, config: APIConfig
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "jobs.spark_etl",
        "--input-products",
        input_path,
        "--output",
        output_path,
        "--output-format",
        config.etl_output_format,
    ]
    if config.etl_spark_master:
        command.extend(["--master", config.etl_spark_master])
    return command


class SubprocessETLJobRunner:
    def run(self, input_path: str, manifest_path: str, config: APIConfig) -> dict:
        manifest_file = Path(manifest_path)
        manifest = _load_upload_manifest(manifest_file)
        output_path = str(
            Path(config.etl_output_dir)
            / f"{Path(input_path).stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        values = {
            "input_path": input_path,
            "output_path": output_path,
            "output_format": config.etl_output_format,
            "manifest_path": manifest_path,
        }
        started_at = datetime.now(timezone.utc).isoformat()

        if not config.etl_command and importlib.util.find_spec("pyspark") is None:
            result = {
                "status": "failed",
                "command": None,
                "returncode": None,
                "output_path": output_path,
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": (
                    "pyspark is not installed in this runtime, so the default "
                    "ETL command cannot run. Set SMARTSHOP_ETL_COMMAND to a "
                    "worker/queue command, or deploy the API with the "
                    "full-runtime image target that bundles Spark."
                ),
            }
            manifest.update({"status": "etl_failed", "etl": result})
            _write_upload_manifest(manifest_file, manifest)
            return result

        command = (
            _format_etl_command(config.etl_command, values)
            if config.etl_command
            else _default_etl_command(input_path, output_path, config)
        )

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=config.etl_timeout_seconds,
            )
            status_label = "completed" if completed.returncode == 0 else "failed"
            result = {
                "status": status_label,
                "command": command,
                "returncode": completed.returncode,
                "output_path": output_path,
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                "status": "failed",
                "command": command,
                "returncode": None,
                "output_path": output_path,
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }

        manifest.update({"status": f"etl_{result['status']}", "etl": result})
        _write_upload_manifest(manifest_file, manifest)
        return result


def _save_upload(target_path: str, filename: str, contents: bytes) -> str:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(contents)
    manifest_path = _manifest_path_for(target_path)
    _write_upload_manifest(
        manifest_path,
        {
            "filename": filename,
            "saved_path": str(target_path),
            "bytes_received": len(contents),
            "status": "queued_for_etl",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return str(target_path)


def _save_upload_and_run_etl(
    target_path: str,
    filename: str,
    contents: bytes,
    etl_runner: ETLJobRunner,
    config: APIConfig,
) -> None:
    saved_path = _save_upload(target_path, filename, contents)
    manifest_path = _manifest_path_for(Path(saved_path))
    manifest = _load_upload_manifest(manifest_path)
    manifest["status"] = "etl_running"
    manifest["etl"] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_upload_manifest(manifest_path, manifest)
    etl_runner.run(saved_path, str(manifest_path), config)


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _approval_payload(
    request: HumanApprovalRequest,
) -> HumanApprovalRequestPayload:
    return HumanApprovalRequestPayload(**request.to_dict())


def create_app(
    cache_service: CacheService | None = None,
    search_service: SearchService | None = None,
    agent: SmartShopAgent | None = None,
    config: APIConfig | None = None,
    monitoring: MonitoringService | None = None,
    kafka_producer: KafkaClickEventProducer | None = None,
    etl_runner: ETLJobRunner | None = None,
    rating_model: RatingModelService | None = None,
) -> FastAPI:
    config = config or APIConfig.from_env()
    monitoring = monitoring or MonitoringService.from_env()
    configure_logging()

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
    app.state.etl_runner = etl_runner or SubprocessETLJobRunner()
    app.state.rating_model = rating_model

    if config.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_allowed_origins),
            allow_credentials=config.cors_allow_credentials,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

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

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or os.urandom(8).hex()
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
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


def get_monitoring(request: Request) -> MonitoringService:
    return request.app.state.monitoring


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


def get_etl_runner(request: Request) -> ETLJobRunner:
    return request.app.state.etl_runner


def get_rating_model(request: Request) -> RatingModelService:
    if request.app.state.rating_model is None:
        request.app.state.rating_model = RatingModelService.from_env()
    return request.app.state.rating_model


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    config: APIConfig = Depends(get_config),
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is required.",
        )
    try:
        payload = verify_access_token(credentials.credentials, config)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return CurrentUser(user_id=payload.sub, scopes=payload.scopes)


ADMIN_SCOPE = "admin"


def require_scopes(*required_scopes: str):
    """Dependency factory enforcing token scopes (RBAC).

    A token passes when it carries every required scope or the ``admin``
    scope. Regular customer tokens without scopes get 403, so endpoints such
    as approvals resolution stay reviewer-only.
    """

    def checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        token_scopes = set(user.scopes)
        if ADMIN_SCOPE in token_scopes:
            return user
        missing = [scope for scope in required_scopes if scope not in token_scopes]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Token is missing required scopes: {', '.join(missing)}.",
            )
        return user

    return checker


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def web_console() -> HTMLResponse:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Web console asset is missing.",
        )
    return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))


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


def _qdrant_ready(request: Request) -> bool:
    """Check Qdrant connectivity without triggering the in-memory fallback."""
    search_service = request.app.state.search_service
    service_ping = getattr(search_service, "ping", None)
    if search_service is not None and callable(service_ping):
        try:
            return bool(service_ping())
        except Exception:  # noqa: BLE001
            return False

    try:
        from qdrant_client import QdrantClient

        from src.vector_store import VectorStoreConfig

        vector_config = VectorStoreConfig.from_env()
        client = QdrantClient(
            host=vector_config.qdrant_host,
            port=vector_config.qdrant_port,
            timeout=2.0,
        )
        client.get_collections()
        return True
    except Exception:  # noqa: BLE001
        return False


@app.get("/health/ready")
async def readiness(request: Request) -> JSONResponse:
    config: APIConfig = request.app.state.config
    try:
        cache_service = get_cache_service(request)
        redis_ok = cache_service.ping()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "redis": "unavailable",
                "detail": str(exc),
            },
        )

    if not redis_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "redis": "unavailable"},
        )

    if config.readiness_check_qdrant:
        qdrant_ok = await asyncio.to_thread(_qdrant_ready, request)
        if not qdrant_ok:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "redis": "ok",
                    "qdrant": "unavailable",
                },
            )
        return JSONResponse(content={"status": "ok", "redis": "ok", "qdrant": "ok"})

    return JSONResponse(content={"status": "ok", "redis": "ok"})


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
        # Redis/Qdrant clients are synchronous; run them in the threadpool so
        # slow calls do not stall the event loop for other requests.
        cached_results = await asyncio.to_thread(
            cache_service.get_cached_search,
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

        results = await asyncio.to_thread(
            search_service.search_products, query, filters=filters, top_k=top_k
        )
        await asyncio.to_thread(
            cache_service.set_cached_search,
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
    monitoring: MonitoringService,
) -> AsyncIterator[str]:
    history: list[dict[str, Any]] = []
    if request.session_id:
        # Sync Redis calls: keep them off the event loop.
        history = await asyncio.to_thread(
            cache_service.get_session_messages,
            request.session_id,
            limit=config.chat_history_limit,
        )
        await asyncio.to_thread(
            cache_service.append_session_message,
            request.session_id,
            "user",
            request.message,
        )

    monitoring.record_chat()
    # The agent may call an LLM synchronously (up to ~30s); running it on the
    # threadpool keeps other requests on this worker responsive.
    response = await asyncio.to_thread(
        agent.handle_message,
        request.message,
        history=history,
        approved_by_human=request.approved_by_human,
    )
    monitoring.record_agent_decision(response.action)
    for event in response.trace_events:
        if event.event == "tool_call" and event.tool:
            monitoring.record_agent_tool_call(event.tool, event.status)
        elif event.event == "llm_call":
            monitoring.record_llm_call(
                name="agent-llm-routing",
                model=event.metadata.get("model"),
                latency_seconds=event.latency_seconds,
                prompt_tokens=int(event.metadata.get("prompt_tokens") or 0),
                completion_tokens=int(event.metadata.get("completion_tokens") or 0),
                session_id=request.session_id,
                input_text=event.metadata.get("input_text"),
                output_text=response.content,
                metadata={"action": response.action},
            )

    approval_request_id: str | None = None
    if response.requires_human_review:
        approval = await asyncio.to_thread(
            cache_service.create_human_approval_request,
            session_id=request.session_id or "ad-hoc",
            message=request.message,
            history=history,
            reason=response.reason,
            metadata={"action": response.action},
        )
        approval_request_id = approval.request_id

    if request.session_id:
        await asyncio.to_thread(
            cache_service.append_session_message,
            request.session_id,
            "assistant",
            response.content,
            metadata={
                "action": response.action,
                "requires_human_review": response.requires_human_review,
                "approval_request_id": approval_request_id,
            },
        )

    yield _sse_event(
        "start",
        {
            "action": response.action,
            "approval_request_id": approval_request_id,
        },
    )
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
            "approval_request_id": approval_request_id,
            "tool_outputs": response.tool_outputs,
        },
    )


@app.post("/chat")
async def chat(
    request: ChatRequest,
    agent: SmartShopAgent = Depends(get_agent),
    cache_service: CacheService = Depends(get_cache_service),
    config: APIConfig = Depends(get_config),
    monitoring: MonitoringService = Depends(get_monitoring),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> StreamingResponse:
    return StreamingResponse(
        stream_agent_response(request, agent, cache_service, config, monitoring),
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
    monitoring: MonitoringService = Depends(get_monitoring),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> StreamingResponse:
    request = ChatRequest(
        message=message,
        session_id=session_id,
        approved_by_human=approved_by_human,
    )
    return StreamingResponse(
        stream_agent_response(request, agent, cache_service, config, monitoring),
        media_type="text/event-stream",
    )


@app.get("/approvals/pending", response_model=HumanApprovalListResponse)
async def list_pending_human_approvals(
    limit: int = 20,
    cache_service: CacheService = Depends(get_cache_service),
    _reviewer: CurrentUser = Depends(require_scopes("approvals:read")),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> HumanApprovalListResponse:
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="limit must be between 1 and 100.",
        )
    requests = await asyncio.to_thread(
        cache_service.list_pending_human_approvals, limit=limit
    )
    return HumanApprovalListResponse(
        requests=[_approval_payload(request) for request in requests]
    )


@app.get(
    "/approvals/{request_id}",
    response_model=HumanApprovalRequestPayload,
)
async def get_human_approval(
    request_id: str,
    cache_service: CacheService = Depends(get_cache_service),
    _reviewer: CurrentUser = Depends(require_scopes("approvals:read")),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> HumanApprovalRequestPayload:
    approval = await asyncio.to_thread(
        cache_service.get_human_approval_request, request_id
    )
    if approval is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval request not found.",
        )
    return _approval_payload(approval)


@app.post(
    "/approvals/{request_id}/resolve",
    response_model=HumanApprovalRequestPayload,
)
async def resolve_human_approval(
    request_id: str,
    payload: ResolveHumanApprovalRequest,
    cache_service: CacheService = Depends(get_cache_service),
    current_user: CurrentUser = Depends(require_scopes("approvals:write")),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> HumanApprovalRequestPayload:
    reviewer = payload.reviewer or current_user.user_id
    try:
        approval = await asyncio.to_thread(
            cache_service.resolve_human_approval_request,
            request_id=request_id,
            approved=payload.approved,
            reviewer=reviewer,
            note=payload.note,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval request not found.",
        ) from exc
    return _approval_payload(approval)


@app.post("/upload", response_model=UploadResponse)
@app.post("/catalog/upload", response_model=UploadResponse)
async def upload_catalog(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: APIConfig = Depends(get_config),
    etl_runner: ETLJobRunner = Depends(get_etl_runner),
    monitoring: MonitoringService = Depends(get_monitoring),
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
    manifest_path = _manifest_path_for(preview_path)
    background_tasks.add_task(
        _save_upload_and_run_etl,
        str(preview_path),
        filename,
        contents,
        etl_runner,
        config,
    )
    monitoring.record_upload(success=True)
    return UploadResponse(
        filename=filename,
        saved_path=str(preview_path),
        manifest_path=str(manifest_path),
        bytes_received=len(contents),
        status="queued_for_etl",
        etl_status="queued",
    )


class UploadStatusSummary(BaseModel):
    upload_name: str
    filename: str | None = None
    status: str = "unknown"
    etl_status: str | None = None
    created_at: str | None = None


class UploadStatusListResponse(BaseModel):
    uploads: list[UploadStatusSummary]


def _manifest_to_summary(
    manifest_path: Path, manifest: dict[str, Any]
) -> UploadStatusSummary:
    upload_name = manifest_path.name.removesuffix(".manifest.json")
    etl_info = manifest.get("etl") or {}
    return UploadStatusSummary(
        upload_name=upload_name,
        filename=manifest.get("filename"),
        status=str(manifest.get("status", "unknown")),
        etl_status=etl_info.get("status"),
        created_at=manifest.get("created_at"),
    )


@app.get("/catalog/uploads", response_model=UploadStatusListResponse)
async def list_catalog_uploads(
    limit: int = 20,
    config: APIConfig = Depends(get_config),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> UploadStatusListResponse:
    """Return recent catalog uploads with their ETL status from manifests."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="limit must be between 1 and 100.",
        )
    upload_dir = Path(config.upload_dir)
    if not upload_dir.exists():
        return UploadStatusListResponse(uploads=[])

    manifests = sorted(
        upload_dir.glob("*.manifest.json"),
        key=lambda path: path.name,
        reverse=True,
    )[:limit]
    uploads = []
    for manifest_path in manifests:
        try:
            manifest = _load_upload_manifest(manifest_path)
        except (json.JSONDecodeError, OSError):
            manifest = {}
        uploads.append(_manifest_to_summary(manifest_path, manifest))
    return UploadStatusListResponse(uploads=uploads)


@app.get("/catalog/uploads/{upload_name}")
async def get_catalog_upload_status(
    upload_name: str,
    config: APIConfig = Depends(get_config),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> dict[str, Any]:
    """Return the full manifest (including ETL result) for one upload."""
    clean_name = PurePath(upload_name).name
    if clean_name != upload_name or not clean_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload name.",
        )
    manifest_path = Path(config.upload_dir) / f"{clean_name}.manifest.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload not found.",
        )
    return _load_upload_manifest(manifest_path)


# ---------------------------------------------------------------------------
# Phase 3 × Phase 8: MLflow registered model serving
# ---------------------------------------------------------------------------


class RatingPredictionProduct(BaseModel):
    product_id: str | None = None
    title: str = ""
    description: str = ""
    brand: str = ""
    category: str = ""
    price_tier: str = ""


class RatingPredictionRequest(BaseModel):
    products: list[RatingPredictionProduct] = Field(min_length=1, max_length=100)


class RatingPredictionResponse(BaseModel):
    model_uri: str
    predictions: list[dict[str, Any]]


@app.post("/predict/rating", response_model=RatingPredictionResponse)
async def predict_rating(
    request: RatingPredictionRequest,
    rating_model: RatingModelService = Depends(get_rating_model),
    _rate_limit: RateLimitResult = Depends(enforce_rate_limit),
) -> RatingPredictionResponse:
    """Score products with the MLflow champion rating classifier."""
    products = [product.model_dump() for product in request.products]
    try:
        predictions = await asyncio.to_thread(rating_model.predict, products)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return RatingPredictionResponse(
        model_uri=rating_model.config.model_uri,
        predictions=predictions,
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
            # log_click retries with backoff sleeps; keep it off the event loop.
            published = await asyncio.to_thread(
                kafka.log_click,
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
