"""Phase 10: Monitoring & Observability module for SmartShop AI Platform.

Provides:
- Prometheus metrics instrumentation via ``prometheus-fastapi-instrumentator``.
- Custom business metrics (search hits/misses, chat requests, token usage).
- Langfuse LLM tracing decorator utilities.
- A lightweight ``MonitoringService`` that wraps both backends so unit tests
  can inject fakes without touching global state.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse LLM observability backend."""

    public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
    host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    enabled: bool = field(
        default_factory=lambda: bool(
            os.getenv("LANGFUSE_PUBLIC_KEY", "")
            and os.getenv("LANGFUSE_SECRET_KEY", "")
        )
    )


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus metrics endpoint."""

    metrics_endpoint: str = "/metrics"
    should_group_status_codes: bool = True
    should_group_untemplated: bool = True
    excluded_handlers: list[str] = field(
        default_factory=lambda: ["/metrics", "/health"]
    )


@dataclass
class MonitoringConfig:
    """Top-level monitoring configuration."""

    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Build config entirely from environment variables."""
        return cls(
            prometheus=PrometheusConfig(
                metrics_endpoint=os.getenv("PROMETHEUS_METRICS_ENDPOINT", "/metrics"),
            ),
            langfuse=LangfuseConfig(),
        )


# ---------------------------------------------------------------------------
# Custom Prometheus metrics (defined lazily to avoid import errors when the
# ``prometheus_client`` package is not installed in the test environment)
# ---------------------------------------------------------------------------

# Private registry so our custom metrics never collide with the default global
# registry used by prometheus-fastapi-instrumentator.  Using a dedicated
# registry also means metric names can be re-registered safely across test
# sessions that reload the module.
_CUSTOM_REGISTRY: Any = None


def _get_or_create_registry() -> Any:
    """Return the module-level private CollectorRegistry, creating it once."""
    global _CUSTOM_REGISTRY
    if _CUSTOM_REGISTRY is None:
        try:
            from prometheus_client import CollectorRegistry  # type: ignore

            _CUSTOM_REGISTRY = CollectorRegistry()
        except ImportError:
            pass
    return _CUSTOM_REGISTRY


def _build_custom_metrics() -> dict[str, Any]:
    """
    Create and register custom Prometheus Counters / Histograms.

    Uses a private :class:`CollectorRegistry` so these metrics are isolated
    from the default global registry and can be safely instantiated multiple
    times within the same test session.

    Returns an empty dict if ``prometheus_client`` is not installed, so the
    rest of the module degrades gracefully.
    """
    registry = _get_or_create_registry()
    if registry is None:
        logger.warning("prometheus_client not installed; custom metrics disabled.")
        return {}

    try:
        from prometheus_client import Counter, Histogram  # type: ignore

        metrics: dict[str, Any] = {
            # Search endpoint
            "search_requests_total": Counter(
                "smartshop_search_requests_total",
                "Total number of /search requests.",
                ["source"],  # labels: cache | vector_store
                registry=registry,
            ),
            "search_latency_seconds": Histogram(
                "smartshop_search_latency_seconds",
                "Search request latency in seconds.",
                buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
                registry=registry,
            ),
            # Chat / LLM
            "chat_requests_total": Counter(
                "smartshop_chat_requests_total",
                "Total number of /chat requests.",
                registry=registry,
            ),
            "llm_tokens_total": Counter(
                "smartshop_llm_tokens_total",
                "Total LLM tokens consumed.",
                ["type"],  # labels: prompt | completion
                registry=registry,
            ),
            "llm_latency_seconds": Histogram(
                "smartshop_llm_latency_seconds",
                "LLM call latency in seconds.",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
                registry=registry,
            ),
            "agent_decisions_total": Counter(
                "smartshop_agent_decisions_total",
                "Total agent decisions by action and routing mode.",
                # `mode` distinguishes LLM routing from the keyword fallback.
                # Without it a dead API key looks identical to healthy traffic,
                # because the agent degrades to keyword matching silently.
                ["action", "mode"],
                registry=registry,
            ),
            "agent_tool_calls_total": Counter(
                "smartshop_agent_tool_calls_total",
                "Total agent tool calls by tool and status.",
                ["tool", "status"],
                registry=registry,
            ),
            # Upload
            "catalog_uploads_total": Counter(
                "smartshop_catalog_uploads_total",
                "Total catalog file uploads.",
                ["status"],  # labels: success | error
                registry=registry,
            ),
            # ----------------------------------------------------------------
            # Phase 5 × Phase 10: Kafka / clickstream event throughput
            # ----------------------------------------------------------------
            "click_events_total": Counter(
                "smartshop_click_events_total",
                "Total clickstream events published to Kafka.",
                ["status"],  # labels: published | accepted_no_broker | error
                registry=registry,
            ),
            "click_events_dlq_total": Counter(
                "smartshop_click_events_dlq_total",
                "Total events forwarded to the dead-letter topic.",
                registry=registry,
            ),
            "click_event_latency_seconds": Histogram(
                "smartshop_click_event_latency_seconds",
                "End-to-end latency for /events/click handler in seconds.",
                buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
                registry=registry,
            ),
        }
        return metrics
    except (ImportError, ValueError) as exc:
        logger.warning("Custom metrics could not be registered: %s", exc)
        return {}


# Module-level metrics dict (populated on first call to ``get_metrics()``)
_METRICS: dict[str, Any] | None = None


def get_metrics() -> dict[str, Any]:
    """Return the shared custom metrics registry (lazily initialised)."""
    global _METRICS
    if _METRICS is None:
        _METRICS = _build_custom_metrics()
    return _METRICS


# ---------------------------------------------------------------------------
# Prometheus instrumentation helpers
# ---------------------------------------------------------------------------


def instrument_app(app: Any, config: PrometheusConfig | None = None) -> Any:
    """
    Attach ``prometheus-fastapi-instrumentator`` to *app* and expose
    the ``/metrics`` endpoint.

    Returns the instrumented app (same object) for chaining.
    Falls back gracefully if the package is not installed.
    """
    if config is None:
        config = PrometheusConfig()

    try:
        from prometheus_client import (  # type: ignore
            CONTENT_TYPE_LATEST,
            REGISTRY,
            generate_latest,
        )
        from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
        from starlette.responses import Response  # type: ignore

        Instrumentator(
            should_group_status_codes=config.should_group_status_codes,
            should_group_untemplated=config.should_group_untemplated,
            excluded_handlers=config.excluded_handlers,
        ).instrument(app)

        # Expose the endpoint ourselves so the output combines the default
        # registry (HTTP metrics from the instrumentator) with the private
        # registry holding the smartshop_* business metrics. The stock
        # ``.expose()`` helper only serves the default registry, which left
        # the business panels in Grafana without data.
        @app.get(config.metrics_endpoint, include_in_schema=False)
        async def metrics() -> Response:
            output = generate_latest(REGISTRY)
            custom_registry = _get_or_create_registry()
            if custom_registry is not None:
                output += generate_latest(custom_registry)
            return Response(content=output, media_type=CONTENT_TYPE_LATEST)

        logger.info(
            "Prometheus instrumentator attached; metrics at %s",
            config.metrics_endpoint,
        )
    except ImportError:
        logger.warning(
            "prometheus-fastapi-instrumentator not installed; "
            "Prometheus metrics endpoint will not be available."
        )

    return app


# ---------------------------------------------------------------------------
# Context manager for timing + recording metrics
# ---------------------------------------------------------------------------


@contextmanager
def track_search(source: str = "vector_store") -> Generator[None, None, None]:
    """Context manager that records search latency and increments counter.

    Usage::

        with track_search(source="cache"):
            results = cache.get(...)
    """
    metrics = get_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if "search_requests_total" in metrics:
            metrics["search_requests_total"].labels(source=source).inc()
        if "search_latency_seconds" in metrics:
            metrics["search_latency_seconds"].observe(elapsed)


@contextmanager
def track_llm_call(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Generator[None, None, None]:
    """Context manager that records LLM latency and token usage.

    Usage::

        with track_llm_call(prompt_tokens=len(prompt)) as tracker:
            response = llm.invoke(prompt)
    """
    metrics = get_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if "llm_latency_seconds" in metrics:
            metrics["llm_latency_seconds"].observe(elapsed)
        if prompt_tokens and "llm_tokens_total" in metrics:
            metrics["llm_tokens_total"].labels(type="prompt").inc(prompt_tokens)
        if completion_tokens and "llm_tokens_total" in metrics:
            metrics["llm_tokens_total"].labels(type="completion").inc(completion_tokens)


# ---------------------------------------------------------------------------
# Langfuse integration
# ---------------------------------------------------------------------------


class LangfuseTracer:
    """
    Thin wrapper around Langfuse SDK.

    Provides a ``@trace`` decorator that automatically creates a Langfuse
    trace for any async function that calls an LLM.  When Langfuse is not
    configured or not installed the decorator is a transparent no-op.
    """

    def __init__(self, config: LangfuseConfig | None = None) -> None:
        self.config = config or LangfuseConfig()
        self._client: Any = None
        self._enabled = self.config.enabled
        self._last_error: str | None = None
        self._init_client()

    def _init_client(self) -> None:
        if not self._enabled:
            return
        try:
            from langfuse import Langfuse  # type: ignore

            self._client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.host,
            )
            # Fail loudly on an incompatible SDK rather than accepting every
            # call and dropping it. v2/v3 lack start_observation, so without
            # this check the tracer would report itself enabled forever while
            # emitting nothing.
            if not hasattr(self._client, "start_observation"):
                raise RuntimeError(
                    "Installed langfuse SDK has no start_observation(); this "
                    "module targets langfuse>=4,<5. Check your pinned version."
                )
            logger.info("Langfuse client initialised (host=%s).", self.config.host)
        except ImportError:
            logger.warning("langfuse package not installed; LLM tracing disabled.")
            self._enabled = False
        except Exception as exc:  # noqa: BLE001
            logger.error("Langfuse client init failed; LLM tracing disabled: %s", exc)
            self._last_error = str(exc)
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None

    def trace(
        self,
        name: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Callable:
        """Decorator factory that wraps an async function with Langfuse tracing.

        If Langfuse is disabled the original function is returned unchanged.

        Usage::

            @langfuse_tracer.trace(name="chat-handler", tags=["production"])
            async def call_llm(prompt: str) -> str:
                ...
        """

        def decorator(fn: Callable) -> Callable:
            if not self.is_enabled:
                return fn

            import functools

            try:
                from langfuse import observe, propagate_attributes  # type: ignore

                @observe(name=name or fn.__name__)
                @functools.wraps(fn)
                async def wrapper(*args: Any, **kwargs: Any) -> Any:
                    # v4 sets trace-level attributes through a context manager
                    # instead of v2's langfuse_context.update_current_trace.
                    with propagate_attributes(user_id=user_id, tags=tags or []):
                        return await fn(*args, **kwargs)

                return wrapper
            except ImportError:
                return fn

        return decorator

    def log_generation(
        self,
        *,
        name: str,
        model: str | None = None,
        latency_seconds: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        session_id: str | None = None,
        user_id: str | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Send one LLM generation event to Langfuse.

        Returns True when the event was handed to the SDK, False when tracing
        is disabled or the SDK call failed (the API keeps running either way).

        Uses the v4 observation API: ``start_observation(as_type="generation")``
        returns a span that must be ended.  v2's ``client.generation(...)`` and
        ``langfuse.decorators`` were removed in v3, so this module pins
        ``langfuse>=4,<5`` -- an unpinned upgrade previously left tracing
        reporting itself as enabled while silently emitting nothing.
        """
        if not self.is_enabled:
            return False
        try:
            from langfuse import propagate_attributes  # type: ignore

            usage = {}
            if prompt_tokens:
                usage["input"] = prompt_tokens
            if completion_tokens:
                usage["output"] = completion_tokens
            if usage:
                usage["total"] = prompt_tokens + completion_tokens

            with propagate_attributes(session_id=session_id, user_id=user_id):
                generation = self._client.start_observation(
                    name=name,
                    as_type="generation",
                    model=model,
                    input=input_text,
                    output=output_text,
                    usage_details=usage or None,
                    metadata={
                        "latency_seconds": latency_seconds,
                        **(metadata or {}),
                    },
                )
                generation.end()
            self._last_error = None
            return True
        except Exception as exc:  # noqa: BLE001
            # Record the failure so /health and `python -m src.monitoring status`
            # can surface "configured but broken", instead of the old behaviour
            # where a dead SDK looked identical to a healthy one.
            self._last_error = str(exc)
            logger.warning("Langfuse generation logging failed: %s", exc)
            return False

    @property
    def last_error(self) -> str | None:
        """Last Langfuse SDK error, or None if the last call succeeded."""
        return self._last_error

    def flush(self) -> None:
        """Flush pending Langfuse events (call on app shutdown)."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse flush failed: %s", exc)


# ---------------------------------------------------------------------------
# MonitoringService – single entry point used by the FastAPI app
# ---------------------------------------------------------------------------


class MonitoringService:
    """Aggregates Prometheus metrics and Langfuse tracing into one object.

    Instantiate once at application startup and pass to ``instrument_app``::

        monitoring = MonitoringService.from_env()
        monitoring.instrument(app)

    Business-level helpers (``record_search``, ``record_chat``, etc.) can
    be called from route handlers for fine-grained telemetry.
    """

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()
        self.langfuse = LangfuseTracer(self.config.langfuse)
        self._metrics = get_metrics()

    @classmethod
    def from_env(cls) -> "MonitoringService":
        return cls(MonitoringConfig.from_env())

    def instrument(self, app: Any) -> Any:
        """Attach Prometheus instrumentator to *app*.  Returns *app*."""
        return instrument_app(app, self.config.prometheus)

    # --- Business metric helpers ---

    def record_search(self, source: str, latency_seconds: float) -> None:
        """Record a completed search request."""
        if "search_requests_total" in self._metrics:
            self._metrics["search_requests_total"].labels(source=source).inc()
        if "search_latency_seconds" in self._metrics:
            self._metrics["search_latency_seconds"].observe(latency_seconds)

    def record_chat(self) -> None:
        """Increment the chat request counter."""
        if "chat_requests_total" in self._metrics:
            self._metrics["chat_requests_total"].inc()

    def record_llm_tokens(
        self, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        """Record token usage from an LLM call."""
        if "llm_tokens_total" not in self._metrics:
            return
        if prompt_tokens:
            self._metrics["llm_tokens_total"].labels(type="prompt").inc(prompt_tokens)
        if completion_tokens:
            self._metrics["llm_tokens_total"].labels(type="completion").inc(
                completion_tokens
            )

    def record_llm_latency(self, latency_seconds: float) -> None:
        """Record how long an LLM call took."""
        if "llm_latency_seconds" in self._metrics:
            self._metrics["llm_latency_seconds"].observe(latency_seconds)

    def record_llm_call(
        self,
        *,
        name: str = "agent-llm-routing",
        model: str | None = None,
        latency_seconds: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        session_id: str | None = None,
        user_id: str | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record one real LLM call in both Prometheus and Langfuse."""
        self.record_llm_latency(latency_seconds)
        self.record_llm_tokens(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self.langfuse.log_generation(
            name=name,
            model=model,
            latency_seconds=latency_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            session_id=session_id,
            user_id=user_id,
            input_text=input_text,
            output_text=output_text,
            metadata=metadata,
        )

    def record_agent_decision(self, action: str, mode: str = "rules") -> None:
        """Record one agent routing decision.

        Args:
            action: the routing action the agent chose.
            mode: ``llm`` when an LLM produced the decision, ``rules`` when the
                keyword fallback did.  Alerting on a drop in ``llm`` is how a
                silently-degraded agent gets noticed.
        """
        if "agent_decisions_total" in self._metrics:
            self._metrics["agent_decisions_total"].labels(
                action=action,
                mode=mode,
            ).inc()

    def record_agent_tool_call(self, tool: str, status: str = "ok") -> None:
        """Record one agent tool call."""
        if "agent_tool_calls_total" in self._metrics:
            self._metrics["agent_tool_calls_total"].labels(
                tool=tool,
                status=status,
            ).inc()

    def record_upload(self, success: bool = True) -> None:
        """Record a catalog upload attempt."""
        if "catalog_uploads_total" in self._metrics:
            label = "success" if success else "error"
            self._metrics["catalog_uploads_total"].labels(status=label).inc()

    def record_click_event(
        self,
        status: str = "published",
        latency_seconds: float = 0.0,
        dlq: bool = False,
    ) -> None:
        """Record one /events/click outcome for Kafka throughput dashboards.

        Args:
            status: ``published`` | ``accepted_no_broker`` | ``error``
            latency_seconds: wall-clock time for the publish call.
            dlq: True when the event was forwarded to the dead-letter topic.
        """
        if "click_events_total" in self._metrics:
            self._metrics["click_events_total"].labels(status=status).inc()
        if "click_event_latency_seconds" in self._metrics:
            self._metrics["click_event_latency_seconds"].observe(latency_seconds)
        if dlq and "click_events_dlq_total" in self._metrics:
            self._metrics["click_events_dlq_total"].inc()

    def flush(self) -> None:
        """Flush Langfuse pending events (call on FastAPI shutdown)."""
        self.langfuse.flush()


# ---------------------------------------------------------------------------
# Convenience: standalone ``observe`` decorator (mirrors roadmap example)
# ---------------------------------------------------------------------------

_default_tracer: LangfuseTracer | None = None


def get_default_tracer() -> LangfuseTracer:
    """Return a module-level default :class:`LangfuseTracer` instance."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = LangfuseTracer()
    return _default_tracer


def observe(
    name: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> Callable:
    """Module-level ``@observe`` decorator backed by the default tracer.

    Usage::

        @observe(name="call-llm", tags=["production"])
        async def call_llm_with_trace(prompt: str) -> str:
            ...
    """
    return get_default_tracer().trace(name=name, user_id=user_id, tags=tags)


# ---------------------------------------------------------------------------
# CLI: python -m src.monitoring status
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    svc = MonitoringService.from_env()
    print("=== SmartShop Monitoring Status ===")
    print(f"Prometheus metrics endpoint : {svc.config.prometheus.metrics_endpoint}")
    print(f"Langfuse enabled            : {svc.langfuse.is_enabled}")
    print(f"Langfuse host               : {svc.config.langfuse.host}")
    print(f"Custom metrics registered   : {list(svc._metrics.keys())}")
    sys.exit(0)
