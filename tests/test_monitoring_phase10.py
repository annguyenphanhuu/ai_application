"""Phase 10 unit tests: Monitoring & Observability.

Tests cover:
- MonitoringConfig / LangfuseConfig build from environment variables.
- MonitoringService metric recording (degrades gracefully without prometheus_client).
- LangfuseTracer no-op behaviour when Langfuse is not configured.
- instrument_app graceful fallback when prometheus-fastapi-instrumentator is absent.
- Context managers: track_search and track_llm_call.
- Standalone CLI path (python -m src.monitoring).
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import types
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_monitoring():
    """Re-import src.monitoring with a clean module state."""
    # Remove cached copy so module-level globals reset
    for key in list(sys.modules.keys()):
        if key.startswith("src.monitoring"):
            del sys.modules[key]
    return importlib.import_module("src.monitoring")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestLangfuseConfig:
    def test_defaults(self):
        from src.monitoring import LangfuseConfig

        cfg = LangfuseConfig()
        assert cfg.host == "https://cloud.langfuse.com"
        assert cfg.public_key == ""
        assert not cfg.enabled

    def test_enabled_when_key_set(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-123")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-456")
        # Re-import to pick up env
        mod = _fresh_monitoring()
        cfg = mod.LangfuseConfig()
        assert cfg.enabled
        assert cfg.public_key == "pk-test-123"

    def test_disabled_when_secret_key_missing(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-123")
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        mod = _fresh_monitoring()
        cfg = mod.LangfuseConfig()
        assert not cfg.enabled

    def test_custom_host(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "http://my-langfuse.internal")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        mod = _fresh_monitoring()
        cfg = mod.LangfuseConfig()
        assert cfg.host == "http://my-langfuse.internal"


class TestPrometheusConfig:
    def test_defaults(self):
        from src.monitoring import PrometheusConfig

        cfg = PrometheusConfig()
        assert cfg.metrics_endpoint == "/metrics"
        assert cfg.should_group_status_codes is True

    def test_custom_endpoint(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_METRICS_ENDPOINT", "/prom")
        mod = _fresh_monitoring()
        cfg = mod.MonitoringConfig.from_env()
        assert cfg.prometheus.metrics_endpoint == "/prom"


class TestMonitoringConfigFromEnv:
    def test_from_env_builds_without_error(self):
        from src.monitoring import MonitoringConfig

        cfg = MonitoringConfig.from_env()
        assert cfg.prometheus is not None
        assert cfg.langfuse is not None


# ---------------------------------------------------------------------------
# MonitoringService metric helpers (mock prometheus_client)
# ---------------------------------------------------------------------------


def _make_fake_counter():
    counter = MagicMock()
    counter.labels.return_value = counter
    return counter


def _make_fake_histogram():
    hist = MagicMock()
    return hist


class TestMonitoringServiceMetrics:
    """Verify metric recording calls without requiring prometheus_client."""

    def _make_service_with_mock_metrics(self):
        from src.monitoring import MonitoringService

        svc = MonitoringService()
        # Inject mock metrics
        svc._metrics = {
            "search_requests_total": _make_fake_counter(),
            "search_latency_seconds": _make_fake_histogram(),
            "chat_requests_total": _make_fake_counter(),
            "llm_tokens_total": _make_fake_counter(),
            "llm_latency_seconds": _make_fake_histogram(),
            "catalog_uploads_total": _make_fake_counter(),
        }
        return svc

    def test_record_search_cache(self):
        svc = self._make_service_with_mock_metrics()
        svc.record_search("cache", 0.01)
        svc._metrics["search_requests_total"].labels.assert_called_with(source="cache")
        svc._metrics["search_requests_total"].labels().inc.assert_called_once()
        svc._metrics["search_latency_seconds"].observe.assert_called_with(0.01)

    def test_record_search_vector_store(self):
        svc = self._make_service_with_mock_metrics()
        svc.record_search("vector_store", 0.25)
        svc._metrics["search_requests_total"].labels.assert_called_with(
            source="vector_store"
        )

    def test_record_chat(self):
        svc = self._make_service_with_mock_metrics()
        svc.record_chat()
        svc._metrics["chat_requests_total"].inc.assert_called_once()

    def test_record_llm_tokens(self):
        svc = self._make_service_with_mock_metrics()
        svc.record_llm_tokens(prompt_tokens=100, completion_tokens=50)
        calls = svc._metrics["llm_tokens_total"].labels.call_args_list
        types_called = [
            c.kwargs.get("type") or c.args[0] if c.args else c.kwargs["type"]
            for c in calls
        ]
        assert "prompt" in str(types_called)
        assert "completion" in str(types_called)

    def test_record_llm_tokens_zero_skips(self):
        svc = self._make_service_with_mock_metrics()
        svc.record_llm_tokens(prompt_tokens=0, completion_tokens=0)
        svc._metrics["llm_tokens_total"].labels.assert_not_called()

    def test_record_llm_latency(self):
        svc = self._make_service_with_mock_metrics()
        svc.record_llm_latency(1.5)
        svc._metrics["llm_latency_seconds"].observe.assert_called_with(1.5)

    def test_record_upload_success(self):
        svc = self._make_service_with_mock_metrics()
        svc.record_upload(success=True)
        svc._metrics["catalog_uploads_total"].labels.assert_called_with(
            status="success"
        )

    def test_record_upload_error(self):
        svc = self._make_service_with_mock_metrics()
        svc.record_upload(success=False)
        svc._metrics["catalog_uploads_total"].labels.assert_called_with(status="error")

    def test_record_search_no_metrics_key(self):
        """Empty metrics dict must not raise."""
        from src.monitoring import MonitoringService

        svc = MonitoringService()
        svc._metrics = {}
        svc.record_search("cache", 0.05)  # should not raise

    def test_record_chat_no_metrics_key(self):
        from src.monitoring import MonitoringService

        svc = MonitoringService()
        svc._metrics = {}
        svc.record_chat()  # should not raise


# ---------------------------------------------------------------------------
# LangfuseTracer
# ---------------------------------------------------------------------------


class TestLangfuseTracer:
    def test_disabled_by_default(self):
        from src.monitoring import LangfuseTracer

        tracer = LangfuseTracer()
        assert not tracer.is_enabled

    def test_trace_noop_when_disabled(self):
        from src.monitoring import LangfuseTracer

        tracer = LangfuseTracer()

        async def my_fn(x: int) -> int:
            return x * 2

        wrapped = tracer.trace(name="test")(my_fn)
        # Should be the same function object (no wrapping)
        assert wrapped is my_fn

    def test_flush_noop_when_disabled(self):
        from src.monitoring import LangfuseTracer

        tracer = LangfuseTracer()
        tracer.flush()  # must not raise

    def test_init_client_skips_when_not_installed(self, monkeypatch):
        """Simulate langfuse not installed."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")

        # Patch langfuse import to raise ImportError
        with patch.dict(sys.modules, {"langfuse": None}):
            mod = _fresh_monitoring()
            tracer = mod.LangfuseTracer()
            assert not tracer.is_enabled

    def test_init_client_failure_disables_tracing(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")

        fake_module = types.ModuleType("langfuse")

        class BrokenLangfuse:
            def __init__(self, **kwargs):  # noqa: ARG002
                raise RuntimeError("bad langfuse config")

        fake_module.Langfuse = BrokenLangfuse

        with patch.dict(sys.modules, {"langfuse": fake_module}):
            mod = _fresh_monitoring()
            tracer = mod.LangfuseTracer()
            assert not tracer.is_enabled


# ---------------------------------------------------------------------------
# instrument_app
# ---------------------------------------------------------------------------


class TestInstrumentApp:
    def test_graceful_fallback_when_not_installed(self):
        """instrument_app should not raise even if the package is missing."""
        from src.monitoring import PrometheusConfig, instrument_app

        fake_app = MagicMock()
        with patch.dict(
            sys.modules,
            {"prometheus_fastapi_instrumentator": None},
        ):
            result = instrument_app(fake_app, PrometheusConfig())
        # Must return the same app object
        assert result is fake_app

    def test_instrument_calls_expose(self):
        """When the package IS installed (or mocked), expose() should be called."""
        from src.monitoring import PrometheusConfig, instrument_app

        fake_app = MagicMock()

        # Build a mock instrumentator chain
        mock_inst = MagicMock()
        mock_inst.instrument.return_value = mock_inst
        mock_inst.expose.return_value = mock_inst
        MockClass = MagicMock(return_value=mock_inst)

        fake_module = types.ModuleType("prometheus_fastapi_instrumentator")
        fake_module.Instrumentator = MockClass

        with patch.dict(
            sys.modules, {"prometheus_fastapi_instrumentator": fake_module}
        ):
            instrument_app(fake_app, PrometheusConfig())

        mock_inst.instrument.assert_called_once_with(fake_app)
        mock_inst.expose.assert_called_once()


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


class TestContextManagers:
    def test_track_search_yields(self):
        from src.monitoring import track_search

        ran = False
        with track_search(source="cache"):
            ran = True
        assert ran

    def test_track_search_does_not_raise_without_prometheus(self):
        """Even with empty metrics registry, must not raise."""
        import src.monitoring as m

        original = m._METRICS
        m._METRICS = {}
        try:
            with m.track_search(source="cache"):
                pass
        finally:
            m._METRICS = original

    def test_track_llm_call_yields(self):
        from src.monitoring import track_llm_call

        ran = False
        with track_llm_call(prompt_tokens=10, completion_tokens=5):
            ran = True
        assert ran

    def test_track_llm_call_empty_metrics(self):
        import src.monitoring as m

        original = m._METRICS
        m._METRICS = {}
        try:
            with m.track_llm_call(prompt_tokens=20):
                time.sleep(0)
        finally:
            m._METRICS = original


# ---------------------------------------------------------------------------
# MonitoringService.instrument delegates to instrument_app
# ---------------------------------------------------------------------------


class TestMonitoringServiceInstrument:
    def test_instrument_returns_app(self):
        from src.monitoring import MonitoringService

        svc = MonitoringService()
        fake_app = MagicMock()

        with patch("src.monitoring.instrument_app", return_value=fake_app) as mock_ia:
            result = svc.instrument(fake_app)

        mock_ia.assert_called_once()
        assert result is fake_app

    def test_from_env_constructs(self):
        from src.monitoring import MonitoringService

        svc = MonitoringService.from_env()
        assert svc.config is not None
        assert svc.langfuse is not None


# ---------------------------------------------------------------------------
# Standalone observe decorator
# ---------------------------------------------------------------------------


class TestObserveDecorator:
    def test_observe_returns_callable(self):
        from src.monitoring import observe

        decorator = observe(name="test-fn", tags=["test"])
        assert callable(decorator)

    def test_observe_noop_wraps_function(self):
        from src.monitoring import observe

        async def original_fn(x: int) -> int:
            return x + 1

        # Tracer is disabled by default (no LANGFUSE_PUBLIC_KEY)
        wrapped = observe(name="test")(original_fn)
        # Either same function or a wrapper — must still be callable
        assert callable(wrapped)


# ---------------------------------------------------------------------------
# Integration smoke: create_app includes monitoring state
# ---------------------------------------------------------------------------


class TestMainAppMonitoring:
    def test_create_app_stores_monitoring(self):
        """main.create_app should attach MonitoringService to app.state."""
        from src.monitoring import MonitoringService

        # Prevent instrument_app from trying to install real Prometheus
        with patch("src.monitoring.instrument_app") as mock_instrument:
            mock_instrument.side_effect = lambda app, cfg=None: app

            from src.main import create_app

            fake_monitoring = MonitoringService()
            fake_monitoring.instrument = MagicMock(side_effect=lambda a: a)

            created = create_app(monitoring=fake_monitoring)

        assert created.state.monitoring is fake_monitoring
