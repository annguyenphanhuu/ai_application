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

import pytest

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

    def test_incompatible_sdk_disables_tracing(self, monkeypatch):
        """A v2/v3 SDK has no start_observation and must not look healthy.

        Regression guard: the module previously called the v2-only
        ``client.generation()``. With langfuse v4 installed every call failed
        and was swallowed, while ``is_enabled`` stayed True -- so a completely
        dead tracer was indistinguishable from a working one.
        """
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")

        fake_module = types.ModuleType("langfuse")

        class LegacyLangfuse:
            def __init__(self, **kwargs):  # noqa: ARG002
                pass

            def generation(self, **kwargs):  # v2 API, no start_observation
                raise AssertionError("v2 API must not be used")

        fake_module.Langfuse = LegacyLangfuse

        with patch.dict(sys.modules, {"langfuse": fake_module}):
            mod = _fresh_monitoring()
            tracer = mod.LangfuseTracer()
            assert not tracer.is_enabled
            assert "start_observation" in (tracer.last_error or "")

    def test_log_generation_uses_v4_observation_api(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")

        recorded = {}

        class FakeGeneration:
            def __init__(self):
                self.ended = False

            def end(self):
                self.ended = True

        generation = FakeGeneration()

        class FakeLangfuse:
            def __init__(self, **kwargs):  # noqa: ARG002
                pass

            def start_observation(self, **kwargs):
                recorded.update(kwargs)
                return generation

        import contextlib

        fake_module = types.ModuleType("langfuse")
        fake_module.Langfuse = FakeLangfuse
        fake_module.propagate_attributes = lambda **kw: contextlib.nullcontext()

        with patch.dict(sys.modules, {"langfuse": fake_module}):
            mod = _fresh_monitoring()
            tracer = mod.LangfuseTracer()
            sent = tracer.log_generation(
                name="agent-llm-routing",
                model="o4-mini",
                latency_seconds=0.5,
                prompt_tokens=10,
                completion_tokens=4,
                input_text="hi",
                output_text="hello",
            )

        assert sent is True
        assert tracer.last_error is None
        assert recorded["as_type"] == "generation"
        assert recorded["model"] == "o4-mini"
        assert recorded["usage_details"] == {"input": 10, "output": 4, "total": 14}
        # The span must be closed, otherwise nothing is exported.
        assert generation.ended is True


class TestInstalledLangfuseSdkSurface:
    """Guards against an unpinned langfuse upgrade breaking tracing again."""

    def test_installed_sdk_exposes_the_api_this_module_targets(self):
        langfuse = pytest.importorskip("langfuse")

        assert hasattr(langfuse.Langfuse, "start_observation")
        assert hasattr(langfuse, "propagate_attributes")
        assert hasattr(langfuse, "observe")


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

    def test_instrument_registers_combined_metrics_endpoint(self):
        """When the package IS installed (or mocked), the app is instrumented
        and a /metrics route serving both registries is registered."""
        from src.monitoring import PrometheusConfig, instrument_app

        fake_app = MagicMock()

        # Build a mock instrumentator chain
        mock_inst = MagicMock()
        mock_inst.instrument.return_value = mock_inst
        MockClass = MagicMock(return_value=mock_inst)

        fake_module = types.ModuleType("prometheus_fastapi_instrumentator")
        fake_module.Instrumentator = MockClass

        with patch.dict(
            sys.modules, {"prometheus_fastapi_instrumentator": fake_module}
        ):
            instrument_app(fake_app, PrometheusConfig())

        mock_inst.instrument.assert_called_once_with(fake_app)
        # The combined /metrics endpoint is registered directly on the app.
        fake_app.get.assert_called_once_with("/metrics", include_in_schema=False)


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


# ---------------------------------------------------------------------------
# record_llm_call: Prometheus + Langfuse in one call
# ---------------------------------------------------------------------------


class TestRecordLLMCall:
    def test_record_llm_call_records_metrics_and_langfuse(self):
        from src.monitoring import MonitoringService

        service = MonitoringService()
        service.langfuse = MagicMock()
        service.record_llm_latency = MagicMock()
        service.record_llm_tokens = MagicMock()

        service.record_llm_call(
            name="agent-llm-routing",
            model="o4-mini",
            latency_seconds=1.5,
            prompt_tokens=42,
            completion_tokens=7,
            session_id="S01",
        )

        service.record_llm_latency.assert_called_once_with(1.5)
        service.record_llm_tokens.assert_called_once_with(
            prompt_tokens=42, completion_tokens=7
        )
        service.langfuse.log_generation.assert_called_once()
        kwargs = service.langfuse.log_generation.call_args.kwargs
        assert kwargs["model"] == "o4-mini"
        assert kwargs["prompt_tokens"] == 42
        assert kwargs["session_id"] == "S01"

    def test_log_generation_returns_false_when_disabled(self):
        from src.monitoring import LangfuseTracer

        tracer = LangfuseTracer()

        assert tracer.is_enabled is False
        assert tracer.log_generation(name="test", model="o4-mini") is False


class TestMetricsEndpointExposesCustomRegistry:
    def test_metrics_output_includes_smartshop_business_metrics(self):
        import pytest

        pytest.importorskip("prometheus_client")
        pytest.importorskip("prometheus_fastapi_instrumentator")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.monitoring import MonitoringService, instrument_app

        service = MonitoringService()
        service.record_search(source="cache", latency_seconds=0.01)
        service.record_click_event(status="published", latency_seconds=0.005)

        app = FastAPI()
        instrument_app(app)
        client = TestClient(app)

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "smartshop_search_requests_total" in response.text
        assert "smartshop_click_events_total" in response.text
