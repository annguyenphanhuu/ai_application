"""Serve the Phase 3 MLflow registered model inside the API layer.

Loads ``models:/SmartShopRatingClassifier@champion`` (configurable) and scores
products with the same text features used by ``src.train``. MLflow is an
optional dependency: when it is not installed or not configured the service
reports itself as unavailable and the API returns 503 instead of crashing.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URI = "models:/SmartShopRatingClassifier@champion"
TEXT_COLUMNS = ("title", "description", "brand", "category", "price_tier")


@dataclass(frozen=True)
class RatingModelConfig:
    tracking_uri: str | None = None
    model_uri: str = DEFAULT_MODEL_URI
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "RatingModelConfig":
        tracking_uri = (
            os.getenv("SMARTSHOP_MLFLOW_TRACKING_URI")
            or os.getenv("MLFLOW_TRACKING_URI")
            or None
        )
        enabled_raw = os.getenv("SMARTSHOP_RATING_MODEL_ENABLED")
        if enabled_raw is not None:
            enabled = enabled_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            enabled = tracking_uri is not None
        return cls(
            tracking_uri=tracking_uri,
            model_uri=os.getenv("SMARTSHOP_RATING_MODEL_URI", DEFAULT_MODEL_URI),
            enabled=enabled,
        )


def build_product_text(product: dict[str, Any]) -> str:
    """Mirror the feature engineering used during Phase 3 training."""
    parts = [str(product.get(column) or "") for column in TEXT_COLUMNS]
    return " ".join(" ".join(parts).split())


class RatingModelService:
    """Lazy-loading wrapper around the registered rating classifier."""

    def __init__(
        self,
        config: RatingModelConfig | None = None,
        model: Any | None = None,
    ):
        self.config = config or RatingModelConfig.from_env()
        self._model = model
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "RatingModelService":
        return cls(RatingModelConfig.from_env())

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled or self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.config.enabled:
            raise RuntimeError(
                "Rating model serving is not configured. Set "
                "SMARTSHOP_MLFLOW_TRACKING_URI (and optionally "
                "SMARTSHOP_RATING_MODEL_URI) to enable it."
            )
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise RuntimeError(self._load_error)
            try:
                import mlflow
            except ImportError as exc:
                self._load_error = (
                    "mlflow is not installed in this runtime. Deploy the API "
                    "with the full-runtime image target or install mlflow."
                )
                raise RuntimeError(self._load_error) from exc

            try:
                if self.config.tracking_uri:
                    mlflow.set_tracking_uri(self.config.tracking_uri)
                try:
                    import mlflow.sklearn

                    self._model = mlflow.sklearn.load_model(self.config.model_uri)
                except Exception:  # noqa: BLE001 - fall back to generic pyfunc
                    self._model = mlflow.pyfunc.load_model(self.config.model_uri)
            except Exception as exc:  # noqa: BLE001
                message = (
                    f"Could not load rating model '{self.config.model_uri}': {exc}"
                )
                logger.error(message)
                raise RuntimeError(message) from exc

            logger.info("Loaded rating model %s", self.config.model_uri)
            return self._model

    def ping(self) -> bool:
        try:
            self._load_model()
            return True
        except RuntimeError:
            return False

    def predict(self, products: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score products; returns one row per product with label + probability."""
        if not products:
            raise ValueError("products must not be empty.")

        model = self._load_model()
        texts = [build_product_text(product) for product in products]
        if any(not text for text in texts):
            raise ValueError(
                "Each product needs at least one of: " f"{', '.join(TEXT_COLUMNS)}."
            )

        predictions = model.predict(texts)
        probabilities: list[float | None]
        predict_proba = getattr(model, "predict_proba", None)
        if callable(predict_proba):
            probabilities = [float(row[1]) for row in predict_proba(texts)]
        else:
            probabilities = [None] * len(texts)

        results = []
        for product, label, probability in zip(products, predictions, probabilities):
            results.append(
                {
                    "product_id": product.get("product_id"),
                    "high_rating_predicted": bool(int(label)),
                    "high_rating_probability": probability,
                    "model_uri": self.config.model_uri,
                }
            )
        return results
