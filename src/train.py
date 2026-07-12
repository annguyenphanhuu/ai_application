"""Phase 3 baseline training with MLflow tracking.

This script reads the processed product table from Phase 2 and trains a simple
text classifier that predicts whether a product has a high average rating.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.model_registry import (
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_MODEL_ARTIFACT_NAME,
    find_model_version_for_run,
    set_model_alias,
    set_model_version_tags,
)

DEFAULT_INPUT_PATH = "data/processed/amazon_reviews_2023_flow_smoke"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "SmartShop_Rating_Classification"
DEFAULT_MODEL_NAME = "SmartShopRatingClassifier"
TEXT_COLUMNS = ("title", "description", "brand", "category", "price_tier")


@dataclass(frozen=True)
class TrainConfig:
    input_path: str = DEFAULT_INPUT_PATH
    tracking_uri: str = DEFAULT_TRACKING_URI
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    model_name: str = DEFAULT_MODEL_NAME
    rating_threshold: float = 4.0
    test_size: float = 0.25
    random_state: int = 42
    max_features: int = 1000
    register_model: bool = False
    registry_alias: str = DEFAULT_CANDIDATE_ALIAS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SmartShop Phase 3 baseline model."
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the model in MLflow Model Registry.",
    )
    parser.add_argument(
        "--registry-alias",
        default=DEFAULT_CANDIDATE_ALIAS,
        help=(
            "Alias assigned to the registered model version when --register-model "
            "is used. Pass an empty string to skip alias assignment."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    return TrainConfig(
        input_path=args.input_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        rating_threshold=args.rating_threshold,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        register_model=args.register_model,
        registry_alias=args.registry_alias,
    )


def infer_input_format(path: str) -> str:
    clean_path = path.rstrip("/").lower()
    if clean_path.endswith(".csv"):
        return "csv"
    if clean_path.endswith(".json") or clean_path.endswith(".jsonl"):
        return "json"
    return "parquet"


def load_processed_products(path: str) -> pd.DataFrame:
    data_format = infer_input_format(path)
    if data_format == "csv":
        return pd.read_csv(path)
    if data_format == "json":
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


def prepare_training_frame(
    df: pd.DataFrame, rating_threshold: float
) -> tuple[pd.Series, pd.Series]:
    required_columns = {"avg_rating", *TEXT_COLUMNS}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {missing_columns}")

    training_df = df.dropna(subset=["avg_rating"]).copy()
    if training_df.empty:
        raise ValueError("Training data has no rows with avg_rating.")

    text = (
        training_df.loc[:, TEXT_COLUMNS]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    target = (training_df["avg_rating"].astype(float) >= rating_threshold).astype(int)

    if target.nunique() < 2:
        raise ValueError("Training target needs at least two classes.")

    return text, target


def split_training_data(
    text: pd.Series, target: pd.Series, config: TrainConfig
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    class_counts = target.value_counts()
    stratify = target if class_counts.min() >= 2 else None
    return train_test_split(
        text,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )


def build_model(config: TrainConfig) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=config.max_features, ngram_range=(1, 2)),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline, x_test: pd.Series, y_test: pd.Series
) -> dict[str, float]:
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }


def write_metrics_summary(metrics: dict[str, float], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def train(config: TrainConfig) -> dict[str, float]:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for Phase 3 training. Install it with "
            "`pip install mlflow` or `pip install -r requirements.txt`."
        ) from exc

    df = load_processed_products(config.input_path)
    text, target = prepare_training_frame(df, config.rating_threshold)
    x_train, x_test, y_train, y_test = split_training_data(text, target, config)

    model = build_model(config)
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run() as run:
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)

        mlflow.log_params(asdict(config))
        mlflow.log_param("train_rows", len(x_train))
        mlflow.log_param("test_rows", len(x_test))
        mlflow.log_param("positive_rows", int(target.sum()))
        mlflow.log_param("negative_rows", int((target == 0).sum()))
        mlflow.log_metrics(metrics)

        registered_model_name = config.model_name if config.register_model else None
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name=DEFAULT_MODEL_ARTIFACT_NAME,
            registered_model_name=registered_model_name,
            tags={
                "smartshop.phase": "phase3",
                "smartshop.model_type": "rating_classifier",
            },
        )

        metrics_summary_path = "artifacts/phase3_metrics.json"
        write_metrics_summary(metrics, metrics_summary_path)
        mlflow.log_artifact(metrics_summary_path)

        if config.register_model:
            client = mlflow.tracking.MlflowClient()
            registered_version = find_model_version_for_run(
                client, config.model_name, run.info.run_id
            )
            set_model_version_tags(
                client,
                config.model_name,
                registered_version.version,
                {
                    "smartshop.phase": "phase3",
                    "smartshop.model_type": "rating_classifier",
                    "smartshop.run_id": run.info.run_id,
                    "smartshop.model_uri": model_info.model_uri,
                    "metrics.accuracy": metrics["accuracy"],
                    "metrics.f1_score": metrics["f1_score"],
                },
            )
            if config.registry_alias:
                set_model_alias(
                    client,
                    config.model_name,
                    config.registry_alias,
                    registered_version.version,
                )
                print(
                    f"Registered model: {config.model_name} "
                    f"version {registered_version.version} "
                    f"alias '{config.registry_alias}'"
                )

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"Metrics: {json.dumps(metrics, indent=2, sort_keys=True)}")
        return metrics


def main(argv: Sequence[str] | None = None) -> int:
    try:
        train(parse_args(argv))
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    return 0


if __name__ == "__main__":
    main()
