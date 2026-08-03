"""Utilities for working with the SmartShop MLflow Model Registry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_MODEL_NAME = "SmartShopRatingClassifier"
DEFAULT_MODEL_ARTIFACT_NAME = "rating_classifier"
DEFAULT_CANDIDATE_ALIAS = "candidate"
DEFAULT_CHAMPION_ALIAS = "champion"


@dataclass(frozen=True)
class RegisteredVersion:
    name: str
    version: str
    run_id: str | None = None
    aliases: tuple[str, ...] = ()
    status: str | None = None
    current_stage: str | None = None


def get_mlflow_client(tracking_uri: str):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for model registry commands. Install it with "
            "`pip install mlflow` or update the Conda environment."
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def _version_number(version: Any) -> int:
    try:
        return int(getattr(version, "version"))
    except (TypeError, ValueError):
        return -1


def _escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def normalize_model_version(version: Any) -> RegisteredVersion:
    return RegisteredVersion(
        name=str(getattr(version, "name", "")),
        version=str(getattr(version, "version", "")),
        run_id=getattr(version, "run_id", None),
        aliases=tuple(getattr(version, "aliases", ()) or ()),
        status=getattr(version, "status", None),
        current_stage=getattr(version, "current_stage", None),
    )


def find_model_version_for_run(client: Any, model_name: str, run_id: str) -> Any:
    """Return the newest registered model version created from an MLflow run."""
    versions = client.search_model_versions(
        f"name = '{_escape_filter_value(model_name)}'"
    )
    matches = [
        version for version in versions if getattr(version, "run_id", None) == run_id
    ]
    if not matches:
        raise ValueError(
            f"No registered model version found for model '{model_name}' and run '{run_id}'."
        )
    return sorted(matches, key=_version_number)[-1]


def list_model_versions(client: Any, model_name: str) -> list[RegisteredVersion]:
    versions = client.search_model_versions(
        f"name = '{_escape_filter_value(model_name)}'"
    )
    alias_map = getattr(client.get_registered_model(model_name), "aliases", {}) or {}
    aliases_by_version: dict[str, list[str]] = {}
    for alias, version in alias_map.items():
        aliases_by_version.setdefault(str(version), []).append(alias)

    return [
        RegisteredVersion(
            name=normalized.name,
            version=normalized.version,
            run_id=normalized.run_id,
            aliases=tuple(sorted(set(normalized.aliases) | set(alias_candidates))),
            status=normalized.status,
            current_stage=normalized.current_stage,
        )
        for version in sorted(versions, key=_version_number, reverse=True)
        for normalized in [normalize_model_version(version)]
        for alias_candidates in [aliases_by_version.get(normalized.version, [])]
    ]


def set_model_version_tags(
    client: Any, model_name: str, version: str | int, tags: dict[str, Any]
) -> None:
    for key, value in tags.items():
        if value is not None:
            client.set_model_version_tag(model_name, str(version), key, str(value))


def set_model_alias(
    client: Any, model_name: str, alias: str, version: str | int
) -> None:
    if not hasattr(client, "set_registered_model_alias"):
        raise RuntimeError(
            "This MLflow version does not support model aliases. Upgrade MLflow to "
            "use the SmartShop registry workflow."
        )
    client.set_registered_model_alias(model_name, alias, str(version))


def get_model_version_by_alias(client: Any, model_name: str, alias: str) -> Any:
    if not hasattr(client, "get_model_version_by_alias"):
        raise RuntimeError(
            "This MLflow version does not support model aliases. Upgrade MLflow to "
            "use the SmartShop registry workflow."
        )
    return client.get_model_version_by_alias(model_name, alias)


class PromotionBlocked(RuntimeError):
    """Raised when a candidate fails the quality gate against the champion."""


def get_version_metric(client: Any, model_name: str, version: str | int, metric: str):
    """Read one metric for a model version, preferring the run over the tag.

    Metrics live on the MLflow run; ``train.py`` also copies them onto the
    model version as ``metrics.<name>`` tags, so fall back to those when the
    run is gone.
    """
    try:
        model_version = client.get_model_version(model_name, str(version))
    except Exception:  # noqa: BLE001
        return None

    run_id = getattr(model_version, "run_id", None)
    if run_id:
        try:
            run = client.get_run(run_id)
            value = run.data.metrics.get(metric)
            if value is not None:
                return float(value)
        except Exception:  # noqa: BLE001
            pass

    tags = getattr(model_version, "tags", None) or {}
    raw = tags.get(f"metrics.{metric}")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def check_promotion_gate(
    client: Any,
    model_name: str,
    version: str | int,
    alias: str = DEFAULT_CHAMPION_ALIAS,
    metric: str = "f1_score",
    min_delta: float = 0.0,
) -> dict[str, Any]:
    """Compare a candidate against the incumbent holder of *alias*.

    Returns a report dict. Promotion is allowed when there is no incumbent
    (first champion), or when the candidate's *metric* is at least the
    incumbent's plus *min_delta*.

    Without this, ``promote`` repointed production traffic unconditionally, so
    a model trained on a bad batch could take over with one command.
    """
    report: dict[str, Any] = {
        "metric": metric,
        "min_delta": min_delta,
        "candidate_version": str(version),
        "candidate_metric": get_version_metric(client, model_name, version, metric),
        "incumbent_version": None,
        "incumbent_metric": None,
        "allowed": True,
        "reason": "",
    }

    try:
        incumbent = get_model_version_by_alias(client, model_name, alias)
    except Exception:  # noqa: BLE001
        incumbent = None

    if incumbent is None:
        report["reason"] = f"No current '{alias}'; promoting the first version."
        return report

    incumbent_version = str(getattr(incumbent, "version", ""))
    report["incumbent_version"] = incumbent_version
    if incumbent_version == str(version):
        report["reason"] = f"Version {version} already holds '{alias}'."
        return report

    report["incumbent_metric"] = get_version_metric(
        client, model_name, incumbent_version, metric
    )

    candidate_metric = report["candidate_metric"]
    incumbent_metric = report["incumbent_metric"]
    if candidate_metric is None or incumbent_metric is None:
        report["allowed"] = False
        report["reason"] = (
            f"Cannot compare '{metric}': candidate={candidate_metric}, "
            f"incumbent={incumbent_metric}. Re-run training so the metric is "
            f"logged, or pass --force to override."
        )
        return report

    required = incumbent_metric + min_delta
    if candidate_metric < required:
        report["allowed"] = False
        report["reason"] = (
            f"Candidate {metric}={candidate_metric:.4f} is below the required "
            f"{required:.4f} (incumbent {incumbent_metric:.4f} + "
            f"min_delta {min_delta}). Promotion blocked."
        )
        return report

    report["reason"] = (
        f"Candidate {metric}={candidate_metric:.4f} >= required {required:.4f}."
    )
    return report


def promote_model_version(
    client: Any,
    model_name: str,
    version: str | int,
    alias: str = DEFAULT_CHAMPION_ALIAS,
    metric: str = "f1_score",
    min_delta: float = 0.0,
    force: bool = False,
) -> RegisteredVersion:
    if not force:
        gate = check_promotion_gate(
            client, model_name, version, alias=alias, metric=metric, min_delta=min_delta
        )
        if not gate["allowed"]:
            raise PromotionBlocked(gate["reason"])

    set_model_alias(client, model_name, alias, version)
    set_model_version_tags(
        client,
        model_name,
        version,
        {
            "smartshop.registry.alias": alias,
            "smartshop.registry.promoted": "true",
        },
    )
    return normalize_model_version(client.get_model_version(model_name, str(version)))


def format_model_versions(versions: Iterable[RegisteredVersion]) -> str:
    rows = ["version\trun_id\tstatus\tstage\taliases"]
    for version in versions:
        rows.append(
            "\t".join(
                [
                    version.version,
                    version.run_id or "-",
                    version.status or "-",
                    version.current_stage or "-",
                    ",".join(version.aliases) or "-",
                ]
            )
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage SmartShop models in the MLflow Model Registry."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
        command_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)

    list_parser = subparsers.add_parser(
        "list", help="List registered versions for a model."
    )
    add_common_options(list_parser)

    promote_parser = subparsers.add_parser(
        "promote", help="Assign an alias such as champion to a registered version."
    )
    add_common_options(promote_parser)
    version_group = promote_parser.add_mutually_exclusive_group(required=True)
    version_group.add_argument("--version", help="Registered model version to promote.")
    version_group.add_argument(
        "--source-alias",
        default=None,
        help="Promote the version currently pointed to by this alias.",
    )
    promote_parser.add_argument("--alias", default=DEFAULT_CHAMPION_ALIAS)
    promote_parser.add_argument(
        "--gate-metric",
        default="f1_score",
        help="Metric compared against the incumbent before promoting.",
    )
    promote_parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Required improvement over the incumbent (default: no regression).",
    )
    promote_parser.add_argument(
        "--force",
        action="store_true",
        help="Promote even if the quality gate fails.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = get_mlflow_client(args.tracking_uri)

    if args.command == "list":
        print(format_model_versions(list_model_versions(client, args.model_name)))
        return 0

    if args.command == "promote":
        version = args.version
        if args.source_alias:
            version = get_model_version_by_alias(
                client, args.model_name, args.source_alias
            ).version
        gate = check_promotion_gate(
            client,
            args.model_name,
            version,
            alias=args.alias,
            metric=args.gate_metric,
            min_delta=args.min_delta,
        )
        print(f"Quality gate: {gate['reason']}")
        if not gate["allowed"] and not args.force:
            print("Promotion aborted. Re-run with --force to override.")
            return 1

        promoted = promote_model_version(
            client,
            args.model_name,
            version,
            args.alias,
            metric=args.gate_metric,
            min_delta=args.min_delta,
            force=True,  # already evaluated above
        )
        print(
            f"Promoted {promoted.name} version {promoted.version} "
            f"to alias '{args.alias}'."
        )
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
