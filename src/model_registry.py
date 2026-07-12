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


def promote_model_version(
    client: Any,
    model_name: str,
    version: str | int,
    alias: str = DEFAULT_CHAMPION_ALIAS,
) -> RegisteredVersion:
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
        promoted = promote_model_version(client, args.model_name, version, args.alias)
        print(
            f"Promoted {promoted.name} version {promoted.version} "
            f"to alias '{args.alias}'."
        )
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
