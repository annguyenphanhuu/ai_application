from types import SimpleNamespace

import pytest

from src.model_registry import (
    build_parser,
    find_model_version_for_run,
    format_model_versions,
    list_model_versions,
    promote_model_version,
    set_model_alias,
)


class FakeRegistryClient:
    def __init__(self):
        self.versions = [
            SimpleNamespace(
                name="SmartShopRatingClassifier",
                version="1",
                run_id="run-a",
                aliases=[],
                status="READY",
                current_stage="None",
            ),
            SimpleNamespace(
                name="SmartShopRatingClassifier",
                version="2",
                run_id="run-b",
                aliases=["candidate"],
                status="READY",
                current_stage="None",
            ),
        ]
        self.aliases = []
        self.tags = []

    def search_model_versions(self, _filter_string):
        return self.versions

    def get_registered_model(self, _name):
        return SimpleNamespace(aliases={"candidate": 2})

    def set_registered_model_alias(self, name, alias, version):
        self.aliases.append((name, alias, version))

    def set_model_version_tag(self, name, version, key, value):
        self.tags.append((name, version, key, value))

    def get_model_version(self, name, version):
        for candidate in self.versions:
            if candidate.name == name and candidate.version == version:
                candidate.aliases = ["champion"]
                return candidate
        raise ValueError(version)


def test_find_model_version_for_run_returns_newest_match():
    client = FakeRegistryClient()
    client.versions.append(
        SimpleNamespace(
            name="SmartShopRatingClassifier",
            version="3",
            run_id="run-b",
            aliases=[],
            status="READY",
            current_stage="None",
        )
    )

    version = find_model_version_for_run(
        client, "SmartShopRatingClassifier", run_id="run-b"
    )

    assert version.version == "3"


def test_find_model_version_for_run_raises_when_missing():
    with pytest.raises(ValueError, match="No registered model version"):
        find_model_version_for_run(
            FakeRegistryClient(), "SmartShopRatingClassifier", run_id="missing"
        )


def test_set_model_alias_uses_mlflow_alias_api():
    client = FakeRegistryClient()

    set_model_alias(client, "SmartShopRatingClassifier", "candidate", "2")

    assert client.aliases == [("SmartShopRatingClassifier", "candidate", "2")]


def test_promote_model_version_assigns_alias_and_tags_version():
    client = FakeRegistryClient()

    promoted = promote_model_version(
        client, "SmartShopRatingClassifier", version="2", alias="champion"
    )

    assert promoted.version == "2"
    assert promoted.aliases == ("champion",)
    assert ("SmartShopRatingClassifier", "champion", "2") in client.aliases
    assert (
        "SmartShopRatingClassifier",
        "2",
        "smartshop.registry.promoted",
        "true",
    ) in client.tags


def test_list_and_format_model_versions_show_latest_first():
    versions = list_model_versions(FakeRegistryClient(), "SmartShopRatingClassifier")

    assert [version.version for version in versions] == ["2", "1"]
    assert "version\trun_id\tstatus\tstage\taliases" in format_model_versions(versions)
    assert "2\trun-b\tREADY\tNone\tcandidate" in format_model_versions(versions)


def test_parser_accepts_common_options_after_subcommand():
    args = build_parser().parse_args(
        [
            "promote",
            "--tracking-uri",
            "sqlite:///mlflow.db",
            "--model-name",
            "SmartShopRatingClassifier",
            "--source-alias",
            "candidate",
            "--alias",
            "champion",
        ]
    )

    assert args.command == "promote"
    assert args.tracking_uri == "sqlite:///mlflow.db"
    assert args.model_name == "SmartShopRatingClassifier"
