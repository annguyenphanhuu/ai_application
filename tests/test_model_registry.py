from types import SimpleNamespace

import pytest

from src import model_registry
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


class GatedRegistryClient:
    """Registry fake with runs + metrics so the quality gate can be exercised."""

    def __init__(self, metrics_by_run, champion_version=None):
        self.metrics_by_run = metrics_by_run
        self.champion_version = champion_version
        self.aliases = []
        self.tags = []
        self.versions = {
            version: SimpleNamespace(
                name="SmartShopRatingClassifier",
                version=version,
                run_id=f"run-{version}",
                aliases=[],
                status="READY",
                current_stage="None",
                tags={},
            )
            for version in metrics_by_run
        }

    def get_model_version(self, _name, version):
        return self.versions[str(version)]

    def get_run(self, run_id):
        version = run_id.split("-", 1)[1]
        return SimpleNamespace(
            data=SimpleNamespace(metrics=self.metrics_by_run[version])
        )

    def get_model_version_by_alias(self, _name, alias):
        if alias != "champion" or self.champion_version is None:
            raise ValueError(alias)
        return self.versions[self.champion_version]

    def set_registered_model_alias(self, name, alias, version):
        self.aliases.append((name, alias, version))

    def set_model_version_tag(self, name, version, key, value):
        self.tags.append((name, version, key, value))


def test_promotion_allowed_when_no_champion_exists():
    client = GatedRegistryClient({"1": {"f1_score": 0.5}})

    gate = model_registry.check_promotion_gate(client, "SmartShopRatingClassifier", "1")

    assert gate["allowed"] is True
    assert "first version" in gate["reason"]


def test_promotion_blocked_when_candidate_is_worse():
    client = GatedRegistryClient(
        {"1": {"f1_score": 0.90}, "2": {"f1_score": 0.70}},
        champion_version="1",
    )

    gate = model_registry.check_promotion_gate(client, "SmartShopRatingClassifier", "2")

    assert gate["allowed"] is False
    assert gate["incumbent_metric"] == 0.90
    assert gate["candidate_metric"] == 0.70

    with pytest.raises(model_registry.PromotionBlocked):
        model_registry.promote_model_version(client, "SmartShopRatingClassifier", "2")
    # Nothing was repointed.
    assert client.aliases == []


def test_promotion_allowed_when_candidate_is_better():
    client = GatedRegistryClient(
        {"1": {"f1_score": 0.70}, "2": {"f1_score": 0.85}},
        champion_version="1",
    )

    promoted = model_registry.promote_model_version(
        client, "SmartShopRatingClassifier", "2"
    )

    assert promoted.version == "2"
    assert client.aliases == [("SmartShopRatingClassifier", "champion", "2")]


def test_min_delta_requires_a_real_improvement():
    client = GatedRegistryClient(
        {"1": {"f1_score": 0.800}, "2": {"f1_score": 0.801}},
        champion_version="1",
    )

    # A 0.001 gain does not clear a 0.01 required delta.
    gate = model_registry.check_promotion_gate(
        client, "SmartShopRatingClassifier", "2", min_delta=0.01
    )
    assert gate["allowed"] is False


def test_force_overrides_a_failing_gate():
    client = GatedRegistryClient(
        {"1": {"f1_score": 0.90}, "2": {"f1_score": 0.10}},
        champion_version="1",
    )

    model_registry.promote_model_version(
        client, "SmartShopRatingClassifier", "2", force=True
    )

    assert client.aliases == [("SmartShopRatingClassifier", "champion", "2")]


def test_missing_metric_blocks_promotion():
    client = GatedRegistryClient(
        {"1": {"f1_score": 0.90}, "2": {}},
        champion_version="1",
    )

    gate = model_registry.check_promotion_gate(client, "SmartShopRatingClassifier", "2")

    assert gate["allowed"] is False
    assert "Cannot compare" in gate["reason"]
