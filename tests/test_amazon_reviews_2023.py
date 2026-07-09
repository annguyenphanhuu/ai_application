import pytest

from jobs.amazon_reviews_2023 import (
    ALL_CATEGORIES,
    DEFAULT_CATEGORY,
    DEFAULT_DATASET_NAME,
    AmazonReviews2023Config,
    discover_hub_categories,
    env_int,
    limit_records,
    parse_categories,
    parse_args,
)


def test_config_builds_hugging_face_config_names():
    config = AmazonReviews2023Config(categories=("All_Beauty",))

    assert config.dataset_name == DEFAULT_DATASET_NAME
    assert config.category == DEFAULT_CATEGORY
    assert config.meta_config == "raw_meta_All_Beauty"
    assert config.review_config == "raw_review_All_Beauty"
    assert config.meta_hub_path("All_Beauty") == (
        "raw/meta_categories/meta_All_Beauty.jsonl"
    )
    assert config.reviews_hub_path("All_Beauty") == (
        "raw/review_categories/All_Beauty.jsonl"
    )
    assert str(config.meta_output_path("All_Beauty")).endswith(
        "All_Beauty\\meta.jsonl"
    ) or str(config.meta_output_path("All_Beauty")).endswith("All_Beauty/meta.jsonl")
    assert str(config.reviews_output_path("All_Beauty")).endswith(
        "All_Beauty\\reviews.jsonl"
    ) or str(config.reviews_output_path("All_Beauty")).endswith(
        "All_Beauty/reviews.jsonl"
    )
    assert str(config.combined_meta_output_path).endswith(
        "combined\\meta.jsonl"
    ) or str(config.combined_meta_output_path).endswith("combined/meta.jsonl")


def test_parse_categories_accepts_comma_separated_values():
    categories = parse_categories("All_Beauty, Electronics,All_Beauty")

    assert categories == ("All_Beauty", "Electronics")


def test_parse_categories_accepts_all_categories():
    categories = parse_categories("all")

    assert categories == ALL_CATEGORIES
    assert len(categories) == 34


def test_parse_categories_defaults_to_all_categories():
    categories = parse_categories(None)

    assert categories == ALL_CATEGORIES


def test_parse_categories_rejects_unknown_category():
    with pytest.raises(ValueError, match="Unknown Amazon Reviews 2023 categories"):
        parse_categories("All_Beauty,Not_A_Category")


def test_discover_hub_categories_uses_only_raw_data_intersection(monkeypatch):
    def fake_list_repo_files(repo_id, repo_type):
        assert repo_id == DEFAULT_DATASET_NAME
        assert repo_type == "dataset"
        return [
            "raw/meta_categories/meta_All_Beauty.jsonl",
            "raw/review_categories/All_Beauty.jsonl",
            "raw/meta_categories/meta_Electronics.jsonl",
            "raw/review_categories/Electronics.jsonl",
            "raw/meta_categories/meta_Books.jsonl",
        ]

    monkeypatch.setattr("huggingface_hub.list_repo_files", fake_list_repo_files)

    categories = discover_hub_categories(AmazonReviews2023Config())

    assert categories == ("All_Beauty", "Electronics")


def test_parse_args_defaults_to_all_categories(monkeypatch):
    monkeypatch.delenv("SMARTSHOP_AMAZON_CATEGORIES", raising=False)
    monkeypatch.delenv("SMARTSHOP_AMAZON_CATEGORY", raising=False)

    config = parse_args([])

    assert config.categories == ALL_CATEGORIES


def test_parse_args_accepts_categories_and_limits():
    config = parse_args(
        [
            "--categories",
            "Cell_Phones_and_Accessories,Electronics",
            "--max-products",
            "100",
            "--max-reviews",
            "250",
        ]
    )

    assert config.categories == ("Cell_Phones_and_Accessories", "Electronics")
    assert config.max_products == 100
    assert config.max_reviews == 250


def test_parse_args_keeps_legacy_category_option():
    config = parse_args(
        [
            "--category",
            "Cell_Phones_and_Accessories",
            "--max-products",
            "100",
            "--max-reviews",
            "250",
        ]
    )

    assert config.categories == ("Cell_Phones_and_Accessories",)
    assert config.max_products == 100
    assert config.max_reviews == 250


def test_parse_args_full_disables_limits():
    config = parse_args(["--full"])

    assert config.categories == ALL_CATEGORIES
    assert config.max_products is None
    assert config.max_reviews is None


def test_env_int_accepts_full_markers(monkeypatch):
    monkeypatch.setenv("SMARTSHOP_MAX_PRODUCTS", "full")

    assert env_int("SMARTSHOP_MAX_PRODUCTS", 100) is None


def test_limit_records_truncates_iterable():
    records = ({"id": value} for value in range(5))

    assert list(limit_records(records, 2)) == [{"id": 0}, {"id": 1}]
