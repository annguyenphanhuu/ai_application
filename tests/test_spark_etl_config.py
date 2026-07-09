import pytest

from jobs.spark_etl import (
    DEFAULT_PRODUCTS_INPUT,
    DEFAULT_PRODUCTS_OUTPUT,
    DEFAULT_REVIEWS_INPUT,
    EtlConfig,
    infer_format,
    parse_args,
    validate_windows_local_spark,
)


def test_parse_args_uses_defaults():
    config = parse_args([])

    assert config == EtlConfig(
        products_input=DEFAULT_PRODUCTS_INPUT,
        products_output=DEFAULT_PRODUCTS_OUTPUT,
        reviews_input=DEFAULT_REVIEWS_INPUT,
        output_format="parquet",
        master=None,
        write_mode="overwrite",
    )


def test_parse_args_accepts_pipeline_parameters():
    config = parse_args(
        [
            "--input-products",
            "dbfs:/mnt/raw/products.csv",
            "--input-reviews",
            "dbfs:/mnt/raw/reviews.jsonl",
            "--output",
            "dbfs:/mnt/processed/products",
            "--output-format",
            "parquet",
            "--master",
            "local[*]",
            "--write-mode",
            "append",
        ]
    )

    assert config.products_input == "dbfs:/mnt/raw/products.csv"
    assert config.reviews_input == "dbfs:/mnt/raw/reviews.jsonl"
    assert config.products_output == "dbfs:/mnt/processed/products"
    assert config.output_format == "parquet"
    assert config.master == "local[*]"
    assert config.write_mode == "append"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("data/raw/products.csv", "csv"),
        ("data/raw/products.json", "json"),
        ("data/raw/products.jsonl", "json"),
        ("data/raw/products.parquet", "parquet"),
        ("data/raw/products_delta", "delta"),
    ],
)
def test_infer_format_from_path(path, expected):
    assert infer_format(path) == expected


def test_infer_format_uses_explicit_format():
    assert infer_format("data/raw/products.unknown", "parquet") == "parquet"


def test_infer_format_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported data format"):
        infer_format("data/raw/products.xml")


def test_windows_local_spark_requires_winutils(monkeypatch):
    config = EtlConfig(
        products_input="data/raw/amazon_reviews_2023/combined/meta.jsonl",
        products_output="data/processed/amazon_reviews_2023_flow_smoke",
        output_format="parquet",
        master="local[*]",
    )
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.delenv("HADOOP_HOME", raising=False)
    monkeypatch.delenv("hadoop.home.dir", raising=False)

    with pytest.raises(RuntimeError, match="winutils.exe"):
        validate_windows_local_spark(config)


def test_windows_local_spark_accepts_configured_winutils(monkeypatch, tmp_path):
    hadoop_home = tmp_path / "hadoop"
    winutils = hadoop_home / "bin" / "winutils.exe"
    winutils.parent.mkdir(parents=True)
    winutils.write_text("", encoding="utf-8")
    config = EtlConfig(
        products_input="data/raw/amazon_reviews_2023/combined/meta.jsonl",
        products_output="data/processed/amazon_reviews_2023_flow_smoke",
        output_format="parquet",
        master="local[*]",
    )
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("HADOOP_HOME", str(hadoop_home))

    validate_windows_local_spark(config)
