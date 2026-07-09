"""SmartShop product ingestion and ETL pipeline.

The module keeps Spark imports inside runtime functions so unit tests can check
configuration and path handling without requiring a local Spark installation.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_PRODUCTS_INPUT = "data/raw/amazon_reviews_2023/combined/meta.jsonl"
DEFAULT_REVIEWS_INPUT = "data/raw/amazon_reviews_2023/combined/reviews.jsonl"
DEFAULT_PRODUCTS_OUTPUT = "data/processed/amazon_reviews_2023_flow_smoke"
SUPPORTED_FORMATS = {"csv", "delta", "json", "jsonl", "parquet"}
REMOTE_PATH_MARKERS = ("://", "dbfs:/")


@dataclass(frozen=True)
class EtlConfig:
    products_input: str
    products_output: str
    reviews_input: str | None = None
    output_format: str = "parquet"
    master: str | None = None
    app_name: str = "SmartShop-Product-ETL"
    write_mode: str = "overwrite"


def env_default(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else fallback


def infer_format(path: str, explicit_format: str | None = None) -> str:
    if explicit_format:
        normalized = explicit_format.lower()
    else:
        clean_path = path.rstrip("/").lower()
        if "." in clean_path:
            normalized = clean_path.rsplit(".", 1)[-1]
        elif clean_path.endswith("_delta") or clean_path.endswith("/delta"):
            normalized = "delta"
        else:
            normalized = "json"

    if normalized == "jsonl":
        return "json"
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported data format '{normalized}'. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )
    return normalized


def is_remote_path(path: str | None) -> bool:
    return bool(path) and any(marker in path.lower() for marker in REMOTE_PATH_MARKERS)


def is_local_spark_master(master: str | None) -> bool:
    return master in (None, "") or master.startswith("local")


def validate_windows_local_spark(config: EtlConfig) -> None:
    if not sys.platform.startswith("win") or not is_local_spark_master(config.master):
        return

    if is_remote_path(config.products_output):
        return

    hadoop_home = os.getenv("HADOOP_HOME") or os.getenv("hadoop.home.dir")
    winutils_path = Path(hadoop_home, "bin", "winutils.exe") if hadoop_home else None
    if winutils_path and winutils_path.exists():
        return

    raise RuntimeError(
        "Spark local mode on Windows needs HADOOP_HOME pointing to a folder that "
        "contains bin\\winutils.exe before writing Parquet/Delta. "
        "Set HADOOP_HOME, add %HADOOP_HOME%\\bin to PATH, then rerun this ETL. "
        "Alternative: run the ETL from WSL or Docker where this Windows helper is "
        "not needed."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartShop AI - Spark ETL Job")
    parser.add_argument(
        "--input-products",
        default=env_default("SMARTSHOP_PRODUCTS_INPUT", DEFAULT_PRODUCTS_INPUT),
        help="Raw product metadata path. Supports JSON/JSONL/CSV/Parquet/Delta.",
    )
    parser.add_argument(
        "--input-reviews",
        default=env_default("SMARTSHOP_REVIEWS_INPUT", DEFAULT_REVIEWS_INPUT),
        help="Optional review data path used to aggregate avg_rating/review_count.",
    )
    parser.add_argument(
        "--output",
        "--output-path",
        dest="output",
        default=env_default("SMARTSHOP_PRODUCTS_OUTPUT", DEFAULT_PRODUCTS_OUTPUT),
        help="Destination path for the processed product table.",
    )
    parser.add_argument(
        "--output-format",
        default=env_default("SMARTSHOP_OUTPUT_FORMAT", "parquet"),
        choices=sorted(SUPPORTED_FORMATS - {"jsonl"}),
        help="Storage format for processed data.",
    )
    parser.add_argument(
        "--master",
        default=env_default("SPARK_MASTER"),
        help="Optional Spark master, for example local[*].",
    )
    parser.add_argument(
        "--write-mode",
        default=env_default("SMARTSHOP_WRITE_MODE", "overwrite"),
        choices=("append", "overwrite"),
        help="Spark write mode for the output dataset.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> EtlConfig:
    args = build_parser().parse_args(argv)
    return EtlConfig(
        products_input=args.input_products,
        products_output=args.output,
        reviews_input=args.input_reviews,
        output_format=args.output_format,
        master=args.master,
        write_mode=args.write_mode,
    )


def create_spark_session(config: EtlConfig):
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(config.app_name)
    if config.master:
        builder = builder.master(config.master)

    if config.output_format == "delta":
        builder = (
            builder.config(
                "spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension",
            )
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        )
        try:
            from delta import configure_spark_with_delta_pip

            return configure_spark_with_delta_pip(builder).getOrCreate()
        except ImportError:
            return builder.getOrCreate()

    return builder.getOrCreate()


def read_dataset(spark, path: str):
    data_format = infer_format(path)
    reader = spark.read
    if data_format == "csv":
        return reader.option("header", True).option("inferSchema", True).csv(path)
    if data_format == "parquet":
        return reader.parquet(path)
    if data_format == "delta":
        return reader.format("delta").load(path)
    try:
        return reader.option("multiLine", False).json(path)
    except Exception as exc:
        if "COLUMN_ALREADY_EXISTS" not in str(exc):
            raise
        return read_projected_json_dataset(spark, path)


def raw_amazon_reviews_2023_schema():
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    return StructType(
        [
            StructField("parent_asin", StringType(), True),
            StructField("asin", StringType(), True),
            StructField("product_id", StringType(), True),
            StructField("id", StringType(), True),
            StructField("title", StringType(), True),
            StructField("name", StringType(), True),
            StructField("main_category", StringType(), True),
            StructField("category", StringType(), True),
            StructField("categories", ArrayType(StringType()), True),
            StructField("main_cat", StringType(), True),
            StructField("description", ArrayType(StringType()), True),
            StructField("features", ArrayType(StringType()), True),
            StructField("feature", ArrayType(StringType()), True),
            StructField("store", StringType(), True),
            StructField("brand", StringType(), True),
            StructField("manufacturer", StringType(), True),
            StructField("price", StringType(), True),
            StructField("list_price", StringType(), True),
            StructField("avg_rating", DoubleType(), True),
            StructField("average_rating", DoubleType(), True),
            StructField("rating", DoubleType(), True),
            StructField("overall", DoubleType(), True),
            StructField("stars", DoubleType(), True),
            StructField("rating_number", LongType(), True),
            StructField("text", StringType(), True),
            StructField("timestamp", LongType(), True),
            StructField("verified_purchase", BooleanType(), True),
        ]
    )


def read_projected_json_dataset(spark, path: str):
    from pyspark.sql.functions import col, from_json

    schema = raw_amazon_reviews_2023_schema()
    return (
        spark.read.text(path)
        .select(from_json(col("value"), schema).alias("record"))
        .select("record.*")
    )


def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    available = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in available:
            return available[candidate.lower()]
    return None


def column_or_default(df, candidates: Sequence[str], default_value=None):
    from pyspark.sql.functions import col, lit

    column_name = first_existing(df.columns, candidates)
    return col(column_name) if column_name else lit(default_value)


def coalesced_column_or_default(df, candidates: Sequence[str], default_value=None):
    from pyspark.sql.functions import coalesce, col, lit

    available = {column.lower(): column for column in df.columns}
    columns = [
        col(available[candidate.lower()])
        for candidate in candidates
        if candidate.lower() in available
    ]
    return coalesce(*columns) if columns else lit(default_value)


def clean_text(column_expression, fallback: str):
    from pyspark.sql.functions import coalesce, lit, regexp_replace, trim

    return trim(
        regexp_replace(
            coalesce(column_expression.cast("string"), lit(fallback)), r"\s+", " "
        )
    )


def parse_price(column_expression):
    from pyspark.sql.functions import regexp_extract

    return regexp_extract(
        column_expression.cast("string"), r"([0-9]+(?:\.[0-9]+)?)", 1
    ).cast("double")


def normalize_products(products_df):
    from pyspark.sql.functions import col, concat_ws, current_timestamp, lit, when

    description_column = column_or_default(
        products_df,
        ("description", "details", "feature", "features"),
        "No description available.",
    )
    category_column = column_or_default(
        products_df,
        ("main_category", "category", "categories", "main_cat"),
        "Uncategorized",
    )
    price_column = coalesced_column_or_default(
        products_df, ("price", "list_price"), None
    )
    rating_column = coalesced_column_or_default(
        products_df,
        ("avg_rating", "average_rating", "rating", "overall"),
        None,
    )

    normalized = products_df.select(
        clean_text(
            column_or_default(products_df, ("parent_asin", "asin", "product_id", "id")),
            "",
        ).alias("product_id"),
        clean_text(
            column_or_default(products_df, ("title", "name"), "Unknown"), "Unknown"
        ).alias("title"),
        clean_text(description_column, "No description available.").alias(
            "description"
        ),
        clean_text(
            column_or_default(
                products_df, ("store", "brand", "manufacturer"), "Unknown"
            ),
            "Unknown",
        ).alias("brand"),
        clean_text(concat_ws(" > ", category_column), "Uncategorized").alias(
            "category"
        ),
        parse_price(price_column).alias("price"),
        rating_column.cast("double").alias("avg_rating"),
        current_timestamp().alias("ingested_at"),
    )

    valid_products = normalized.filter((col("product_id") != "") & (col("title") != ""))

    return valid_products.withColumn(
        "price_tier",
        when(col("price").isNull(), lit("Unknown"))
        .when(col("price") < 20, lit("Budget"))
        .when(col("price") < 100, lit("Mid-range"))
        .otherwise(lit("Premium")),
    )


def aggregate_reviews(reviews_df):
    from pyspark.sql.functions import avg, col, count

    product_id_column = column_or_default(
        reviews_df, ("parent_asin", "asin", "product_id")
    )
    rating_column = column_or_default(reviews_df, ("overall", "rating", "stars"))

    return (
        reviews_df.select(
            clean_text(product_id_column, "").alias("product_id"),
            rating_column.cast("double").alias("rating"),
        )
        .filter((col("product_id") != "") & col("rating").isNotNull())
        .groupBy("product_id")
        .agg(avg("rating").alias("review_avg_rating"), count("*").alias("review_count"))
    )


def enrich_with_reviews(products_df, reviews_df=None):
    from pyspark.sql.functions import coalesce, col, lit, round as spark_round

    if reviews_df is None:
        return (
            products_df.withColumn("avg_rating", spark_round(col("avg_rating"), 2))
            .withColumn("review_count", lit(0))
            .select(
                "product_id",
                "title",
                "description",
                "brand",
                "category",
                "price",
                "price_tier",
                "avg_rating",
                "review_count",
                "ingested_at",
            )
        )

    review_features = aggregate_reviews(reviews_df)
    joined = products_df.join(review_features, on="product_id", how="left")
    return (
        joined.withColumn(
            "avg_rating",
            spark_round(coalesce(col("review_avg_rating"), col("avg_rating")), 2),
        )
        .withColumn("review_count", coalesce(col("review_count"), lit(0)))
        .drop("review_avg_rating")
        .select(
            "product_id",
            "title",
            "description",
            "brand",
            "category",
            "price",
            "price_tier",
            "avg_rating",
            "review_count",
            "ingested_at",
        )
    )


def write_dataset(df, config: EtlConfig) -> None:
    writer = df.write.mode(config.write_mode)
    if config.output_format == "csv":
        writer.option("header", True).csv(config.products_output)
    elif config.output_format == "parquet":
        writer.parquet(config.products_output)
    else:
        writer.format(config.output_format).save(config.products_output)


def run_etl(config: EtlConfig) -> int:
    print("Starting SmartShop ETL job:")
    print(f"  Input products: {config.products_input}")
    print(f"  Input reviews: {config.reviews_input or '(none)'}")
    print(f"  Output: {config.products_output}")
    print(f"  Output format: {config.output_format}")
    print(f"  Spark master: {config.master or '(default)'}")

    validate_windows_local_spark(config)

    spark = create_spark_session(config)
    try:
        products_df = read_dataset(spark, config.products_input)
        reviews_df = (
            read_dataset(spark, config.reviews_input) if config.reviews_input else None
        )

        processed_df = enrich_with_reviews(normalize_products(products_df), reviews_df)
        write_dataset(processed_df, config)

        row_count = processed_df.count()
        print(
            "SmartShop ETL completed: "
            f"{row_count} products written to {config.products_output} "
            f"as {config.output_format}."
        )
        processed_df.show(5, truncate=False)
        processed_df.printSchema()
        return row_count
    finally:
        spark.stop()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_etl(parse_args(argv))
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
