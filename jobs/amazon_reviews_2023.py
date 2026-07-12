"""Materialize Amazon Reviews 2023 data from Hugging Face for SmartShop ETL.

The Spark ETL works best with concrete JSONL/Parquet paths. This helper keeps
Hugging Face access as a separate ingest step, then writes category-scoped files
that the existing pipeline can process locally or copy to Databricks/DBFS.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_CATEGORY = "All_Beauty"
DEFAULT_CATEGORY_SPEC = "all"
ALL_CATEGORIES = (
    "All_Beauty",
    "Toys_and_Games",
    "Cell_Phones_and_Accessories",
    "Industrial_and_Scientific",
    "Gift_Cards",
    "Musical_Instruments",
    "Electronics",
    "Handmade_Products",
    "Arts_Crafts_and_Sewing",
    "Baby_Products",
    "Health_and_Household",
    "Office_Products",
    "Digital_Music",
    "Grocery_and_Gourmet_Food",
    "Sports_and_Outdoors",
    "Home_and_Kitchen",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Pet_Supplies",
    "Video_Games",
    "Kindle_Store",
    "Clothing_Shoes_and_Jewelry",
    "Patio_Lawn_and_Garden",
    "Unknown",
    "Books",
    "Automotive",
    "CDs_and_Vinyl",
    "Beauty_and_Personal_Care",
    "Amazon_Fashion",
    "Magazine_Subscriptions",
    "Software",
    "Health_and_Personal_Care",
    "Appliances",
    "Movies_and_TV",
)
DEFAULT_OUTPUT_DIR = "data/raw/amazon_reviews_2023"
DEFAULT_MAX_PRODUCTS = 5_000
DEFAULT_MAX_REVIEWS = 20_000


@dataclass(frozen=True)
class AmazonReviews2023Config:
    dataset_name: str = DEFAULT_DATASET_NAME
    categories: tuple[str, ...] = ALL_CATEGORIES
    category_spec: str = DEFAULT_CATEGORY_SPEC
    output_dir: str = DEFAULT_OUTPUT_DIR
    max_products: int | None = DEFAULT_MAX_PRODUCTS
    max_reviews: int | None = DEFAULT_MAX_REVIEWS

    @property
    def category(self) -> str:
        return self.categories[0]

    @property
    def meta_config(self) -> str:
        return f"raw_meta_{self.category}"

    @property
    def review_config(self) -> str:
        return f"raw_review_{self.category}"

    def meta_hub_path(self, category: str) -> str:
        return f"raw/meta_categories/meta_{category}.jsonl"

    def reviews_hub_path(self, category: str) -> str:
        return f"raw/review_categories/{category}.jsonl"

    def category_output_dir(self, category: str) -> Path:
        return Path(self.output_dir) / category

    def meta_output_path(self, category: str) -> Path:
        return self.category_output_dir(category) / "meta.jsonl"

    def reviews_output_path(self, category: str) -> Path:
        return self.category_output_dir(category) / "reviews.jsonl"

    @property
    def combined_output_dir(self) -> Path:
        return Path(self.output_dir) / "combined"

    @property
    def combined_meta_output_path(self) -> Path:
        return self.combined_output_dir / "meta.jsonl"

    @property
    def combined_reviews_output_path(self) -> Path:
        return self.combined_output_dir / "reviews.jsonl"


def env_int(name: str, fallback: int | None) -> int | None:
    value = os.getenv(name)
    if value in (None, ""):
        return fallback
    if value.strip().lower() in {"none", "all", "full", "0"}:
        return None
    return int(value)


def parse_categories(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return ALL_CATEGORIES

    normalized = value.strip()
    if normalized.lower() in {"all", "*"}:
        return ALL_CATEGORIES

    categories = []
    seen = set()
    for category in normalized.replace(";", ",").split(","):
        category = category.strip()
        if category and category not in seen:
            categories.append(category)
            seen.add(category)

    if not categories:
        return ALL_CATEGORIES

    unknown_categories = sorted(set(categories) - set(ALL_CATEGORIES))
    if unknown_categories:
        raise ValueError(
            "Unknown Amazon Reviews 2023 categories: "
            f"{', '.join(unknown_categories)}. "
            "Use --categories all or one of: "
            f"{', '.join(ALL_CATEGORIES)}."
        )

    return tuple(categories)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews 2023 category data from Hugging Face."
    )
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("SMARTSHOP_HF_DATASET_NAME", DEFAULT_DATASET_NAME),
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Legacy single category option, for example All_Beauty. "
            "Prefer --categories for multi-category ingestion."
        ),
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=(
            "Comma-separated Amazon Reviews 2023 categories, or 'all' for every "
            "category listed in all_categories.txt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("SMARTSHOP_RAW_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=env_int("SMARTSHOP_MAX_PRODUCTS", DEFAULT_MAX_PRODUCTS),
        help="Maximum metadata rows to write. Use --full to disable limits.",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=env_int("SMARTSHOP_MAX_REVIEWS", DEFAULT_MAX_REVIEWS),
        help="Maximum review rows to write. Use --full to disable limits.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Write the full selected category without row limits.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AmazonReviews2023Config:
    args = build_parser().parse_args(argv)
    max_products = None if args.full else args.max_products
    max_reviews = None if args.full else args.max_reviews
    category_spec = (
        args.categories
        or args.category
        or os.getenv("SMARTSHOP_AMAZON_CATEGORIES")
        or os.getenv("SMARTSHOP_AMAZON_CATEGORY")
        or DEFAULT_CATEGORY_SPEC
    )
    try:
        categories = parse_categories(category_spec)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    return AmazonReviews2023Config(
        dataset_name=args.dataset_name,
        categories=categories,
        category_spec=category_spec,
        output_dir=args.output_dir,
        max_products=max_products,
        max_reviews=max_reviews,
    )


def hub_url(config: AmazonReviews2023Config, hub_path: str) -> str:
    try:
        from huggingface_hub import hf_hub_url
    except ImportError as exc:
        raise RuntimeError(
            "The huggingface-hub package is required. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    return hf_hub_url(
        repo_id=config.dataset_name,
        filename=hub_path,
        repo_type="dataset",
    )


def is_all_categories_spec(category_spec: str) -> bool:
    return category_spec.strip().lower() in {"all", "*"}


def discover_hub_categories(config: AmazonReviews2023Config) -> tuple[str, ...]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "The huggingface-hub package is required. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    files = list_repo_files(config.dataset_name, repo_type="dataset")
    meta_prefix = "raw/meta_categories/meta_"
    review_prefix = "raw/review_categories/"
    meta_suffix = ".jsonl"
    review_suffix = ".jsonl"

    meta_categories = {
        path[len(meta_prefix) : -len(meta_suffix)]
        for path in files
        if path.startswith(meta_prefix) and path.endswith(meta_suffix)
    }
    review_categories = {
        path[len(review_prefix) : -len(review_suffix)]
        for path in files
        if path.startswith(review_prefix) and path.endswith(review_suffix)
    }
    available_categories = meta_categories & review_categories
    ordered_categories = [
        category for category in ALL_CATEGORIES if category in available_categories
    ]

    extra_categories = sorted(available_categories - set(ALL_CATEGORIES))
    return tuple(ordered_categories + extra_categories)


def limit_records(records: Iterable[str], max_records: int | None) -> Iterable[str]:
    if max_records is None:
        return records
    for index, record in enumerate(records):
        if index >= max_records:
            break
        yield record


def stream_hub_jsonl_lines(
    config: AmazonReviews2023Config, hub_path: str
) -> Iterable[str]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The requests package is required. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    url = hub_url(config, hub_path)

    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(10, 120),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if isinstance(line, bytes):
                        line = line.decode(response.encoding or "utf-8")
                    yield line
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not stream {hub_path} from Hugging Face: {exc}"
        ) from exc


def write_jsonl(records: Iterable[str], output_paths: Sequence[Path], mode: str) -> int:
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    handles = [output_path.open(mode, encoding="utf-8") for output_path in output_paths]
    try:
        for line in records:
            for handle in handles:
                handle.write(line)
                handle.write("\n")
            count += 1
    finally:
        for handle in handles:
            handle.close()
    return count


def materialize(config: AmazonReviews2023Config) -> dict[str, Any]:
    categories = (
        discover_hub_categories(config)
        if is_all_categories_spec(config.category_spec)
        else config.categories
    )
    if not categories:
        raise RuntimeError(
            "No Amazon Reviews 2023 categories were found with both metadata and "
            "review JSONL files."
        )

    print("Materializing Amazon Reviews 2023 from Hugging Face:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Category source: {config.category_spec}")
    print(f"  Categories: {', '.join(categories)}")
    print(f"  Combined metadata: {config.combined_meta_output_path}")
    print(f"  Combined reviews: {config.combined_reviews_output_path}")

    total_product_count = 0
    total_review_count = 0
    category_counts = {}

    for index, category in enumerate(categories):
        mode = "w" if index == 0 else "a"
        meta_hub_path = config.meta_hub_path(category)
        reviews_hub_path = config.reviews_hub_path(category)
        meta_output_path = config.meta_output_path(category)
        reviews_output_path = config.reviews_output_path(category)

        print(f"  [{category}] Metadata path: {meta_hub_path}")
        product_count = write_jsonl(
            limit_records(
                stream_hub_jsonl_lines(config, meta_hub_path),
                config.max_products,
            ),
            (meta_output_path, config.combined_meta_output_path),
            mode=mode,
        )

        print(f"  [{category}] Review path: {reviews_hub_path}")
        review_count = write_jsonl(
            limit_records(
                stream_hub_jsonl_lines(config, reviews_hub_path),
                config.max_reviews,
            ),
            (reviews_output_path, config.combined_reviews_output_path),
            mode=mode,
        )

        total_product_count += product_count
        total_review_count += review_count
        category_counts[category] = {
            "product_count": product_count,
            "review_count": review_count,
        }

    result = {
        "products_path": str(config.combined_meta_output_path),
        "reviews_path": str(config.combined_reviews_output_path),
        "product_count": total_product_count,
        "review_count": total_review_count,
        "category_count": len(categories),
        "category_counts": category_counts,
    }
    print(
        "Amazon Reviews 2023 materialized: "
        f"{total_product_count} products and {total_review_count} reviews "
        f"across {len(categories)} categories."
    )
    print("Next ETL command:")
    print(
        "python -m jobs.spark_etl "
        f"--input-products {config.combined_meta_output_path} "
        f"--input-reviews {config.combined_reviews_output_path} "
        "--output-path data/processed/amazon_reviews_2023_flow_smoke "
        "--output-format parquet "
        "--master local[*]"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        materialize(parse_args(argv))
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    return 0


if __name__ == "__main__":
    main()
