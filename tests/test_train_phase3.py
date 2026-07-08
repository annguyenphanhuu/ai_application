import pandas as pd
import pytest

from src.train import (
    TrainConfig,
    build_model,
    infer_input_format,
    prepare_training_frame,
    split_training_data,
)


def sample_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": [
                "Wireless Headphones",
                "Office Chair",
                "Green Tea",
                "Running Shoes",
                "Budget Lamp",
                "Kitchen Knife",
            ],
            "description": [
                "Noise cancelling with strong battery life",
                "Ergonomic chair with lumbar support",
                "Organic tea bags with mild flavor",
                "Comfortable shoes for daily runs",
                "Simple desk lamp with dim light",
                "Sharp stainless steel chef knife",
            ],
            "brand": [
                "SoundWave",
                "ErgoFlex",
                "PureLeaf",
                "RunFast",
                "HomeLite",
                "Zelite",
            ],
            "category": [
                "Electronics",
                "Furniture",
                "Grocery",
                "Sports",
                "Home",
                "Kitchen",
            ],
            "price_tier": [
                "Mid-range",
                "Premium",
                "Budget",
                "Mid-range",
                "Budget",
                "Mid-range",
            ],
            "avg_rating": [4.5, 4.0, 3.0, 5.0, 3.5, 4.25],
        }
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("data/products.csv", "csv"),
        ("data/products.jsonl", "json"),
        ("data/products.json", "json"),
        ("data/processed/products_processed", "parquet"),
    ],
)
def test_infer_input_format(path, expected):
    assert infer_input_format(path) == expected


def test_prepare_training_frame_builds_text_and_target():
    text, target = prepare_training_frame(sample_products(), rating_threshold=4.0)

    assert len(text) == 6
    assert target.tolist() == [1, 1, 0, 1, 0, 1]
    assert "Wireless Headphones" in text.iloc[0]
    assert "Mid-range" in text.iloc[0]


def test_prepare_training_frame_requires_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        prepare_training_frame(
            pd.DataFrame({"title": ["Only title"]}), rating_threshold=4.0
        )


def test_split_training_data_keeps_both_classes_when_possible():
    text, target = prepare_training_frame(sample_products(), rating_threshold=4.0)
    config = TrainConfig(test_size=0.34, random_state=7)

    x_train, x_test, y_train, y_test = split_training_data(text, target, config)

    assert len(x_train) == 3
    assert len(x_test) == 3
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_build_model_can_fit_small_dataset():
    text, target = prepare_training_frame(sample_products(), rating_threshold=4.0)
    model = build_model(TrainConfig(max_features=50))

    model.fit(text, target)

    predictions = model.predict(text)
    assert len(predictions) == len(target)
