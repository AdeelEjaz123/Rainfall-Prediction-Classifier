"""
Unit tests for data loading, preprocessing, and pipeline modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import date_to_season, preprocess_data, get_feature_lists, split_data
from src.model import build_preprocessor, create_random_forest_pipeline
from src.predict import predict_rainfall


@pytest.fixture
def sample_df():
    """Create a minimal synthetic weather dataframe matching BOM schema."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = {
        "Date": dates,
        "Location": ["Melbourne", "MelbourneAirport", "Watsonia", "Sydney"] * 2 + ["Melbourne", "Watsonia"],
        "MinTemp": np.random.uniform(10, 20, 10),
        "MaxTemp": np.random.uniform(20, 35, 10),
        "Rainfall": np.random.uniform(0, 10, 10),
        "Evaporation": np.random.uniform(2, 8, 10),
        "Sunshine": np.random.uniform(0, 12, 10),
        "WindGustDir": ["SSW", "N", "W", "S"] * 2 + ["E", "W"],
        "WindGustSpeed": np.random.uniform(20, 60, 10),
        "WindDir9am": ["S", "N", "W", "E"] * 2 + ["S", "N"],
        "WindDir3pm": ["SSW", "NNW", "W", "SE"] * 2 + ["SW", "NE"],
        "WindSpeed9am": np.random.uniform(5, 25, 10),
        "WindSpeed3pm": np.random.uniform(10, 30, 10),
        "Humidity9am": np.random.uniform(40, 90, 10),
        "Humidity3pm": np.random.uniform(30, 80, 10),
        "Pressure9am": np.random.uniform(1010, 1025, 10),
        "Pressure3pm": np.random.uniform(1008, 1020, 10),
        "Cloud9am": np.random.uniform(0, 8, 10),
        "Cloud3pm": np.random.uniform(0, 8, 10),
        "Temp9am": np.random.uniform(12, 22, 10),
        "Temp3pm": np.random.uniform(18, 30, 10),
        "RainToday": ["No", "Yes"] * 5,
        "RainTomorrow": ["No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "Yes"],
    }
    return pd.DataFrame(data)


def test_date_to_season():
    """Verify Australian meteorological season mapping."""
    summer_date = pd.Timestamp("2021-01-15")
    autumn_date = pd.Timestamp("2021-04-10")
    winter_date = pd.Timestamp("2021-07-20")
    spring_date = pd.Timestamp("2021-10-05")

    assert date_to_season(summer_date) == "Summer"
    assert date_to_season(autumn_date) == "Autumn"
    assert date_to_season(winter_date) == "Winter"
    assert date_to_season(spring_date) == "Spring"


def test_preprocess_data(sample_df):
    """Test data cleaning, column renaming, and location filtering."""
    df_clean = preprocess_data(sample_df, locations=["Melbourne", "MelbourneAirport", "Watsonia"])

    # Ensure Sydney is excluded
    assert "Sydney" not in df_clean["Location"].values
    # Check column renaming
    assert "RainYesterday" in df_clean.columns
    assert "RainToday" in df_clean.columns
    assert "RainTomorrow" not in df_clean.columns
    # Check Season feature created and Date dropped
    assert "Season" in df_clean.columns
    assert "Date" not in df_clean.columns


def test_get_feature_lists(sample_df):
    """Test feature division into numeric and categorical subsets."""
    df_clean = preprocess_data(sample_df)
    X = df_clean.drop(columns=["RainToday"])
    num_cols, cat_cols = get_feature_lists(X)

    assert "Humidity3pm" in num_cols
    assert "Pressure9am" in num_cols
    assert "Location" in cat_cols
    assert "Season" in cat_cols


def test_build_preprocessor_and_pipeline(sample_df):
    """Test that pipeline transforms input features without errors."""
    df_clean = preprocess_data(sample_df)
    X = df_clean.drop(columns=["RainToday"])
    y = df_clean["RainToday"]
    num_cols, cat_cols = get_feature_lists(X)

    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipeline, param_grid = create_random_forest_pipeline(preprocessor)

    # Fit pipeline
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({"No", "Yes"})
