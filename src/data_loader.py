"""
Data loading and preprocessing module for the Rainfall Prediction Classifier.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
)

TARGET_LOCATIONS = ["Melbourne", "MelbourneAirport", "Watsonia"]


def date_to_season(date: pd.Timestamp) -> str:
    """Map a datetime object to an Australian meteorological season.

    - Summer: December, January, February
    - Autumn: March, April, May
    - Winter: June, July, August
    - Spring: September, October, November
    """
    month = date.month
    if month in (12, 1, 2):
        return "Summer"
    elif month in (3, 4, 5):
        return "Autumn"
    elif month in (6, 7, 8):
        return "Winter"
    elif month in (9, 10, 11):
        return "Spring"
    return "Unknown"


def load_raw_data(data_source: str = DEFAULT_DATA_URL) -> pd.DataFrame:
    """Load the raw weather observations dataset."""
    return pd.read_csv(data_source)


def preprocess_data(
    df: pd.DataFrame,
    locations: List[str] = TARGET_LOCATIONS,
) -> pd.DataFrame:
    """Clean and prepare the dataset:

    1. Drop missing observations.
    2. Align columns: 'RainToday' becomes 'RainYesterday',
       'RainTomorrow' becomes 'RainToday' (the target).
    3. Filter observations to targeted geographic locations.
    4. Derive the 'Season' feature from 'Date' and drop 'Date'.
    """
    # Remove records with NaN values
    df_clean = df.dropna().copy()

    # Rename columns to reflect predictive framing
    df_clean = df_clean.rename(
        columns={"RainToday": "RainYesterday", "RainTomorrow": "RainToday"}
    )

    # Filter target locations
    df_clean = df_clean[df_clean["Location"].isin(locations)].copy()

    # Date to Season extraction
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    df_clean["Season"] = df_clean["Date"].apply(date_to_season)
    df_clean = df_clean.drop(columns=["Date"])

    return df_clean


def get_feature_lists(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric (float64) and categorical (object/category) column names."""
    numeric_features = X.select_dtypes(include=["float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numeric_features, categorical_features


def split_data(
    df: pd.DataFrame,
    target_column: str = "RainToday",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into stratified train and test partitions."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
