"""
Model training and hyperparameter optimization module.
"""

from typing import Tuple, Dict, Any, Optional
import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_loader import (
    load_raw_data,
    preprocess_data,
    get_feature_lists,
    split_data,
    DEFAULT_DATA_URL,
)
from src.model import (
    build_preprocessor,
    create_random_forest_pipeline,
    create_logistic_regression_pipeline,
)


def train_grid_search(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    random_state: int = 42,
) -> GridSearchCV:
    """Run stratified cross-validated Grid Search over pipeline hyperparameters."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1,
        n_jobs=n_jobs,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def train_model(
    model_type: str = "rf",
    data_source: str = DEFAULT_DATA_URL,
    save_path: Optional[str] = None,
) -> Tuple[GridSearchCV, pd.DataFrame, pd.Series, list, list]:
    """Complete end-to-end training entry point.

    Args:
        model_type: 'rf' for Random Forest, 'lr' for Logistic Regression.
        data_source: URL or file path to raw data CSV.
        save_path: Optional file path to serialize best estimator.

    Returns:
        (grid_search, X_test, y_test, numeric_features, categorical_features)
    """
    print(f"Loading data from: {data_source}...")
    df_raw = load_raw_data(data_source)
    df_clean = preprocess_data(df_raw)

    X_train, X_test, y_train, y_test = split_data(df_clean)
    num_cols, cat_cols = get_feature_lists(X_train)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    if model_type.lower() == "rf":
        pipeline, param_grid = create_random_forest_pipeline(preprocessor)
        print("Starting GridSearchCV for Random Forest Classifier...")
    elif model_type.lower() == "lr":
        pipeline, param_grid = create_logistic_regression_pipeline(preprocessor)
        print("Starting GridSearchCV for Logistic Regression Classifier...")
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'rf' or 'lr'.")

    grid_search = train_grid_search(pipeline, param_grid, X_train, y_train)

    print("\nBest Parameters:", grid_search.best_params_)
    print(f"Best 5-Fold CV Score: {grid_search.best_score_:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(grid_search.best_estimator_, save_path)
        print(f"Model successfully saved to: {save_path}")

    return grid_search, X_test, y_test, num_cols, cat_cols


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Rainfall Classification Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "lr"],
        help="Model type: 'rf' (Random Forest) or 'lr' (Logistic Regression)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="models/best_model.joblib",
        help="Path to save serialized model",
    )
    args = parser.parse_args()

    train_model(model_type=args.model, save_path=args.save)

