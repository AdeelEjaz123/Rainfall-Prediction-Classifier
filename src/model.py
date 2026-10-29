"""
Pipeline definitions and hyperparameter grids for classification models.
"""

from typing import List, Dict, Any, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Construct a ColumnTransformer that standardizes numeric features

    and one-hot encodes categorical features.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def create_random_forest_pipeline(
    preprocessor: ColumnTransformer,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Build Random Forest pipeline and hyperparameter search space."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=random_state)),
    ])

    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }

    return pipeline, param_grid


def create_logistic_regression_pipeline(
    preprocessor: ColumnTransformer,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Build Logistic Regression pipeline and hyperparameter search space."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=random_state, max_iter=1000)),
    ])

    param_grid = {
        "classifier__solver": ["liblinear"],
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"],
    }

    return pipeline, param_grid
