"""
Main Entry Point - End-to-End Rainfall Prediction Benchmark and Pipeline Runner.
"""

import argparse
import os
import joblib
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
from src.train import train_grid_search
from src.evaluate import (
    evaluate_classifier,
    get_feature_importances,
    print_evaluation_summary,
)


def run_pipeline(
    data_source: str = DEFAULT_DATA_URL,
    models_dir: str = "models",
    evaluate_both: bool = True,
    selected_model: str = "all",
):
    """Execute end-to-end ML workflow:

    1. Data Ingestion & Preprocessing
    2. Feature Extraction & Stratified Train/Test Split
    3. Pipeline Construction & Hyperparameter Optimization
    4. Model Evaluation & Benchmarking
    5. Feature Importance Extraction
    6. Artifact Serialization
    """
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("RAINFALL PREDICTION CLASSIFIER - WEATHERTECH INC. BENCHMARK")
    print("=" * 70)

    print(f"\n[1/4] Ingesting dataset from: {data_source}")
    df_raw = load_raw_data(data_source)
    print(f"      Raw records: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    print("\n[2/4] Cleaning data & engineering seasonal features...")
    df_clean = preprocess_data(df_raw)
    print(f"      Filtered observations (Melbourne region): {df_clean.shape[0]:,} rows")
    counts = df_clean["RainToday"].value_counts()
    print(f"      Class Balance: No = {counts['No']:,} ({counts['No']/len(df_clean):.1%}), "
          f"Yes = {counts['Yes']:,} ({counts['Yes']/len(df_clean):.1%})")

    X_train, X_test, y_train, y_test = split_data(df_clean)
    num_cols, cat_cols = get_feature_lists(X_train)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    print(f"      Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    models_to_run = []
    if selected_model in ("rf", "all"):
        models_to_run.append("rf")
    if selected_model in ("lr", "all"):
        models_to_run.append("lr")

    eval_results = {}

    print("\n[3/4] Tuning models with Stratified 5-Fold Cross-Validation...")

    if "rf" in models_to_run:
        print("\n--> Training Random Forest Classifier...")
        rf_pipeline, rf_params = create_random_forest_pipeline(preprocessor)
        rf_grid = train_grid_search(rf_pipeline, rf_params, X_train, y_train)
        rf_path = os.path.join(models_dir, "random_forest_model.joblib")
        joblib.dump(rf_grid.best_estimator_, rf_path)
        eval_results["Random Forest"] = evaluate_classifier(rf_grid.best_estimator_, X_test, y_test)
        print_evaluation_summary(eval_results["Random Forest"], "Random Forest Classifier")

        # Feature Importance
        imp_df = get_feature_importances(rf_grid.best_estimator_, num_cols, cat_cols)
        print("\nTop 5 Most Influential Features (Random Forest):")
        for idx, row in imp_df.head(5).iterrows():
            print(f"  {idx+1}. {row['Feature']:<20} : {row['Importance']*100:.2f}%")

    if "lr" in models_to_run:
        print("\n--> Training Logistic Regression Classifier...")
        lr_pipeline, lr_params = create_logistic_regression_pipeline(preprocessor)
        lr_grid = train_grid_search(lr_pipeline, lr_params, X_train, y_train)
        lr_path = os.path.join(models_dir, "logistic_regression_model.joblib")
        joblib.dump(lr_grid.best_estimator_, lr_path)
        eval_results["Logistic Regression"] = evaluate_classifier(lr_grid.best_estimator_, X_test, y_test)
        print_evaluation_summary(eval_results["Logistic Regression"], "Logistic Regression Classifier")

    print("\n[4/4] Summary Comparison:")
    print("-" * 75)
    print(f"{'Model':<25} {'Accuracy':<12} {'Rain Recall':<14} {'Rain Precision':<16} {'F1-Score'}")
    print("-" * 75)
    for model_name, res in eval_results.items():
        print(f"{model_name:<25} {res['accuracy']*100:>6.2f}%     "
              f"{res['true_positive_rate']*100:>6.2f}%        "
              f"{res['precision_rain']*100:>6.2f}%         "
              f"{res['f1_rain']:>6.4f}")
    print("-" * 75)
    print("\nPipeline execution complete! Models saved in directory:", models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rainfall Prediction ML Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "rf", "lr"],
        help="Model to run: 'all', 'rf' (Random Forest), or 'lr' (Logistic Regression)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_URL,
        help="URL or local path to raw weather CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory where serialized models will be saved",
    )
    args = parser.parse_args()

    run_pipeline(
        data_source=args.data,
        models_dir=args.models_dir,
        selected_model=args.model,
    )
