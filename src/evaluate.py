"""
Model evaluation, metrics extraction, and diagnostic utilities.
"""

from typing import Dict, Any, List
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classifier(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: List[str] = ["No", "Yes"],
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics on unseen test data."""
    y_pred = estimator.predict(X_test)
    test_accuracy = estimator.score(X_test, y_test)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate / Recall for 'Yes'
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * tpr) / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    results = {
        "accuracy": float(test_accuracy),
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_positive_rate": float(tpr),
        "false_positive_rate": float(fpr),
        "precision_rain": float(precision),
        "recall_rain": float(tpr),
        "f1_rain": float(f1),
        "classification_report_str": report_text,
        "classification_report_dict": report_dict,
    }

    return results


def get_feature_importances(
    fitted_pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """Extract and sort feature importances from a fitted Random Forest pipeline."""
    classifier = fitted_pipeline.named_steps["classifier"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        raise AttributeError("Underlying classifier does not have feature_importances_")

    onehot_step = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(onehot_step.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_names

    importances = classifier.feature_importances_
    df_importance = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return df_importance


def print_evaluation_summary(results: Dict[str, Any], model_name: str = "Classifier"):
    """Pretty-print evaluation metrics."""
    print("=" * 60)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("=" * 60)
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Rain Recall (TPR): {results['true_positive_rate']:.4f} ({results['true_positive_rate']*100:.2f}%)")
    print(f"Rain Precision:    {results['precision_rain']:.4f} ({results['precision_rain']*100:.2f}%)")
    print(f"Rain F1-Score:     {results['f1_rain']:.4f}")
    print("\nConfusion Matrix (Labels: ['No', 'Yes']):")
    print(results["confusion_matrix"])
    print(f"TN: {results['true_negatives']}, FP: {results['false_positives']}, "
          f"FN: {results['false_negatives']}, TP: {results['true_positives']}")
    print("\nFull Classification Report:")
    print(results["classification_report_str"])
    print("=" * 60)
