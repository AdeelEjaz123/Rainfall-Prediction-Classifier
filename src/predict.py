"""
Inference module for predicting next-day rainfall on single or batch observations.
"""

from typing import Dict, Any, Union
import pandas as pd
import joblib


SAMPLE_WEATHER_RECORD = {
    "Location": "MelbourneAirport",
    "MinTemp": 11.2,
    "MaxTemp": 18.2,
    "Rainfall": 5.4,
    "Evaporation": 4.8,
    "Sunshine": 3.2,
    "WindGustDir": "SSW",
    "WindGustSpeed": 48.0,
    "WindDir9am": "S",
    "WindDir3pm": "SSW",
    "WindSpeed9am": 19.0,
    "WindSpeed3pm": 26.0,
    "Humidity9am": 82.0,
    "Humidity3pm": 71.0,
    "Pressure9am": 1014.2,
    "Pressure3pm": 1012.9,
    "Cloud9am": 7.0,
    "Cloud3pm": 7.0,
    "Temp9am": 13.8,
    "Temp3pm": 16.5,
    "RainYesterday": "Yes",
    "Season": "Spring",
}


def predict_rainfall(
    model_or_path: Union[str, Any],
    observation: Union[Dict[str, Any], pd.DataFrame],
) -> Dict[str, Any]:
    """Predict rainfall probability and class for an observation.

    Args:
        model_or_path: Loaded sklearn Pipeline or file path to joblib model.
        observation: Dictionary of meteorological features or DataFrame.

    Returns:
        Dictionary containing predicted class ('Yes' or 'No') and probabilities.
    """
    if isinstance(model_or_path, str):
        model = joblib.load(model_or_path)
    else:
        model = model_or_path

    if isinstance(observation, dict):
        df_input = pd.DataFrame([observation])
    else:
        df_input = observation.copy()

    prediction = model.predict(df_input)[0]
    probabilities = model.predict_proba(df_input)[0]
    classes = model.classes_

    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

    return {
        "prediction": prediction,
        "rain_predicted": bool(prediction == "Yes"),
        "confidence": float(max(probabilities)),
        "class_probabilities": prob_dict,
    }


if __name__ == "__main__":
    import os
    from src.train import train_model

    default_model_path = "models/random_forest_model.joblib"

    if not os.path.exists(default_model_path):
        print(f"Serialized model not found at {default_model_path}. Training a quick model...")
        grid, _, _, _, _ = train_model(model_type="rf", save_path=default_model_path)
        model = grid.best_estimator_
    else:
        model = default_model_path

    print("\nRunning inference on sample meteorological observation:")
    result = predict_rainfall(model, SAMPLE_WEATHER_RECORD)
    print("\nPrediction Result:")
    print(f"  Prediction:          {'RAIN (Yes)' if result['rain_predicted'] else 'NO RAIN (No)'}")
    print(f"  Rain Probability:    {result['class_probabilities'].get('Yes', 0.0):.2%}")
    print(f"  Confidence:          {result['confidence']:.2%}")
