from pathlib import Path
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("backend/ml/artifacts/logreg_baseline.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    data = joblib.load(MODEL_PATH)
    return data["model"], data["features"]


def predict(features: dict) -> dict:
    model, feature_order = load_model()

    # Create DataFrame with correct column names and order
    input_df = pd.DataFrame([features])[feature_order]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "predicted_class": int(prediction),
        "blue_win_probability": f"{probabilities[1] * 100:.2f}%",
        "red_win_probability": f"{probabilities[0] * 100:.2f}%"
    }


if __name__ == "__main__":
    # Example test input
    test_input = {
        "wr_diff": 0.15,
        "blue_team_games": 15,
        "red_team_games": 17
    }

    result = predict(test_input)
    print(result)