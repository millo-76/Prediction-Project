# predict.py
# 04/16/2026 Version 1.0.0

from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("backend/ml/artifacts/logreg_phase2_rolling.joblib")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    data = joblib.load(MODEL_PATH)
    return data["model"], data["features"]


def predict(features: dict) -> dict:
    model, feature_order = load_model()

    input_df = pd.DataFrame([features])[feature_order]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return {
        "predicted_class": int(prediction),
        "predicted_winner": "blue" if int(prediction) == 1 else "red",        
        "blue_win_probability": float(probabilities[1]),
        "red_win_probability": float(probabilities[0]),
    }


if __name__ == "__main__":
    test_input = {
        "wr_diff": 0.15,
        "blue_team_games": 10,
        "red_team_games": 12,
        "wr_diff_5": 0.05
    }

    result = predict(test_input)
    print(result)