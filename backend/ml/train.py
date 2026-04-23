# 04/16/2026 Version 1.0.0

"""Train and evaluate the baseline logistic regression model.

Goal:
Build a simple baseline classifier from engineered pre-match features,
report evaluation metrics, inspect coefficients, and save the trained model.

Model setup:
- Model: Logistic Regression
- Target: blue_side_win
- Features: wr_diff, blue_team_games, red_team_games, wr_diff_5
- Split: 80/20 train/test with random_state=42
- Metrics: accuracy, precision, recall, f1-score
"""
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib


DATA_PATH = Path("data/processed/phase2_best_features.csv")
MODEL_PATH = Path("backend/ml/artifacts/logreg_phase2_rolling.joblib")


# Baseline feature columns used for training.
FEATURES = [
    "wr_diff",
    "blue_team_games",
    "red_team_games",
    "wr_diff_5"
]

# Binary target column: 1 if blue side won, otherwise 0.
TARGET = "blue_side_win"


def load_data(path: Path) -> pd.DataFrame:
    # Load the engineered dataset from disk.
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def train_model(df: pd.DataFrame) -> None:
    # Train the baseline model, print diagnostics, and persist the artifact.
    # 1) Build feature matrix and target vector.
    X = df[FEATURES]
    y = df[TARGET]

    # 2) Create a reproducible train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 3) Fit logistic regression on the training set.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 4) Evaluate predictions on the held-out test set.
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 5) Print learned coefficients for quick interpretability checks.
    coefficients = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_[0]
    })

    print("Coefficients:")
    print(coefficients)

    # 6) Ensure artifact directory exists, then save the trained model.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURES
    }, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


def main() -> None:
    # Run the end-to-end training workflow.
    print("=== Phase 2 Combined Baseline Model ===\n")
    df = load_data(DATA_PATH)
    train_model(df)


if __name__ == "__main__":
    main()