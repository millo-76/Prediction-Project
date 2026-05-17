"""Train and evaluate the current best logistic regression model.

Goal:
Train a logistic regression classifier on the best-performing Phase 2
feature set, report evaluation metrics, inspect coefficients, and save
the trained model artifact.

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

from evaluate import DATE_COLUMN, TEST_SIZE


DATA_PATH = Path("data/processed/phase3_rolling_features.csv")
MODEL_PATH = Path("backend/ml/artifacts/logreg_phase3_rolling.joblib")


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
    # Train the model, print diagnostics, and persist the artifact.
    df = df.copy()

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = int(len(df) * (1 - TEST_SIZE))

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    coefficients = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_[0]
    })

    print("Coefficients:")
    print(coefficients)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "data_path": str(DATA_PATH),
        "model_name": "logreg_phase3_rolling",
        "split_strategy": "chronological",
        "test_size": TEST_SIZE,
    }, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

def main() -> None:
    # Run the end-to-end training workflow.
    print("=== Phase 3 Rolling Features Logistic Regression Model ===\n")
    df = load_data(DATA_PATH)
    train_model(df)


if __name__ == "__main__":
    main()