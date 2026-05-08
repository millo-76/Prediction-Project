from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/phase2_best_features.csv")
MODEL_PATH = Path("backend/ml/artifacts/logreg_phase3_rolling.joblib")

TARGET = "blue_side_win"
DATE_COLUMN = "date"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    artifact = joblib.load(path)
    return artifact["model"], artifact["features"]


def validate_columns(df: pd.DataFrame, features: list[str]) -> None:
    required_columns = features + [TARGET]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")


def evaluate_predictions(y_test, y_pred, title: str) -> None:
    accuracy = accuracy_score(y_test, y_pred)

    print(f"=== {title} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def evaluate_random_split(df: pd.DataFrame, model, features: list[str]) -> None:
    X = df[features]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    y_pred = model.predict(X_test)

    evaluate_predictions(
        y_test=y_test,
        y_pred=y_pred,
        title="Random Split Evaluation",
    )


def evaluate_chronological_split(df: pd.DataFrame, model, features: list[str]) -> None:
    if DATE_COLUMN not in df.columns:
        raise KeyError(
            f"Chronological evaluation requires a '{DATE_COLUMN}' column in the dataset."
        )

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    split_index = int(len(df) * (1 - TEST_SIZE))

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_test = test_df[features]
    y_test = test_df[TARGET]

    y_pred = model.predict(X_test)

    print(f"Train period: {train_df[DATE_COLUMN].min().date()} -> {train_df[DATE_COLUMN].max().date()}")
    print(f"Test period:  {test_df[DATE_COLUMN].min().date()} -> {test_df[DATE_COLUMN].max().date()}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print()

    evaluate_predictions(
        y_test=y_test,
        y_pred=y_pred,
        title="Chronological Split Evaluation",
    )


def main() -> None:
    df = load_dataset(DATA_PATH)
    model, features = load_artifact(MODEL_PATH)

    validate_columns(df, features)

    print("=== Model Evaluation Setup ===")
    print(f"Artifact: {MODEL_PATH}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Features: {features}")
    print(f"Target: {TARGET}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Random state: {RANDOM_STATE}")
    print()

    evaluate_random_split(df, model, features)
    print()
    evaluate_chronological_split(df, model, features)


if __name__ == "__main__":
    main()