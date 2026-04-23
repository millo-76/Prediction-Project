from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/phase2_best_features.csv")
MODEL_PATH = Path("backend/ml/artifacts/logreg_phase2_rolling.joblib")
TARGET = "blue_side_win"
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


def evaluate_model(df: pd.DataFrame, model, features: list[str]) -> None:
    missing = [col for col in features + [TARGET] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    X = df[features]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("=== Model Evaluation ===")
    print(f"Artifact: {MODEL_PATH}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Features: {features}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Random state: {RANDOM_STATE}")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def main() -> None:
    df = load_dataset(DATA_PATH)
    model, features = load_artifact(MODEL_PATH)
    evaluate_model(df, model, features)


if __name__ == "__main__":
    main()