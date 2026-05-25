import csv
import joblib
import pandas as pd

from pathlib import Path
from datetime import date
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/phase4_elo_features.csv")
MODEL_PATH = Path("backend/ml/artifacts/logreg_phase4_elo.joblib")

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


def evaluate_predictions(y_test, y_pred, title: str) -> tuple[float, dict]:
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"=== {title} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy, report


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

    accuracy, report = evaluate_predictions(
        y_test=y_test,
        y_pred=y_pred,
        title="Chronological Split Evaluation",
    )

    log_experiment_result(
        experiment_id="phase4_logreg_elo",
        dataset_path=DATA_PATH,
        model_path=MODEL_PATH,
        model_name="LogisticRegression",
        features=features,
        train_df=train_df,
        test_df=test_df,
        accuracy=accuracy,
        report=report,
        notes="Phase 4 Elo features baseline",
    )

def log_experiment_result(
    experiment_id: str,
    dataset_path: Path,
    model_path: Path,
    model_name: str,
    features: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    accuracy: float,
    report: dict,
    notes: str,
) -> None:
    results_path = Path("experiments/results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment_id": experiment_id,
        "date": str(date.today()),
        "dataset": str(dataset_path),
        "model": model_name,
        "features": "|".join(features),
        "split_type": "chronological",
        "train_start": str(train_df[DATE_COLUMN].min().date()),
        "train_end": str(train_df[DATE_COLUMN].max().date()),
        "test_start": str(test_df[DATE_COLUMN].min().date()),
        "test_end": str(test_df[DATE_COLUMN].max().date()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": round(accuracy, 4),
        "precision_0": round(report["0"]["precision"], 4),
        "recall_0": round(report["0"]["recall"], 4),
        "f1_0": round(report["0"]["f1-score"], 4),
        "precision_1": round(report["1"]["precision"], 4),
        "recall_1": round(report["1"]["recall"], 4),
        "f1_1": round(report["1"]["f1-score"], 4),
        "artifact_path": str(model_path),
        "notes": notes,
    }

    write_header = not results_path.exists()

    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    print(f"\nSaved experiment result to: {results_path}")

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