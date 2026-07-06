import csv
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


PREDICTION_LOG_PATH = Path("data/predictions/prediction_log.csv")


FIELDNAMES = [
    "prediction_id",
    "timestamp_utc",
    "blue_team",
    "red_team",
    "predicted_winner",
    "blue_win_probability",
    "red_win_probability",
    "model_name",
    "model_artifact",
    "league",
    "tournament",
    "actual_winner",
    "correct",
]


def _ensure_prediction_log_schema() -> None:
    if not PREDICTION_LOG_PATH.exists():
        return

    df = pd.read_csv(PREDICTION_LOG_PATH)
    changed = False

    for column in FIELDNAMES:
        if column not in df.columns:
            df[column] = ""
            changed = True

    if not changed:
        return

    # Keep known schema columns first and preserve unknown trailing columns.
    ordered_columns = FIELDNAMES + [col for col in df.columns if col not in FIELDNAMES]
    df = df[ordered_columns]
    df.to_csv(PREDICTION_LOG_PATH, index=False)


def _next_prediction_id() -> str:
    if not PREDICTION_LOG_PATH.exists():
        return "PRED-00001"

    max_id = 0

    with PREDICTION_LOG_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            value = (row.get("prediction_id") or "").strip()

            if not value.startswith("PRED-"):
                continue

            suffix = value[5:]

            if not suffix.isdigit():
                continue

            max_id = max(max_id, int(suffix))

    return f"PRED-{max_id + 1:05d}"


def log_prediction(
    blue_team: str,
    red_team: str,
    predicted_winner: str,
    blue_win_probability: float,
    red_win_probability: float,
    model_name: str,
    model_artifact: str,
    league: str = "",
    tournament: str = "",
) -> dict:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ensure_prediction_log_schema()

    row = {
        "prediction_id": _next_prediction_id(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "blue_team": blue_team,
        "red_team": red_team,
        "predicted_winner": predicted_winner,
        "blue_win_probability": round(float(blue_win_probability), 4),
        "red_win_probability": round(float(red_win_probability), 4),
        "model_name": model_name,
        "model_artifact": model_artifact,
        "league": str(league).strip(),
        "tournament": str(tournament).strip(),
        "actual_winner": "",
        "correct": "",
    }

    write_header = not PREDICTION_LOG_PATH.exists()

    with PREDICTION_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    return row