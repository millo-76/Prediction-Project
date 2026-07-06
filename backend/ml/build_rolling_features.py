from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/combined_feature_matches.csv")
OUTPUT_PATH = Path("data/processed/phase3_rolling_features.csv")

REQUIRED_COLUMNS = [
    "gameid",
    "date",
    "patch",
    "league",
    "year",
    "blue_team",
    "red_team",
    "blue_side_win",
    "date_year",
    "blue_team_wr",
    "red_team_wr",
    "blue_team_games",
    "red_team_games",
    "wr_diff",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build rolling-form features from combined_feature_matches.csv "
            "and write phase3_rolling_features.csv"
        )
    )
    parser.add_argument(
        "--input",
        default=str(INPUT_PATH),
        help=f"Input CSV path (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help=f"Output CSV path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling window size in prior matches (default: 5)",
    )
    return parser.parse_args()


def _load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "blue_team", "red_team", "blue_side_win"]).copy()
    df["blue_side_win"] = pd.to_numeric(df["blue_side_win"], errors="coerce")
    df = df.dropna(subset=["blue_side_win"]).copy()
    df["blue_side_win"] = df["blue_side_win"].astype(int)
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    return df


def _rolling_wr(results: deque[int]) -> float:
    if not results:
        return 0.5
    return float(sum(results) / len(results))


def build_rolling_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("--window must be a positive integer")

    history_by_team: dict[str, deque[int]] = {}
    wr_diff_5_values: list[float] = []

    for _, row in df.iterrows():
        blue_team = str(row["blue_team"]).strip()
        red_team = str(row["red_team"]).strip()
        blue_win = int(row["blue_side_win"])

        blue_history = history_by_team.setdefault(blue_team, deque(maxlen=window))
        red_history = history_by_team.setdefault(red_team, deque(maxlen=window))

        blue_wr_recent = _rolling_wr(blue_history)
        red_wr_recent = _rolling_wr(red_history)
        wr_diff_5_values.append(blue_wr_recent - red_wr_recent)

        blue_history.append(blue_win)
        red_history.append(1 - blue_win)

    output_df = df.copy()
    output_df["wr_diff_5"] = wr_diff_5_values

    ordered_columns = REQUIRED_COLUMNS + ["wr_diff_5"]
    return output_df[ordered_columns]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = _load_input(input_path)
    output_df = build_rolling_features(df, window=args.window)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Input rows: {len(df)} from {input_path}")
    print(f"Output rows: {len(output_df)} written to {output_path}")
    print(f"Rolling window: {args.window}")


if __name__ == "__main__":
    main()