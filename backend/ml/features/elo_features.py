from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/processed/phase3_rolling_features.csv")
OUTPUT_PATH = Path("data/processed/phase4_elo_features.csv")

DATE_COLUMN = "date"
BLUE_TEAM_COLUMN = "blue_team"
RED_TEAM_COLUMN = "red_team"
TARGET_COLUMN = "blue_side_win"

STARTING_ELO = 1500
K_FACTOR = 32


def expected_score(team_elo: float, opponent_elo: float) -> float:
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def update_elo(current_elo: float, expected: float, actual: int) -> float:
    return current_elo + K_FACTOR * (actual - expected)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        DATE_COLUMN,
        BLUE_TEAM_COLUMN,
        RED_TEAM_COLUMN,
        TARGET_COLUMN,
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    ratings = {}

    blue_elos = []
    red_elos = []
    elo_diffs = []

    for _, row in df.iterrows():
        blue_team = row[BLUE_TEAM_COLUMN]
        red_team = row[RED_TEAM_COLUMN]
        blue_win = int(row[TARGET_COLUMN])

        blue_elo = ratings.get(blue_team, STARTING_ELO)
        red_elo = ratings.get(red_team, STARTING_ELO)

        blue_elos.append(blue_elo)
        red_elos.append(red_elo)
        elo_diffs.append(blue_elo - red_elo)

        blue_expected = expected_score(blue_elo, red_elo)
        red_expected = expected_score(red_elo, blue_elo)

        red_win = 1 - blue_win

        ratings[blue_team] = update_elo(
            current_elo=blue_elo,
            expected=blue_expected,
            actual=blue_win,
        )

        ratings[red_team] = update_elo(
            current_elo=red_elo,
            expected=red_expected,
            actual=red_win,
        )

    df["blue_elo"] = blue_elos
    df["red_elo"] = red_elos
    df["elo_diff"] = elo_diffs

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved Elo feature dataset to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Unique rated teams: {len(ratings)}")


if __name__ == "__main__":
    main()