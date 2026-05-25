from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backend.ml.predict import predict


DATA_PATH = Path("data/processed/phase4_elo_features.csv")

_df_cache = None


def load_feature_data() -> pd.DataFrame:
    global _df_cache

    if _df_cache is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Feature dataset not found: {DATA_PATH}")

        _df_cache = pd.read_csv(DATA_PATH)
        _df_cache["date"] = pd.to_datetime(_df_cache["date"])

    return _df_cache


def get_team_stats(df: pd.DataFrame, team_name: str) -> dict:
    blue_df = df[df["blue_team"] == team_name].copy()
    red_df = df[df["red_team"] == team_name].copy()

    total_games = len(blue_df) + len(red_df)

    if total_games == 0:
        raise ValueError(f"No data found for team: {team_name}")

    # All-time wins
    blue_wins = blue_df["blue_side_win"].sum()
    red_wins = (1 - red_df["blue_side_win"]).sum()
    total_wins = blue_wins + red_wins

    all_time_wr = total_wins / total_games if total_games > 0 else 0.5

    # Rolling 5 results and latest team ELO in chronological order
    team_matches = []

    for _, row in blue_df.iterrows():
        team_matches.append((row["date"], int(row["blue_side_win"]), float(row["blue_elo"])))

    for _, row in red_df.iterrows():
        team_matches.append((row["date"], int(1 - row["blue_side_win"]), float(row["red_elo"])))

    team_matches.sort(key=lambda x: x[0])

    recent_results = [result for _, result, _ in team_matches[-5:]]
    recent_games = len(recent_results)
    rolling_wr_5 = sum(recent_results) / recent_games if recent_games > 0 else 0.5
    latest_elo = team_matches[-1][2] if team_matches else 1500.0

    return {
        "games": int(total_games),
        "win_rate": float(all_time_wr),
        "rolling_win_rate_5": float(rolling_wr_5),
        "elo": float(latest_elo),
    }


def build_features_for_matchup(blue_team: str, red_team: str) -> dict:
    df = load_feature_data()

    blue_stats = get_team_stats(df, blue_team)
    red_stats = get_team_stats(df, red_team)

    return {
        "wr_diff": blue_stats["win_rate"] - red_stats["win_rate"],
        "blue_team_games": blue_stats["games"],
        "red_team_games": red_stats["games"],
        "wr_diff_5": blue_stats["rolling_win_rate_5"] - red_stats["rolling_win_rate_5"],
        "elo_diff": blue_stats["elo"] - red_stats["elo"],
    }


def predict_from_teams(blue_team: str, red_team: str) -> dict:
    features = build_features_for_matchup(blue_team, red_team)
    prediction = predict(features)

    return {
        "blue_team": blue_team,
        "red_team": red_team,
        **prediction,
        "features_used": features,
    }

def get_available_teams() -> list[str]:
    df = load_feature_data()

    teams = sorted(
        set(df["blue_team"].dropna().unique())
        | set(df["red_team"].dropna().unique())
    )

    return teams