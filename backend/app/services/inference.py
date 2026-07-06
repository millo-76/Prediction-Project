from pathlib import Path
import difflib
import re
import sys
import unicodedata
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backend.ml.predict import predict


DATA_PATH = Path("data/processed/phase4_elo_features.csv")
TEAM_ALIASES_PATH = Path("data/processed/team_name_aliases.csv")

_df_cache = None
_known_teams_cache: set[str] | None = None
_team_index_cache: dict[str, list[str]] | None = None
_team_aliases_cache: dict[str, str] | None = None

_PLACEHOLDER_TEAM_NAMES = {
    "",
    "tbd",
    "bye",
    "n/a",
    "na",
    "unknown",
    "to be decided",
}

_GENERIC_TEAM_TOKENS = {
    "team",
    "esports",
    "esport",
    "gaming",
    "club",
    "academy",
    "challengers",
    "challenger",
    "organization",
}


def load_feature_data() -> pd.DataFrame:
    global _df_cache

    if _df_cache is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Feature dataset not found: {DATA_PATH}")

        _df_cache = pd.read_csv(DATA_PATH)
        _df_cache["date"] = pd.to_datetime(_df_cache["date"])

    return _df_cache


def _canonical_team_name(team_name: str) -> str:
    raw = str(team_name).strip().lower()
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    raw = re.sub(r"\(.*?\)", "", raw)
    raw = raw.replace("&", " and ")
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    raw = " ".join(raw.split())

    tokens = [token for token in raw.split() if token not in _GENERIC_TEAM_TOKENS]
    return " ".join(tokens) if tokens else raw


def _build_team_index() -> tuple[set[str], dict[str, list[str]]]:
    global _known_teams_cache, _team_index_cache

    if _known_teams_cache is None or _team_index_cache is None:
        df = load_feature_data()
        teams = sorted(set(df["blue_team"].dropna().astype(str)) | set(df["red_team"].dropna().astype(str)))
        canonical_index: dict[str, list[str]] = {}

        for team in teams:
            key = _canonical_team_name(team)
            canonical_index.setdefault(key, []).append(team)

        _known_teams_cache = set(teams)
        _team_index_cache = canonical_index

    return _known_teams_cache, _team_index_cache


def _load_team_aliases() -> dict[str, str]:
    global _team_aliases_cache

    if _team_aliases_cache is not None:
        return _team_aliases_cache

    aliases: dict[str, str] = {}
    if TEAM_ALIASES_PATH.exists():
        try:
            aliases_df = pd.read_csv(TEAM_ALIASES_PATH)
            cols = {str(col).strip().lower(): col for col in aliases_df.columns}
            alias_col = cols.get("alias")
            canonical_col = cols.get("canonical")

            if alias_col and canonical_col:
                for _, row in aliases_df.iterrows():
                    alias = str(row.get(alias_col, "")).strip()
                    canonical = str(row.get(canonical_col, "")).strip()
                    if alias and canonical:
                        aliases[alias.lower()] = canonical
        except Exception:
            aliases = {}

    _team_aliases_cache = aliases
    return _team_aliases_cache


def _is_academy_like(name: str) -> bool:
    lower = name.lower()
    return "academy" in lower or "challenger" in lower or "challengers" in lower


def _select_preferred_team(candidates: list[str], original_name: str) -> str | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    original_lower = original_name.lower()
    wants_academy = _is_academy_like(original_lower)

    filtered = [name for name in candidates if _is_academy_like(name) == wants_academy]
    if not filtered:
        filtered = candidates

    # Prefer shorter canonical labels for non-academy teams (usually main roster).
    return sorted(filtered, key=lambda name: (len(name), name))[0]


def resolve_team_name(team_name: str) -> str | None:
    cleaned = str(team_name).strip()
    if cleaned.lower() in _PLACEHOLDER_TEAM_NAMES:
        return None

    known_teams, canonical_index = _build_team_index()
    aliases = _load_team_aliases()

    alias_target = aliases.get(cleaned.lower())
    if alias_target and alias_target in known_teams:
        return alias_target

    if cleaned in known_teams:
        return cleaned

    canonical = _canonical_team_name(cleaned)
    if not canonical:
        return None

    direct_matches = canonical_index.get(canonical, [])
    if direct_matches:
        return _select_preferred_team(direct_matches, cleaned)

    close = difflib.get_close_matches(canonical, canonical_index.keys(), n=1, cutoff=0.93)
    if not close:
        return None

    candidate_matches = canonical_index.get(close[0], [])
    return _select_preferred_team(candidate_matches, cleaned)


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

    resolved_blue_team = resolve_team_name(blue_team)
    if resolved_blue_team is None:
        raise ValueError(f"No data found for team: {blue_team}")

    resolved_red_team = resolve_team_name(red_team)
    if resolved_red_team is None:
        raise ValueError(f"No data found for team: {red_team}")

    blue_stats = get_team_stats(df, resolved_blue_team)
    red_stats = get_team_stats(df, resolved_red_team)

    return {
        "wr_diff": blue_stats["win_rate"] - red_stats["win_rate"],
        "blue_team_games": blue_stats["games"],
        "red_team_games": red_stats["games"],
        "wr_diff_5": blue_stats["rolling_win_rate_5"] - red_stats["rolling_win_rate_5"],
        "elo_diff": blue_stats["elo"] - red_stats["elo"],
        "blue_team_resolved": resolved_blue_team,
        "red_team_resolved": resolved_red_team,
    }


def predict_from_teams(blue_team: str, red_team: str) -> dict:
    features = build_features_for_matchup(blue_team, red_team)
    blue_team_resolved = features.pop("blue_team_resolved")
    red_team_resolved = features.pop("red_team_resolved")
    prediction = predict(features)

    return {
        "blue_team": blue_team,
        "red_team": red_team,
        "blue_team_resolved": blue_team_resolved,
        "red_team_resolved": red_team_resolved,
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