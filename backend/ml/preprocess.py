from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
DEFAULT_PATTERN = "*LoL_OraclesElixir*.csv"

PROCESSED_OUTPUT_PATH = Path("data/processed/combined_processed_matches.csv")
FEATURE_OUTPUT_PATH = Path("data/processed/combined_feature_matches.csv")

REQUIRED_COLUMNS = [
	"gameid",
	"datacompleteness",
	"date",
	"patch",
	"league",
	"year",
	"side",
	"teamname",
	"result",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Build match-level datasets from raw Oracle's Elixir exports. "
			"Outputs combined_processed_matches.csv and combined_feature_matches.csv."
		)
	)
	parser.add_argument(
		"--pattern",
		default=DEFAULT_PATTERN,
		help=f"Glob pattern inside {RAW_DATA_DIR} (default: {DEFAULT_PATTERN})",
	)
	parser.add_argument(
		"--processed-output",
		default=str(PROCESSED_OUTPUT_PATH),
		help=f"Output path for processed match-level CSV (default: {PROCESSED_OUTPUT_PATH})",
	)
	parser.add_argument(
		"--feature-output",
		default=str(FEATURE_OUTPUT_PATH),
		help=f"Output path for feature CSV (default: {FEATURE_OUTPUT_PATH})",
	)
	return parser.parse_args()


def _read_oracles_files(pattern: str) -> pd.DataFrame:
	input_paths = sorted(RAW_DATA_DIR.glob(pattern))
	if not input_paths:
		raise FileNotFoundError(
			f"No Oracle files found in {RAW_DATA_DIR} matching pattern: {pattern}"
		)

	frames: list[pd.DataFrame] = []
	for path in input_paths:
		frame = pd.read_csv(path, usecols=lambda c: c in REQUIRED_COLUMNS)
		missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
		if missing:
			raise KeyError(f"Missing required columns in {path}: {missing}")
		frame["_source_file"] = path.name
		frames.append(frame)

	return pd.concat(frames, ignore_index=True)


def _build_match_level(raw_df: pd.DataFrame) -> pd.DataFrame:
	df = raw_df.copy()

	df["datacompleteness"] = df["datacompleteness"].astype(str).str.lower().str.strip()
	df = df[df["datacompleteness"] == "complete"].copy()

	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df = df.dropna(subset=["date", "gameid", "side", "teamname", "result"]).copy()

	df["side"] = df["side"].astype(str).str.lower().str.strip()
	df = df[df["side"].isin(["blue", "red"])].copy()

	# Keep one representative row per game-side across all player/team rows.
	df = (
		df.sort_values(["gameid", "side", "date", "_source_file"])
		.drop_duplicates(subset=["gameid", "side"], keep="first")
		.reset_index(drop=True)
	)

	blue_rows = df[df["side"] == "blue"].copy()
	red_rows = df[df["side"] == "red"].copy()

	blue_rows = blue_rows.rename(
		columns={
			"teamname": "blue_team",
			"result": "blue_side_win",
			"date": "blue_date",
			"patch": "blue_patch",
			"league": "blue_league",
			"year": "blue_year",
		}
	)

	red_rows = red_rows.rename(
		columns={
			"teamname": "red_team",
			"date": "red_date",
			"patch": "red_patch",
			"league": "red_league",
			"year": "red_year",
		}
	)

	merged = blue_rows[["gameid", "blue_team", "blue_side_win", "blue_date", "blue_patch", "blue_league", "blue_year"]].merge(
		red_rows[["gameid", "red_team", "red_date", "red_patch", "red_league", "red_year"]],
		on="gameid",
		how="inner",
	)

	merged["date"] = merged["blue_date"].combine_first(merged["red_date"])
	merged["patch"] = merged["blue_patch"].combine_first(merged["red_patch"])
	merged["league"] = merged["blue_league"].combine_first(merged["red_league"])
	merged["year"] = merged["blue_year"].combine_first(merged["red_year"])
	merged["blue_side_win"] = pd.to_numeric(merged["blue_side_win"], errors="coerce")

	out_df = merged[
		[
			"gameid",
			"date",
			"patch",
			"league",
			"year",
			"blue_team",
			"red_team",
			"blue_side_win",
		]
	].copy()

	out_df = out_df.dropna(subset=["date", "blue_side_win", "blue_team", "red_team"])
	out_df["blue_side_win"] = out_df["blue_side_win"].astype(int)
	out_df["year"] = pd.to_numeric(out_df["year"], errors="coerce").fillna(out_df["date"].dt.year).astype(int)

	out_df = out_df.sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid"], keep="last")
	out_df = out_df.reset_index(drop=True)
	return out_df


def _build_feature_dataset(processed_df: pd.DataFrame) -> pd.DataFrame:
	df = processed_df.copy()
	df["date_year"] = df["date"].dt.year

	team_games: dict[str, int] = {}
	team_wins: dict[str, int] = {}

	blue_team_wr: list[float] = []
	red_team_wr: list[float] = []
	blue_team_games: list[int] = []
	red_team_games: list[int] = []

	for _, row in df.iterrows():
		blue_team = str(row["blue_team"])
		red_team = str(row["red_team"])
		blue_win = int(row["blue_side_win"])

		blue_games_before = team_games.get(blue_team, 0)
		red_games_before = team_games.get(red_team, 0)

		blue_wins_before = team_wins.get(blue_team, 0)
		red_wins_before = team_wins.get(red_team, 0)

		blue_wr_before = (
			blue_wins_before / blue_games_before if blue_games_before > 0 else 0.5
		)
		red_wr_before = red_wins_before / red_games_before if red_games_before > 0 else 0.5

		blue_team_wr.append(float(blue_wr_before))
		red_team_wr.append(float(red_wr_before))
		blue_team_games.append(int(blue_games_before))
		red_team_games.append(int(red_games_before))

		team_games[blue_team] = blue_games_before + 1
		team_games[red_team] = red_games_before + 1
		team_wins[blue_team] = blue_wins_before + blue_win
		team_wins[red_team] = red_wins_before + (1 - blue_win)

	df["blue_team_wr"] = blue_team_wr
	df["red_team_wr"] = red_team_wr
	df["blue_team_games"] = blue_team_games
	df["red_team_games"] = red_team_games
	df["wr_diff"] = df["blue_team_wr"] - df["red_team_wr"]

	ordered_columns = [
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
	return df[ordered_columns]


def main() -> None:
	args = parse_args()

	raw_df = _read_oracles_files(pattern=args.pattern)
	processed_df = _build_match_level(raw_df)
	feature_df = _build_feature_dataset(processed_df)

	processed_output = Path(args.processed_output)
	feature_output = Path(args.feature_output)

	processed_output.parent.mkdir(parents=True, exist_ok=True)
	feature_output.parent.mkdir(parents=True, exist_ok=True)

	processed_df.to_csv(processed_output, index=False)
	feature_df.to_csv(feature_output, index=False)

	print(f"Raw rows read: {len(raw_df)}")
	print(f"Match rows written: {len(processed_df)} -> {processed_output}")
	print(f"Feature rows written: {len(feature_df)} -> {feature_output}")
	print(
		"Date range: "
		f"{processed_df['date'].min().date()} -> {processed_df['date'].max().date()}"
	)


if __name__ == "__main__":
	main()
