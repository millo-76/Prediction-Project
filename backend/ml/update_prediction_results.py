import argparse
import re
import unicodedata
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError


PREDICTION_LOG_PATH = Path("data/predictions/prediction_log.csv")
DEFAULT_RESULTS_PATH = Path("data/predictions/schedule_predictions.csv")
RAW_DATA_DIR = Path("data/raw")
PREDICTIONS_DIR = Path("data/predictions")
PROCESSED_SCHEDULE_PATH = Path("data/processed/schedule.csv")


def _empty_metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "prediction_id",
            "blue_team_key",
            "red_team_key",
            "scheduled_ts",
            "league",
            "tournament",
        ]
    )


def _parse_league_tournament(
    league_value: object,
    tournament_value: object,
    source_row_value: object,
) -> tuple[str, str]:
    league = str(league_value).strip() if pd.notna(league_value) else ""
    tournament = str(tournament_value).strip() if pd.notna(tournament_value) else ""
    source_row = str(source_row_value).strip() if pd.notna(source_row_value) else ""

    if source_row and (not league or not tournament):
        parts = [part.strip() for part in source_row.split("/") if part.strip()]

        if not league and parts:
            league = parts[0]

        if not tournament and len(parts) > 1:
            tournament = " / ".join(parts[1:])

    return league, tournament


def _load_prediction_metadata() -> pd.DataFrame:
    """Load metadata for league/tournament from schedule and prediction exports."""
    source_paths: list[Path] = []

    if PROCESSED_SCHEDULE_PATH.exists():
        source_paths.append(PROCESSED_SCHEDULE_PATH)

    if PREDICTIONS_DIR.exists():
        source_paths.extend(sorted(PREDICTIONS_DIR.glob("schedule_predictions*.csv")))

    if not source_paths:
        return _empty_metadata_df()

    metadata_rows: list[pd.DataFrame] = []

    for path in source_paths:
        try:
            csv_df = pd.read_csv(path)
        except Exception:
            continue

        required = {"blue_team", "red_team"}
        if not required.issubset(set(csv_df.columns)):
            continue

        working = csv_df.copy()

        if "prediction_id" not in working.columns:
            working["prediction_id"] = ""

        if "scheduled_date" not in working.columns:
            working["scheduled_date"] = pd.NA

        if "league" not in working.columns:
            working["league"] = ""

        if "tournament" not in working.columns:
            working["tournament"] = ""

        if "source_row" not in working.columns:
            working["source_row"] = ""

        subset = working[
            [
                "prediction_id",
                "blue_team",
                "red_team",
                "scheduled_date",
                "league",
                "tournament",
                "source_row",
            ]
        ].copy()

        parsed = subset.apply(
            lambda row: _parse_league_tournament(
                row["league"],
                row["tournament"],
                row["source_row"],
            ),
            axis=1,
            result_type="expand",
        )
        parsed.columns = ["league", "tournament"]

        subset["league"] = parsed["league"]
        subset["tournament"] = parsed["tournament"]
        subset["prediction_id"] = subset["prediction_id"].astype(str).str.strip()
        subset["blue_team_key"] = subset["blue_team"].map(_canonical_team_name)
        subset["red_team_key"] = subset["red_team"].map(_canonical_team_name)
        subset["scheduled_ts"] = pd.to_datetime(subset["scheduled_date"], errors="coerce", utc=True)

        subset = subset[
            (subset["league"].astype(str).str.strip() != "")
            | (subset["tournament"].astype(str).str.strip() != "")
        ]

        if subset.empty:
            continue

        metadata_rows.append(
            subset[
                [
                    "prediction_id",
                    "blue_team_key",
                    "red_team_key",
                    "scheduled_ts",
                    "league",
                    "tournament",
                ]
            ]
        )

    if not metadata_rows:
        return _empty_metadata_df()

    metadata = pd.concat(metadata_rows, ignore_index=True)
    metadata = metadata.drop_duplicates(
        subset=["prediction_id", "blue_team_key", "red_team_key", "scheduled_ts", "league", "tournament"],
        keep="last",
    )
    return metadata


def _fill_missing_metadata_from_team_matches(scored_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    missing_mask = (
        scored_df["league"].fillna("").astype(str).str.strip().eq("")
        | scored_df["tournament"].fillna("").astype(str).str.strip().eq("")
    )

    if not missing_mask.any() or metadata_df.empty:
        return

    candidates_by_pair: dict[tuple[str, str], list[dict]] = {}

    for _, row in metadata_df.iterrows():
        blue_key = str(row["blue_team_key"]).strip()
        red_key = str(row["red_team_key"]).strip()
        if not blue_key or not red_key:
            continue

        league = str(row["league"]).strip()
        tournament = str(row["tournament"]).strip()
        if not league and not tournament:
            continue

        item = {
            "scheduled_ts": row["scheduled_ts"],
            "league": league,
            "tournament": tournament,
        }

        candidates_by_pair.setdefault((blue_key, red_key), []).append(item)

    if not candidates_by_pair:
        return

    for idx in scored_df[missing_mask].index:
        blue_key = scored_df.at[idx, "blue_team_key"]
        red_key = scored_df.at[idx, "red_team_key"]
        prediction_ts = scored_df.at[idx, "timestamp_utc"]

        pair_candidates = candidates_by_pair.get((blue_key, red_key), [])
        if not pair_candidates:
            pair_candidates = candidates_by_pair.get((red_key, blue_key), [])

        if not pair_candidates:
            continue

        best_candidate = pair_candidates[0]

        if pd.notna(prediction_ts):
            dated = [c for c in pair_candidates if pd.notna(c["scheduled_ts"])]
            if dated:
                best_candidate = min(
                    dated,
                    key=lambda c: abs((c["scheduled_ts"] - prediction_ts).total_seconds()),
                )

        if not str(scored_df.at[idx, "league"]).strip() and best_candidate["league"]:
            scored_df.at[idx, "league"] = best_candidate["league"]

        if not str(scored_df.at[idx, "tournament"]).strip() and best_candidate["tournament"]:
            scored_df.at[idx, "tournament"] = best_candidate["tournament"]


def _print_group_accuracy(scored_df: pd.DataFrame, group_column: str, title: str) -> None:
    print()
    print(title)

    group_values = scored_df[group_column].fillna("").astype(str).str.strip()
    group_values = group_values.where(group_values != "", "Unknown")

    grouped = (
        scored_df.assign(_group=group_values)
        .groupby("_group")["correct"]
        .agg(["count", "sum", "mean"])
        .sort_values(["count", "mean"], ascending=[False, False])
    )

    for group_name, row in grouped.iterrows():
        total = int(row["count"])
        correct = int(row["sum"])
        accuracy = float(row["mean"])
        print(f"{group_name}: {correct}/{total} ({accuracy:.2%})")


def _print_confidence_bucket_accuracy(scored_df: pd.DataFrame) -> None:
    print()
    print("By Confidence Bucket")

    bucket_order = [
        "Probability >= 80%",
        "Probability 70-80%",
        "Probability 60-70%",
        "Probability 50-60%",
        "Probability <= 50%",
    ]

    blue_prob = pd.to_numeric(scored_df.get("blue_win_probability"), errors="coerce")
    red_prob = pd.to_numeric(scored_df.get("red_win_probability"), errors="coerce")
    confidence = pd.concat([blue_prob, red_prob], axis=1).max(axis=1)

    bucket_series = pd.Series("Probability <= 50%", index=scored_df.index, dtype="string")
    bucket_series = bucket_series.mask(confidence > 0.5, "Probability 50-60%")
    bucket_series = bucket_series.mask(confidence >= 0.6, "Probability 60-70%")
    bucket_series = bucket_series.mask(confidence >= 0.7, "Probability 70-80%")
    bucket_series = bucket_series.mask(confidence >= 0.8, "Probability >= 80%")

    bucket_grouped = (
        scored_df.assign(_bucket=bucket_series)
        .groupby("_bucket")["correct"]
        .agg(["count", "sum", "mean"])
    )

    for bucket in bucket_order:
        if bucket not in bucket_grouped.index:
            print(f"{bucket}: 0/0 (n/a)")
            continue

        row = bucket_grouped.loc[bucket]
        total = int(row["count"])
        correct = int(row["sum"])
        accuracy = float(row["mean"])
        print(f"{bucket}: {correct}/{total} ({accuracy:.2%})")


def normalize_winner(value: str) -> str:
    value = value.strip().lower()

    if value in {"blue", "b"}:
        return "blue"

    if value in {"red", "r"}:
        return "red"

    raise ValueError("Winner must be 'blue' or 'red'.")


def _pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]
    return None


def _canonical_team_name(team_name: str) -> str:
    raw = str(team_name).strip().lower()
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    raw = re.sub(r"\(.*?\)", "", raw)
    raw = raw.replace("&", " and ")
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return " ".join(raw.split())


def _latest_oracles_elixir_2026_csv() -> Path:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    candidates = list(RAW_DATA_DIR.glob("2026_LoL_OraclesElixir*.csv"))
    if not candidates:
        raise FileNotFoundError("No 2026_LoL_OraclesElixir*.csv files found in data/raw")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_prediction_log() -> pd.DataFrame:
    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(f"Prediction log not found: {PREDICTION_LOG_PATH}")

    return pd.read_csv(
        PREDICTION_LOG_PATH,
        dtype={
            "actual_winner": "string",
            "correct": "string",
        },
    )


def update_result(prediction_id: str, actual_winner: str) -> None:
    df = load_prediction_log()

    if prediction_id not in df["prediction_id"].values:
        raise ValueError(f"No prediction found with ID: {prediction_id}")

    actual_winner = normalize_winner(actual_winner)

    mask = df["prediction_id"] == prediction_id

    predicted_winner = df.loc[mask, "predicted_winner"].iloc[0]
    predicted_winner = normalize_winner(predicted_winner)

    correct = predicted_winner == actual_winner

    df.loc[mask, "actual_winner"] = actual_winner
    df.loc[mask, "correct"] = str(correct)

    df.to_csv(PREDICTION_LOG_PATH, index=False)

    print("Updated prediction result")
    print(f"Prediction ID: {prediction_id}")
    print(f"Predicted winner: {predicted_winner}")
    print(f"Actual winner: {actual_winner}")
    print(f"Correct: {correct}")


def update_results_from_csv(
    results_path: Path,
    id_column: str = "prediction_id",
    winner_column: str = "actual_winner",
    dry_run: bool = False,
) -> None:
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    try:
        updates_df = pd.read_csv(results_path)
    except EmptyDataError:
        print(f"Results CSV is empty: {results_path}")
        print("Add header/rows (prediction_id, actual_winner) and rerun.")
        return

    id_col = id_column if id_column in updates_df.columns else _pick_existing_column(
        updates_df,
        ["prediction_id", "id"],
    )

    winner_col = winner_column if winner_column in updates_df.columns else _pick_existing_column(
        updates_df,
        ["actual_winner", "winner", "match_winner", "result"],
    )

    if id_col is None:
        raise KeyError(
            "Results CSV is missing a prediction ID column. "
            "Expected one of: prediction_id, id"
        )

    if winner_col is None:
        raise KeyError(
            "Results CSV is missing a winner column. "
            "Expected one of: actual_winner, winner, match_winner, result"
        )

    # Keep the latest row for duplicate IDs in the update file.
    total_rows = len(updates_df)
    updates_df = updates_df.dropna(subset=[id_col, winner_col]).copy()
    dropped_na_rows = total_rows - len(updates_df)

    updates_df[id_col] = updates_df[id_col].astype(str).str.strip()
    updates_df[winner_col] = updates_df[winner_col].astype(str).str.strip()

    empty_id_rows = int((updates_df[id_col] == "").sum())
    empty_winner_rows = int((updates_df[winner_col] == "").sum())

    updates_df = updates_df[updates_df[id_col] != ""]
    updates_df = updates_df[updates_df[winner_col] != ""]
    updates_df = updates_df.drop_duplicates(subset=[id_col], keep="last")

    df = load_prediction_log()

    if "prediction_id" not in df.columns:
        raise KeyError("Prediction log is missing 'prediction_id' column.")

    valid_updates: list[tuple[str, str]] = []
    invalid_winner_count = 0

    for _, row in updates_df.iterrows():
        prediction_id = row[id_col]
        winner_raw = row[winner_col]

        try:
            winner = normalize_winner(winner_raw)
        except ValueError:
            invalid_winner_count += 1
            continue

        valid_updates.append((prediction_id, winner))

    if not valid_updates:
        print("No valid updates found in CSV.")
        print(f"Rows in source CSV: {total_rows}")
        if dropped_na_rows:
            print(f"Rows dropped due to missing ID/winner: {dropped_na_rows}")
        if empty_id_rows:
            print(f"Rows with empty prediction ID: {empty_id_rows}")
        if empty_winner_rows:
            print(f"Rows with empty winner: {empty_winner_rows}")
            print("Fill winner values with 'blue' or 'red' and rerun.")
        if invalid_winner_count:
            print(f"Skipped rows with invalid winner values: {invalid_winner_count}")
        return

    updated = 0
    not_found = 0

    for prediction_id, actual_winner in valid_updates:
        mask = df["prediction_id"] == prediction_id

        if not mask.any():
            not_found += 1
            continue

        predicted_winner = normalize_winner(df.loc[mask, "predicted_winner"].iloc[0])
        correct = predicted_winner == actual_winner

        df.loc[mask, "actual_winner"] = actual_winner
        df.loc[mask, "correct"] = str(correct)
        updated += 1

    if dry_run:
        print("Dry run complete. No file changes were written.")
    else:
        df.to_csv(PREDICTION_LOG_PATH, index=False)
        print(f"Updated prediction log: {PREDICTION_LOG_PATH}")

    print(f"Requested updates: {len(valid_updates)}")
    print(f"Applied updates: {updated}")
    print(f"Prediction IDs not found: {not_found}")
    if invalid_winner_count:
        print(f"Skipped rows with invalid winner values: {invalid_winner_count}")


def update_results_from_oracles_elixir(
    oracles_csv_path: Path | None = None,
    dry_run: bool = False,
) -> None:
    source_path = oracles_csv_path or _latest_oracles_elixir_2026_csv()

    if not source_path.exists():
        raise FileNotFoundError(f"Oracle's Elixir CSV not found: {source_path}")

    raw_df = pd.read_csv(source_path, low_memory=False)

    required_columns = {
        "gameid",
        "date",
        "side",
        "teamname",
        "result",
        "position",
        "datacompleteness",
    }
    missing = required_columns - set(raw_df.columns)
    if missing:
        raise KeyError(
            "Oracle's Elixir CSV missing required columns: "
            + ", ".join(sorted(missing))
        )

    raw_df = raw_df[
        (raw_df["datacompleteness"] == "complete")
        & (raw_df["position"] == "team")
    ].copy()

    raw_df["side"] = raw_df["side"].astype(str).str.lower().str.strip()
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce", utc=True)
    raw_df = raw_df.dropna(subset=["date"])

    match_rows: list[dict] = []

    for gameid, group in raw_df.groupby("gameid"):
        if len(group) != 2:
            continue

        blue = group[group["side"] == "blue"]
        red = group[group["side"] == "red"]
        if blue.empty or red.empty:
            continue

        blue_row = blue.iloc[0]
        red_row = red.iloc[0]

        try:
            blue_win = int(blue_row["result"])
        except (TypeError, ValueError):
            continue

        actual_winner = "blue" if blue_win == 1 else "red"

        match_rows.append(
            {
                "gameid": str(gameid),
                "date": blue_row["date"],
                "blue_team": str(blue_row["teamname"]).strip(),
                "red_team": str(red_row["teamname"]).strip(),
                "actual_winner": actual_winner,
                "blue_team_key": _canonical_team_name(blue_row["teamname"]),
                "red_team_key": _canonical_team_name(red_row["teamname"]),
                "used": False,
            }
        )

    if not match_rows:
        print(f"No usable team-level matches found in {source_path}")
        return

    match_df = pd.DataFrame(match_rows).sort_values("date").reset_index(drop=True)
    match_lookup: dict[tuple[str, str], list[int]] = {}

    for idx, row in match_df.iterrows():
        key = (row["blue_team_key"], row["red_team_key"])
        match_lookup.setdefault(key, []).append(idx)

    log_df = load_prediction_log()
    log_df["timestamp_utc"] = pd.to_datetime(log_df["timestamp_utc"], errors="coerce", utc=True)

    unresolved_mask = log_df["actual_winner"].isna() | (log_df["actual_winner"].astype(str).str.strip() == "")
    unresolved_df = log_df[unresolved_mask].copy()
    unresolved_df = unresolved_df.sort_values("timestamp_utc").reset_index()

    if unresolved_df.empty:
        print("No unresolved predictions found in prediction_log.csv")
        return

    matched = 0

    for _, pred in unresolved_df.iterrows():
        pred_idx = pred["index"]
        pred_time = pred["timestamp_utc"]
        blue_key = _canonical_team_name(pred["blue_team"])
        red_key = _canonical_team_name(pred["red_team"])

        candidate_ids = match_lookup.get((blue_key, red_key), [])
        if not candidate_ids:
            continue

        selected_idx = None

        for candidate_idx in candidate_ids:
            if bool(match_df.at[candidate_idx, "used"]):
                continue

            match_time = match_df.at[candidate_idx, "date"]
            if pd.isna(pred_time) or match_time >= pred_time:
                selected_idx = candidate_idx
                break

        if selected_idx is None:
            for candidate_idx in candidate_ids:
                if not bool(match_df.at[candidate_idx, "used"]):
                    selected_idx = candidate_idx
                    break

        if selected_idx is None:
            continue

        actual_winner = str(match_df.at[selected_idx, "actual_winner"])
        predicted_winner = normalize_winner(str(log_df.at[pred_idx, "predicted_winner"]))
        correct = predicted_winner == actual_winner

        log_df.at[pred_idx, "actual_winner"] = actual_winner
        log_df.at[pred_idx, "correct"] = str(correct)
        match_df.at[selected_idx, "used"] = True
        matched += 1

    if dry_run:
        print("Dry run complete. No file changes were written.")
    else:
        log_df.to_csv(PREDICTION_LOG_PATH, index=False)
        print(f"Updated prediction log: {PREDICTION_LOG_PATH}")

    print(f"Oracle's Elixir source: {source_path}")
    print(f"Unresolved predictions considered: {len(unresolved_df)}")
    print(f"Predictions matched and updated: {matched}")
    print(f"Unmatched unresolved predictions: {len(unresolved_df) - matched}")


def print_summary() -> None:
    try:
        df = load_prediction_log()
    except FileNotFoundError:
        print(f"No prediction log found at {PREDICTION_LOG_PATH}; summary skipped.")
        return

    scored = df.dropna(subset=["actual_winner", "correct"])
    scored = scored[scored["actual_winner"].astype(str).str.strip() != ""]

    if scored.empty:
        print("No scored predictions yet.")
        return

    scored["correct"] = scored["correct"].astype(str).str.lower() == "true"
    scored["timestamp_utc"] = pd.to_datetime(scored.get("timestamp_utc"), errors="coerce", utc=True)
    scored["blue_team_key"] = scored["blue_team"].map(_canonical_team_name)
    scored["red_team_key"] = scored["red_team"].map(_canonical_team_name)

    if "league" not in scored.columns:
        scored["league"] = ""
    scored["league"] = scored["league"].fillna("").astype("string")

    if "tournament" not in scored.columns:
        scored["tournament"] = ""
    scored["tournament"] = scored["tournament"].fillna("").astype("string")

    metadata = _load_prediction_metadata()
    if not metadata.empty and "prediction_id" in scored.columns:
        id_metadata = metadata[metadata["prediction_id"].astype(str).str.strip() != ""].copy()
        id_metadata = id_metadata.drop_duplicates(subset=["prediction_id"], keep="last")

        id_metadata = id_metadata.rename(
            columns={
                "league": "metadata_league",
                "tournament": "metadata_tournament",
            }
        )

        scored = scored.merge(
            id_metadata[["prediction_id", "metadata_league", "metadata_tournament"]],
            on="prediction_id",
            how="left",
        )

        league_missing = scored["league"].fillna("").astype(str).str.strip().eq("")
        tournament_missing = scored["tournament"].fillna("").astype(str).str.strip().eq("")

        scored.loc[league_missing, "league"] = scored.loc[league_missing, "metadata_league"]
        scored.loc[tournament_missing, "tournament"] = scored.loc[tournament_missing, "metadata_tournament"]

        scored = scored.drop(columns=["metadata_league", "metadata_tournament"])

    _fill_missing_metadata_from_team_matches(scored, metadata)

    accuracy = scored["correct"].mean()

    print()
    print("=== Prediction Log Summary ===")
    print(f"Scored predictions: {len(scored)}")
    print(f"Correct predictions: {scored['correct'].sum()}")
    print(f"Accuracy: {accuracy:.2%}")

    _print_group_accuracy(scored, group_column="league", title="By League")
    _print_group_accuracy(scored, group_column="tournament", title="By Tournament")
    _print_confidence_bucket_accuracy(scored)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update prediction outcomes in data/predictions/prediction_log.csv. "
            "Supports single-update mode and bulk CSV mode."
        )
    )
    parser.add_argument("--prediction-id", help="Prediction ID to update")
    parser.add_argument(
        "--actual-winner",
        help="Actual winner for --prediction-id (blue/red)",
    )
    parser.add_argument(
        "--results-csv",
        default="",
        help=(
            "Path to CSV containing bulk updates "
            "(optional; when omitted the script uses single-update or interactive mode)"
        ),
    )
    parser.add_argument(
        "--id-column",
        default="prediction_id",
        help="Column name for prediction ID in --results-csv",
    )
    parser.add_argument(
        "--winner-column",
        default="actual_winner",
        help="Column name for winner value in --results-csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview bulk updates without writing to prediction_log.csv",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print scored prediction summary without applying updates",
    )
    parser.add_argument(
        "--from-oracles-elixir",
        action="store_true",
        help="Update outcomes from the latest 2026 Oracle's Elixir raw CSV",
    )
    parser.add_argument(
        "--oracles-csv",
        default="",
        help="Optional explicit Oracle's Elixir CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_csv_arg = str(args.results_csv).strip()

    if args.summary_only:
        print_summary()
        return

    if args.from_oracles_elixir:
        try:
            source_path = Path(args.oracles_csv) if args.oracles_csv else None
            update_results_from_oracles_elixir(
                oracles_csv_path=source_path,
                dry_run=args.dry_run,
            )
        except (FileNotFoundError, KeyError) as error:
            print(str(error))
            return

        if not args.dry_run:
            print_summary()
        return

    if results_csv_arg:
        try:
            update_results_from_csv(
                results_path=Path(results_csv_arg),
                id_column=args.id_column,
                winner_column=args.winner_column,
                dry_run=args.dry_run,
            )
        except (KeyError, FileNotFoundError) as error:
            print(str(error))
            print(
                "Tip: ensure the file has prediction IDs plus winners, "
                "then rerun with --id-column/--winner-column if needed."
            )
            return

        if not args.dry_run:
            print_summary()
        return

    if args.prediction_id or args.actual_winner:
        if not (args.prediction_id and args.actual_winner):
            raise ValueError("Both --prediction-id and --actual-winner are required together.")

        update_result(args.prediction_id, args.actual_winner)
        print_summary()
        return

    print("=== Update Prediction Result (Interactive Mode) ===")
    prediction_id = input("Prediction ID: ").strip()
    actual_winner = input("Actual winner blue/red: ").strip()

    update_result(prediction_id, actual_winner)
    print_summary()


if __name__ == "__main__":
    main()