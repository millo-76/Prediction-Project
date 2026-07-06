from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.inference import predict_from_teams
from backend.ml.prediction_logger import log_prediction
from backend.ml.update_prediction_results import print_summary, update_results_from_csv, update_results_from_oracles_elixir


DEFAULT_INPUT = Path("data/processed/schedule.csv")
DEFAULT_OUTPUT = Path("data/predictions/schedule_predictions.csv")
PLACEHOLDER_TEAM_NAMES = {
    "",
    "tbd",
    "bye",
    "n/a",
    "na",
    "unknown",
    "to be decided",
}

# Major pro-tier league/tournament identifiers.
DEFAULT_PRO_KEYWORDS = [
    "lck",
    "lpl",
    "lec",
    "lta",
    "lcs",
    "lcp",
    "lfl",
    "superliga",
    "msi",
    "world championship",
    "worlds",
]

# Terms that usually indicate lower-tier/non-pro pipelines.
NON_PRO_HINTS = [
    "academy",
    "challenger",
    "challengers",
    "2nd division",
    "3rd division",
    "open qualifier",
    "promotion",
    "rising",
    "amateur",
]


def _derive_league_tournament(row: pd.Series) -> tuple[str, str]:
    league = str(row.get("league", "") or "").strip()
    tournament = str(row.get("tournament", "") or "").strip()

    if league and tournament:
        return league, tournament

    source_row = str(row.get("source_row", "") or "").strip()
    if not source_row:
        return league, tournament

    parts = [part.strip() for part in source_row.split("/") if part.strip()]

    if not league and parts:
        league = parts[0]

    if not tournament and len(parts) > 1:
        tournament = " / ".join(parts[1:])

    return league, tournament


def _normalize_text(value: object) -> str:
    return str(value).strip().lower() if value is not None else ""


def _is_pro_row(row: pd.Series, pro_keywords: list[str]) -> bool:
    haystack = " ".join(
        [
            _normalize_text(row.get("league", "")),
            _normalize_text(row.get("tournament", "")),
            _normalize_text(row.get("source_row", "")),
            _normalize_text(row.get("source", "")),
        ]
    )

    if not haystack:
        return False

    has_pro_hint = any(keyword in haystack for keyword in pro_keywords)
    has_non_pro_hint = any(hint in haystack for hint in NON_PRO_HINTS)
    return has_pro_hint and not has_non_pro_hint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions for a schedule CSV.")
    parser.add_argument("input_csv", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--pro-only",
        action="store_true",
        help="Only predict rows matching major pro leagues/tournaments",
    )
    parser.add_argument(
        "--pro-keywords",
        default=",".join(DEFAULT_PRO_KEYWORDS),
        help="Comma-separated keywords used to identify pro rows when --pro-only is enabled",
    )
    parser.add_argument(
        "--sync-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After batch prediction, sync actual winners from a results CSV into prediction_log.csv",
    )
    parser.add_argument(
        "--sync-results-from",
        default="",
        help=(
            "CSV path to sync winners from when --sync-source csv is selected. "
            "Defaults to input schedule CSV if not set."
        ),
    )
    parser.add_argument(
        "--sync-source",
        choices=["oracles", "csv"],
        default="oracles",
        help="Source to sync actual winners from (default: oracles)",
    )
    parser.add_argument(
        "--sync-oracles-csv",
        default="",
        help="Optional Oracle's Elixir CSV path. Defaults to latest 2026_LoL_OraclesElixir*.csv",
    )
    parser.add_argument(
        "--sync-dry-run",
        action="store_true",
        help="Preview winner sync without writing to prediction_log.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Schedule CSV not found: {input_path}")

    schedule_df = pd.read_csv(input_path)

    required_columns = {"blue_team", "red_team"}
    missing_columns = required_columns - set(schedule_df.columns)
    if missing_columns:
        raise ValueError(f"Schedule CSV is missing required columns: {', '.join(sorted(missing_columns))}")

    prediction_rows: list[dict] = []
    skipped_rows: list[dict] = []
    placeholder_rows: list[dict] = []
    non_pro_rows: list[dict] = []

    pro_keywords = [part.strip().lower() for part in args.pro_keywords.split(",") if part.strip()]

    for _, row in schedule_df.iterrows():
        blue_team = str(row["blue_team"]).strip()
        red_team = str(row["red_team"]).strip()

        if not blue_team or not red_team:
            continue

        if blue_team.lower() in PLACEHOLDER_TEAM_NAMES or red_team.lower() in PLACEHOLDER_TEAM_NAMES:
            placeholder_rows.append(
                {
                    "blue_team": blue_team,
                    "red_team": red_team,
                    "league": row.get("league", ""),
                    "tournament": row.get("tournament", ""),
                    "source_row": row.get("source_row", ""),
                    "reason": "placeholder_team_name",
                }
            )
            continue

        is_pro_row = _is_pro_row(row, pro_keywords=pro_keywords)

        if not is_pro_row:
            non_pro_rows.append(
                {
                    "blue_team": blue_team,
                    "red_team": red_team,
                    "league": row.get("league", ""),
                    "tournament": row.get("tournament", ""),
                    "source_row": row.get("source_row", ""),
                    "reason": "non_pro_row_filtered",
                }
            )
            continue

        try:
            result = predict_from_teams(blue_team, red_team)

        except ValueError as error:
            message = str(error)

            if "No data found for team" in message:
                skipped_rows.append(
                    {
                        "blue_team": blue_team,
                        "red_team": red_team,
                        "league": row.get("league", ""),
                        "tournament": row.get("tournament", ""),
                        "source_row": row.get("source_row", ""),
                        "reason": message,
                    }
                )
                continue

            raise

        league, tournament = _derive_league_tournament(row)

        log_row = log_prediction(
            blue_team=blue_team,
            red_team=red_team,
            predicted_winner=result["predicted_winner"],
            blue_win_probability=result["blue_win_probability"],
            red_win_probability=result["red_win_probability"],
            model_name="Logistic Regression",
            model_artifact="logreg_phase4_elo.joblib",
            league=league,
            tournament=tournament,
        )

        prediction_rows.append(
            {
                **row.to_dict(),
                "blue_team_resolved": result.get("blue_team_resolved", blue_team),
                "red_team_resolved": result.get("red_team_resolved", red_team),
                "prediction_id": log_row["prediction_id"],
                **result,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(output_path, index=False)

    print(f"Wrote {len(prediction_rows)} predictions to {output_path}")

    if placeholder_rows:
        placeholder_path = output_path.with_name(f"{output_path.stem}_placeholders.csv")
        pd.DataFrame(placeholder_rows).to_csv(placeholder_path, index=False)
        print(f"Filtered {len(placeholder_rows)} placeholder matches. Details: {placeholder_path}")

    non_pro_path = output_path.with_name(f"{output_path.stem}_filtered.csv")
    pd.DataFrame(non_pro_rows).to_csv(non_pro_path, index=False)
    if args.pro_only:
        print(f"Filtered {len(non_pro_rows)} non-pro matches. Details: {non_pro_path}")
    else:
        print(f"Identified {len(non_pro_rows)} non-pro matches. Details: {non_pro_path}")

    if skipped_rows:
        skipped_path = output_path.with_name(f"{output_path.stem}_skipped.csv")
        pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)
        print(f"Skipped {len(skipped_rows)} matches due to missing team history. Details: {skipped_path}")

    if args.sync_results:
        try:
            if args.sync_source == "oracles":
                oracles_path = Path(args.sync_oracles_csv) if args.sync_oracles_csv else None
                print("Attempting prediction result sync from latest Oracle's Elixir outcomes")
                update_results_from_oracles_elixir(
                    oracles_csv_path=oracles_path,
                    dry_run=args.sync_dry_run,
                )
            else:
                sync_path = Path(args.sync_results_from) if args.sync_results_from else input_path
                print(f"Attempting prediction result sync from CSV: {sync_path}")
                update_results_from_csv(
                    results_path=sync_path,
                    dry_run=args.sync_dry_run,
                )
        except FileNotFoundError as error:
            print(f"Skipped result sync: {error}")
        except KeyError as error:
            print(f"Skipped result sync: {error}")

        if not args.sync_dry_run:
            print_summary()


if __name__ == "__main__":
    main()