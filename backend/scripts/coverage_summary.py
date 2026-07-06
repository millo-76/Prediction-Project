from pathlib import Path

import pandas as pd

SCHEDULE_PATH = Path("data/processed/schedule.csv")
PREDICTIONS_PATH = Path("data/predictions/schedule_predictions.csv")
FILTERED_PATH = Path("data/predictions/schedule_predictions_filtered.csv")
SKIPPED_PATH = Path("data/predictions/schedule_predictions_skipped.csv")
MISSING_TEAMS_PATH = Path("data/predictions/missing_teams_not_in_raw.csv")

REPORT_DIR = Path("reports/coverage")

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

def print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")

def save_report(df: pd.DataFrame, filename: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def summarize_counts(
        schedule_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
) -> pd.DataFrame:
    total = len(schedule_df)
    predicted = len(predictions_df)
    filtered = len(filtered_df)
    skipped = len(skipped_df)

    coverage_rate = predicted / total if total else 0
    filtered_rate = filtered / total if total else 0
    skipped_rate = skipped / total if total else 0
    usable_rate = (predicted + skipped) / total if total else 0

    summary_df = pd.DataFrame(
        [
            {
                "schedule_rows": total,
                "predictions_generated": predicted,
                "filtered_rows": filtered,
                "skipped_missing_history": skipped,
                "coverage_rate": round(coverage_rate, 4),
                "usable_rate": round(usable_rate, 4),
                "filtered_rate": round(filtered_rate, 4),
                "missing_history_rate": round(skipped_rate, 4),
            }
        ]
    )

    print_section("Coverage Summary")
    print(f"Schedule rows analyzed:   {total}")
    print(f"Predictions generated:    {predicted}")
    print(f"Filtered rows:            {filtered}")
    print(f"Skipped missing history:  {skipped}")
    print()
    print(f"Coverage Rate:            {coverage_rate:.2%}")
    print(f"Usable Rate:              {usable_rate:.2%}")
    print(f"Filtered Rate:            {filtered_rate:.2%}")
    print(f"Missing History Rate:     {skipped_rate:.2%}")

    return summary_df

def summarize_missing_teams(skipped_df: pd.DataFrame) -> pd.DataFrame:
    if skipped_df.empty:
        return pd.DataFrame(columns=["team_name", "count"])

    teams = []

    if "blue_team" in skipped_df.columns:
        teams.extend(skipped_df["blue_team"].dropna().astype(str).tolist())

    if "red_team" in skipped_df.columns:
        teams.extend(skipped_df["red_team"].dropna().astype(str).tolist())

    missing_df = (
        pd.Series(teams, name="team_name")
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .reset_index()
    )

    missing_df.columns = ["team_name", "count"]

    print_section("Top Missing Teams")
    print(missing_df.head(20).to_string(index=False))

    return missing_df


def summarize_filtered_reasons(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if filtered_df.empty or "reason" not in filtered_df.columns:
        return pd.DataFrame(columns=["reason", "count"])

    reasons_df = (
        filtered_df["reason"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    reasons_df.columns = ["reason", "count"]

    print_section("Filter Reasons")
    print(reasons_df.to_string(index=False))

    return reasons_df


def summarize_by_group(
    schedule_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    if group_column not in schedule_df.columns:
        return pd.DataFrame()

    def prep(df: pd.DataFrame, status: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[group_column, "status"])

        temp = df.copy()

        if group_column not in temp.columns:
            temp[group_column] = "Unknown"

        temp[group_column] = (
            temp[group_column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

        temp["status"] = status
        return temp[[group_column, "status"]]

    combined = pd.concat(
        [
            prep(schedule_df, "scheduled"),
            prep(predictions_df, "predicted"),
            prep(filtered_df, "filtered"),
            prep(skipped_df, "skipped"),
        ],
        ignore_index=True,
    )

    total_df = (
        combined[combined["status"] == "scheduled"]
        .groupby(group_column)
        .size()
        .reset_index(name="total")
    )

    predicted_df = (
        combined[combined["status"] == "predicted"]
        .groupby(group_column)
        .size()
        .reset_index(name="predicted")
    )

    filtered_count_df = (
        combined[combined["status"] == "filtered"]
        .groupby(group_column)
        .size()
        .reset_index(name="filtered")
    )

    skipped_count_df = (
        combined[combined["status"] == "skipped"]
        .groupby(group_column)
        .size()
        .reset_index(name="skipped")
    )

    summary = total_df.merge(predicted_df, on=group_column, how="left")
    summary = summary.merge(filtered_count_df, on=group_column, how="left")
    summary = summary.merge(skipped_count_df, on=group_column, how="left")

    for col in ["predicted", "filtered", "skipped"]:
        summary[col] = summary[col].fillna(0).astype(int)

    summary["coverage_rate"] = (summary["predicted"] / summary["total"]).round(4)
    summary = summary.sort_values(
        by=["coverage_rate", "total"],
        ascending=[True, False],
    )

    print_section(f"Coverage by {group_column.title()}")
    print(summary.head(25).to_string(index=False))

    return summary


def summarize_missing_team_inventory(missing_teams_df: pd.DataFrame) -> pd.DataFrame:
    if missing_teams_df.empty or "status" not in missing_teams_df.columns:
        return pd.DataFrame(columns=["status", "count"])

    status_df = (
        missing_teams_df["status"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    status_df.columns = ["status", "count"]

    print_section("Missing Team Inventory Status")
    print(status_df.to_string(index=False))

    return status_df


def main() -> None:
    schedule_df = read_csv_if_exists(SCHEDULE_PATH)
    predictions_df = read_csv_if_exists(PREDICTIONS_PATH)
    filtered_df = read_csv_if_exists(FILTERED_PATH)
    skipped_df = read_csv_if_exists(SKIPPED_PATH)
    missing_teams_df = read_csv_if_exists(MISSING_TEAMS_PATH)

    if schedule_df.empty:
        raise SystemExit(f"No schedule data found at {SCHEDULE_PATH}")

    summary_df = summarize_counts(
        schedule_df=schedule_df,
        predictions_df=predictions_df,
        filtered_df=filtered_df,
        skipped_df=skipped_df,
    )

    missing_frequency_df = summarize_missing_teams(skipped_df)
    filter_reasons_df = summarize_filtered_reasons(filtered_df)
    by_league_df = summarize_by_group(
        schedule_df,
        predictions_df,
        filtered_df,
        skipped_df,
        "league",
    )
    by_tournament_df = summarize_by_group(
        schedule_df,
        predictions_df,
        filtered_df,
        skipped_df,
        "tournament",
    )
    missing_status_df = summarize_missing_team_inventory(missing_teams_df)

    save_report(summary_df, "coverage_summary.csv")
    save_report(missing_frequency_df, "top_missing_teams.csv")
    save_report(filter_reasons_df, "filter_reasons.csv")

    if not by_league_df.empty:
        save_report(by_league_df, "coverage_by_league.csv")

    if not by_tournament_df.empty:
        save_report(by_tournament_df, "coverage_by_tournament.csv")

    if not missing_status_df.empty:
        save_report(missing_status_df, "missing_team_status.csv")


if __name__ == "__main__":
    main()