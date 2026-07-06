from pathlib import Path

import pandas as pd

def load_schedule(source: str) -> pd.DataFrame:
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Schedule file not found: {path}")
    
    raw_df = pd.read_csv(path)

    processed = pd.DataFrame({
        "source": str(path),
        "source_row": raw_df["OverviewPage"],
        "scheduled_date": pd.to_datetime(
            raw_df["DateTime UTC"],
            utc=True,
        ).dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "league": raw_df["OverviewPage"].str.split("/").str[0],
        "tournament": raw_df["OverviewPage"],
        "patch": raw_df.get("Patch", ""),
        "blue_team": raw_df["Team1"],
        "red_team": raw_df["Team2"],
    })

    processed["scheduled_date"] = (
        processed["scheduled_date"]
        .str.replace(r"(\+0000)$", "+00:00", regex=True)
    )

    return processed

def main() -> None:
    schedule_df = load_schedule("data/raw/schedule_export.csv")
    schedule_df.to_csv("data/processed/schedule.csv", index=False)

    print(f"Wrote {len(schedule_df)} rows to data/processed/schedule.csv")

if __name__ == "__main__":
    main()