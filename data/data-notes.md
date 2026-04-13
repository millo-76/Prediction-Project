# Data Notes

## Data Source
Oracle’s Elixir – Professional League of Legends match dataset (2026 season)

## Overview
The dataset contains detailed match data at the player and team level. Each match is originally represented by multiple rows, requiring transformation into a match-level dataset for modeling.

---

## Raw Data Structure

- Each match contains **12 rows**:
  - 10 player-level rows
  - 2 team-level rows

- Key columns available:
  - `gameid`
  - `date`
  - `patch`
  - `league`
  - `side`
  - `teamname`
  - `result`
  - `position`
  - `datacompleteness`

---

## Data Filtering (Phase 1)

To ensure data quality and simplify processing:

- Only **complete matches** are used:
  ```python
  df = df[df["datacompleteness"] == "complete"]