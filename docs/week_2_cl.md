# Week 2 Checklist – Phase 1 Continuation

## Goal

Expand the training dataset by integrating 2025 season data, rebuild the feature pipeline on the combined timeline, and retrain the baseline model to evaluate the impact of additional data.

---

## Day 8 – 2025 Data Integration Plan

- [X] Download 2025 Oracle's Elixir CSV
- [X] Download latest 2026 Oracle's Elixer CSV
- [X] Save to `data/raw/`
- [X] Compare latest 2026 file to current 2026 file
- [X] Inspect column compatibility with 2026 file
- [X] Filter both datasets to:
  - [X] `datacompleteness == "complete"`
  - [X] `position == "team"`
- [X] Select common core columns
- [X] Add or confirm `year` column on each dataset
- [X] Concatenate 2025 + 2026 data
- [X] Convert `date` to datetime
- [X] Sort full dataset by date
- [X] Rebuild match-level dataset
- [X] Recompute features on combined timeline
- [X] Retrain baseline Logistic Regression
- [X] Compare against 2026-only results
