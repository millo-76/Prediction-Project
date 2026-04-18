# Week 2 Checklist – Phase 1 Continuation

## Goal

Expand the training dataset by integrating 2025 season data, rebuild the feature pipeline on the combined timeline, and retrain the baseline model to evaluate the impact of additional data.

---

## Day 8 – 2025 Data Integration Plan

- [ ] Download 2025 Oracle's Elixir CSV
- [ ] Download latest 2026 Oracle's Elixer CSV
- [ ] Save to `data/raw/`
- [ ] Compare latest 2026 file to current 2026 file
- [ ] Inspect column compatibility with 2026 file
- [ ] Filter both datasets to:
  - [ ] `datacompleteness == "complete"`
  - [ ] `position == "team"`
- [ ] Select common core columns
- [ ] Add or confirm `year` column on each dataset
- [ ] Concatenate 2025 + 2026 data
- [ ] Convert `date` to datetime
- [ ] Sort full dataset by date
- [ ] Rebuild match-level dataset
- [ ] Recompute features on combined timeline
- [ ] Retrain baseline Logistic Regression
- [ ] Compare against 2026-only results
