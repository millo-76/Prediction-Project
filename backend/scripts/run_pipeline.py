from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ML pipeline by mode.")
    parser.add_argument(
        "--mode",
        choices=["full-retrain", "predict-only", "sync-only"],
        required=True,
        help="Pipeline mode to execute",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run child scripts (default: current interpreter)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command sequence without executing it",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later steps even if one step fails",
    )
    return parser.parse_args()


def _steps_for_mode(mode: str) -> list[tuple[str, list[str]]]:
    if mode == "full-retrain":
        return [
            ("preprocess", ["backend/ml/preprocess.py"]),
            ("build rolling features", ["backend/ml/build_rolling_features.py"]),
            ("build elo features", ["backend/ml/features/elo_features.py"]),
            ("train model", ["backend/ml/train.py"]),
            ("evaluate model", ["backend/ml/evaluate.py"]),
            ("batch predict", ["backend/scripts/batch_predict.py"]),
            (
                "sync prediction outcomes",
                ["backend/ml/update_prediction_results.py", "--from-oracles-elixir"],
            ),
            ("coverage summary", ["backend/scripts/coverage_summary.py"]),
        ]

    if mode == "predict-only":
        return [
            ("batch predict", ["backend/scripts/batch_predict.py"]),
            ("coverage summary", ["backend/scripts/coverage_summary.py"]),
        ]

    if mode == "sync-only":
        return [
            (
                "sync prediction outcomes",
                ["backend/ml/update_prediction_results.py", "--from-oracles-elixir"],
            ),
            ("coverage summary", ["backend/scripts/coverage_summary.py"]),
        ]

    raise ValueError(f"Unsupported mode: {mode}")


def _run_step(python_exec: str, step_name: str, step_args: list[str]) -> int:
    command = [python_exec, *step_args]
    print()
    print(f"=== Step: {step_name} ===")
    print("Command:", " ".join(command))

    completed = subprocess.run(command, cwd=REPO_ROOT)
    return int(completed.returncode)


def main() -> None:
    args = parse_args()
    steps = _steps_for_mode(args.mode)

    print(f"Mode: {args.mode}")
    print(f"Repository: {REPO_ROOT}")
    print(f"Python: {args.python}")

    if args.dry_run:
        print("Dry run enabled. Planned steps:")
        for index, (step_name, step_args) in enumerate(steps, start=1):
            print(f"{index}. {step_name}: {args.python} {' '.join(step_args)}")
        return

    failures: list[tuple[str, int]] = []

    for step_name, step_args in steps:
        code = _run_step(args.python, step_name, step_args)
        if code != 0:
            failures.append((step_name, code))
            print(f"Step failed: {step_name} (exit code {code})")
            if not args.continue_on_error:
                break

    print()
    if failures:
        print("Pipeline finished with failures:")
        for step_name, code in failures:
            print(f"- {step_name}: exit code {code}")
        raise SystemExit(1)

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()