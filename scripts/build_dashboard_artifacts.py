"""
Build pre-computed dashboard artifacts (run once on a capable machine or in CI).

    python scripts/build_dashboard_artifacts.py
    python scripts/build_dashboard_artifacts.py --scenario 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.artifacts import artifacts_exist, save_artifacts  # noqa: E402
from dashboard.pipeline import execute_scenario  # noqa: E402
from src.data.bootstrap_dataset import ensure_online_retail_xlsx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard result cache")
    parser.add_argument("--scenario", type=int, choices=[1, 2], help="Only build one scenario")
    args = parser.parse_args()

    ensure_online_retail_xlsx(ROOT)
    scenarios = [args.scenario] if args.scenario else [1, 2]

    for sid in scenarios:
        print(f"\n=== Building artifacts for scenario {sid} ===")
        stages = execute_scenario(sid)
        path = save_artifacts(sid, stages)
        print(f"Saved {len(stages)} stages -> {path}")
        print(f"Exists check: {artifacts_exist(sid)}")


if __name__ == "__main__":
    main()
