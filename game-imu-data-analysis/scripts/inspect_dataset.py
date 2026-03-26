#!/usr/bin/env python3
"""Lightweight dataset inspection for raw IMU session folders."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imu_pipeline.io import iter_session_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Root directory containing raw session folders.",
    )
    args = parser.parse_args()

    sessions = iter_session_dirs(args.raw_root)
    if not sessions:
        print(f"No session folders found under {Path(args.raw_root).resolve()}")
        return

    for session in sessions:
        print(session)


if __name__ == "__main__":
    main()
