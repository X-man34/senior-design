#!/usr/bin/env python3
"""Trim raw game files and keep only conservative gameplay-focused sensor columns."""

from __future__ import annotations

import json
from pathlib import Path

from imu_pipeline.io import CORE_GAMEPLAY_COLUMNS, TrimSpec, TrimWindow, load_game_csv, trim_game_data


TRIM_SPECS: dict[str, TrimSpec] = {
    "Game1CharlesPhone": TrimSpec(
        keep_start_min=4.0,
        keep_end_min=57.5,
        remove_windows=(TrimWindow(14.5, 18.5),),
    ),
    "Game2CharlesPhone": TrimSpec(
        keep_start_min=1.0,
        keep_end_min=38.4,
        remove_windows=(),
    ),
}


def build_manifest(raw_rows: int, cleaned_rows: int, spec: TrimSpec) -> dict:
    """Build a JSON-safe record of the cleaning decisions."""
    return {
        "kept_window_minutes": {
            "start": spec.keep_start_min,
            "end": spec.keep_end_min,
        },
        "removed_windows_minutes": [
            {"start": window.start_min, "end": window.end_min}
            for window in spec.remove_windows
        ],
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "rows_removed": raw_rows - cleaned_rows,
        "columns_kept": CORE_GAMEPLAY_COLUMNS + ["elapsed_min_from_trim_start"],
    }


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed/clean_games")
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    for stem, spec in TRIM_SPECS.items():
        raw_path = raw_dir / f"{stem}.csv"
        game = load_game_csv(raw_path, columns=CORE_GAMEPLAY_COLUMNS)
        cleaned = trim_game_data(game, spec)
        cleaned["loggingTime(txt)"] = cleaned["loggingTime(txt)"].dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%f%z"
        )
        cleaned.to_csv(processed_dir / f"{stem}_clean.csv", index=False)
        manifest[stem] = build_manifest(len(game), len(cleaned), spec)

    (processed_dir / "cleaning_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
