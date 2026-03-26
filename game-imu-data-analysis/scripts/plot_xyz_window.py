#!/usr/bin/env python3
"""Plot X/Y/Z user acceleration over a selected gameplay window."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imu_pipeline.io import load_game_csv  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INPUT_PATH = Path("data/processed/clean_games/Game1CharlesPhone_clean.csv")
OUTPUT_DIR = Path("data/processed/acceleration_review")
WINDOW_START_MIN = 13.0
WINDOW_DURATION_MIN = 2.0
G_TO_M_S2 = 9.80665


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_game_csv(INPUT_PATH)
    timestamps = pd.to_datetime(frame["loggingTime(txt)"])
    elapsed_min = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60.0

    window_end_min = WINDOW_START_MIN + WINDOW_DURATION_MIN
    mask = elapsed_min.between(WINDOW_START_MIN, window_end_min)
    window = frame.loc[mask].copy()
    window_elapsed_min = elapsed_min.loc[mask].reset_index(drop=True)

    axis_columns = [
        ("motionUserAccelerationX(G)", "#4c78a8", "X"),
        ("motionUserAccelerationY(G)", "#f58518", "Y"),
        ("motionUserAccelerationZ(G)", "#54a24b", "Z"),
    ]

    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    figure.suptitle(
        "Game1 IMU user acceleration by axis\n2-minute review window starting at 13 minutes",
        fontsize=18,
        weight="bold",
    )

    for subplot, (column, color, label) in zip(axes, axis_columns):
        values_m_s2 = window[column].to_numpy(dtype=float) * G_TO_M_S2
        subplot.plot(window_elapsed_min, values_m_s2, color=color, linewidth=0.8)
        subplot.axhline(0.0, color="#444444", linewidth=0.9, alpha=0.8)
        subplot.set_ylabel(f"{label} accel\n(m/s^2)")
        subplot.grid(alpha=0.25)
        subplot.text(
            0.99,
            0.90,
            f"{label}-axis\npeak abs: {np.max(np.abs(values_m_s2)):.2f} m/s^2",
            transform=subplot.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    axes[-1].set_xlabel("Elapsed time in cleaned game (min)")

    output_path = OUTPUT_DIR / "Game1CharlesPhone_xyz_13_to_15_min.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
