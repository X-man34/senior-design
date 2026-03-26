#!/usr/bin/env python3
"""Rotate IMU axes so average gravity aligns with +Z."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imu_pipeline.battery_sizing import align_vectors_to_average_gravity  # noqa: E402
from imu_pipeline.io import load_game_csv  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INPUT_DIR = Path("data/processed/clean_games")
OUTPUT_DIR = Path("data/processed/aligned_gravity_frame")
G = 9.80665


def summarize_and_plot(path: Path) -> dict[str, object]:
    frame = load_game_csv(path)
    timestamps = pd.to_datetime(frame["loggingTime(txt)"])
    elapsed_min = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60.0

    raw_accel = frame[
        [
            "accelerometerAccelerationX(G)",
            "accelerometerAccelerationY(G)",
            "accelerometerAccelerationZ(G)",
        ]
    ].to_numpy(dtype=float) * G
    gravity = frame[
        [
            "motionGravityX(G)",
            "motionGravityY(G)",
            "motionGravityZ(G)",
        ]
    ].to_numpy(dtype=float) * G
    user_accel = frame[
        [
            "motionUserAccelerationX(G)",
            "motionUserAccelerationY(G)",
            "motionUserAccelerationZ(G)",
        ]
    ].to_numpy(dtype=float) * G

    raw_aligned, rotation = align_vectors_to_average_gravity(raw_accel, gravity)
    gravity_aligned, _ = align_vectors_to_average_gravity(gravity, gravity)
    user_aligned, _ = align_vectors_to_average_gravity(user_accel, gravity)

    means = {
        "file": path.name,
        "rotation_matrix": rotation.tolist(),
        "raw_mean_before_m_s2": raw_accel.mean(axis=0).tolist(),
        "raw_mean_after_m_s2": raw_aligned.mean(axis=0).tolist(),
        "gravity_mean_before_m_s2": gravity.mean(axis=0).tolist(),
        "gravity_mean_after_m_s2": gravity_aligned.mean(axis=0).tolist(),
        "user_mean_before_m_s2": user_accel.mean(axis=0).tolist(),
        "user_mean_after_m_s2": user_aligned.mean(axis=0).tolist(),
    }

    start_min = 13.0 if "Game1" in path.name else 10.0
    end_min = start_min + 2.0
    mask = elapsed_min.between(start_min, end_min)
    time_window = elapsed_min.loc[mask].reset_index(drop=True)
    raw_window = raw_aligned[mask.to_numpy()]

    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    figure.suptitle(
        f"{path.stem.replace('_clean', '')}: rotated frame with average gravity aligned to +Z",
        fontsize=17,
        weight="bold",
    )

    labels = [("X'", "#4c78a8"), ("Y'", "#f58518"), ("Z'", "#54a24b")]
    for axis, (label, color), column in zip(axes, labels, range(3)):
        axis.plot(time_window, raw_window[:, column], color=color, linewidth=0.8)
        axis.axhline(0.0 if column < 2 else 9.80665, color="#444444", linewidth=0.9, alpha=0.8)
        axis.set_ylabel(f"{label}\n(m/s^2)")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Elapsed time in cleaned game (min)")

    output_plot = OUTPUT_DIR / f"{path.stem}_aligned_xyz_window.png"
    figure.savefig(output_plot, dpi=180, bbox_inches="tight")
    plt.close(figure)
    means["window_plot"] = output_plot.name
    return means


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [summarize_and_plot(path) for path in sorted(INPUT_DIR.glob("*.csv"))]
    summary_path = OUTPUT_DIR / "aligned_gravity_summary.json"
    summary_path.write_text(pd.Series(summaries).to_json(indent=2), encoding="utf-8")

    for summary in summaries:
        print(summary["file"])
        print("  raw mean before (m/s^2):", [round(value, 4) for value in summary["raw_mean_before_m_s2"]])
        print("  raw mean after  (m/s^2):", [round(value, 4) for value in summary["raw_mean_after_m_s2"]])
        print("  gravity before  (m/s^2):", [round(value, 4) for value in summary["gravity_mean_before_m_s2"]])
        print("  gravity after   (m/s^2):", [round(value, 4) for value in summary["gravity_mean_after_m_s2"]])
        print("  user accel after(m/s^2):", [round(value, 4) for value in summary["user_mean_after_m_s2"]])
    print(f"\nWrote aligned-frame outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
