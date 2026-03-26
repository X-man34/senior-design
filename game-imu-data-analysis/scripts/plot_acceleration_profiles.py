#!/usr/bin/env python3
"""Create presentation-friendly acceleration plots from processed gameplay IMU data."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imu_pipeline.battery_sizing import preprocess_game_csv  # noqa: E402
from imu_pipeline.io import load_game_csv  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INPUT_DIR = Path("data/processed/clean_games")
OUTPUT_DIR = Path("data/processed/acceleration_review")

G = 9.80665


def build_horizontal_magnitude(path: Path) -> pd.DataFrame:
    frame = load_game_csv(path)
    timestamps = pd.to_datetime(frame["loggingTime(txt)"])
    elapsed_min = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60.0

    accel = frame[
        [
            "motionUserAccelerationX(G)",
            "motionUserAccelerationY(G)",
            "motionUserAccelerationZ(G)",
        ]
    ].to_numpy(dtype=float) * G
    gravity = frame[
        [
            "motionGravityX(G)",
            "motionGravityY(G)",
            "motionGravityZ(G)",
        ]
    ].to_numpy(dtype=float)
    gravity_hat = gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
    vertical = np.sum(accel * gravity_hat, axis=1)
    horizontal = accel - vertical[:, None] * gravity_hat
    horizontal_mag = np.linalg.norm(horizontal, axis=1)

    return pd.DataFrame(
        {
            "loggingTime(txt)": timestamps,
            "elapsed_min": elapsed_min,
            "horizontal_accel_mag_m_s2": horizontal_mag,
        }
    )


def plot_game(path: Path) -> None:
    magnitude_signal = build_horizontal_magnitude(path)
    signed_signal = preprocess_game_csv(path, assumptions=_signed_signal_assumptions())

    signed_frame = signed_signal.frame.copy()
    signed_frame["elapsed_min"] = signed_frame["time_s"] / 60.0

    strongest_positive_idx = int(np.argmax(signed_frame["forward_accel_m_s2"].to_numpy()))
    strongest_negative_idx = int(np.argmin(signed_frame["forward_accel_m_s2"].to_numpy()))
    peak_windows = []
    for peak_index in [strongest_positive_idx, strongest_negative_idx]:
        center_time = float(signed_frame.iloc[peak_index]["elapsed_min"])
        peak_windows.append((max(0.0, center_time - 0.12), center_time + 0.12))

    figure = plt.figure(figsize=(15, 11), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0])
    clipped_ylim = float(np.percentile(magnitude_signal["horizontal_accel_mag_m_s2"], 99.5) * 1.15)

    ax_main = figure.add_subplot(grid[0, :])
    ax_main.plot(
        magnitude_signal["elapsed_min"],
        magnitude_signal["horizontal_accel_mag_m_s2"],
        color="#4c78a8",
        linewidth=0.8,
        alpha=0.35,
        label="Horizontal acceleration magnitude",
    )
    ax_main.plot(
        signed_frame["elapsed_min"],
        signed_frame["forward_accel_m_s2"].abs(),
        color="#f58518",
        linewidth=1.0,
        alpha=0.8,
        label="Filtered demand signal used by current model",
    )
    ax_main.set_title(f"{path.stem.replace('_clean', '')}: acceleration demand over time", fontsize=15, pad=12)
    ax_main.set_xlabel("Time in cleaned game (min)")
    ax_main.set_ylabel("Acceleration (m/s^2)")
    ax_main.set_ylim(0.0, clipped_ylim)
    ax_main.grid(alpha=0.25)
    ax_main.legend(loc="upper right")
    ax_main.text(
        0.995,
        0.92,
        "Y-axis clipped near 99.5th percentile\nso normal gameplay bursts stay visible",
        transform=ax_main.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    ax_hist = figure.add_subplot(grid[1, 0])
    ax_hist.hist(
        magnitude_signal["horizontal_accel_mag_m_s2"],
        bins=80,
        color="#4c78a8",
        alpha=0.75,
        density=True,
        label="Horizontal magnitude",
    )
    ax_hist.hist(
        np.abs(signed_frame["forward_accel_m_s2"]),
        bins=80,
        color="#f58518",
        alpha=0.55,
        density=True,
        label="Filtered model demand",
    )
    ax_hist.set_title("Distribution of acceleration magnitudes")
    ax_hist.set_xlabel("Acceleration (m/s^2)")
    ax_hist.set_ylabel("Density")
    ax_hist.grid(alpha=0.2)
    ax_hist.legend(loc="upper right")

    ax_signed = figure.add_subplot(grid[1, 1])
    signed_minutes = signed_frame["elapsed_min"]
    signed_values = signed_frame["forward_accel_m_s2"]
    ax_signed.plot(signed_minutes, signed_values, color="#54a24b", linewidth=0.7, alpha=0.9)
    ax_signed.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.8)
    ax_signed.set_title("Signed filtered acceleration view")
    ax_signed.set_xlabel("Time in cleaned game (min)")
    ax_signed.set_ylabel("Signed accel (m/s^2)")
    ax_signed.grid(alpha=0.25)

    zoom_titles = ["Strongest positive burst", "Strongest negative burst"]
    for axis, window, zoom_title in zip(
        [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])],
        peak_windows,
        zoom_titles,
    ):
        mask = signed_frame["elapsed_min"].between(window[0], window[1])
        axis.plot(
            signed_frame.loc[mask, "elapsed_min"],
            signed_frame.loc[mask, "forward_accel_m_s2"],
            color="#e45756",
            linewidth=1.6,
            label="Signed filtered acceleration",
        )
        axis.plot(
            signed_frame.loc[mask, "elapsed_min"],
            signed_frame.loc[mask, "filtered_forward_accel_m_s2"],
            color="#72b7b2",
            linewidth=1.0,
            alpha=0.8,
            label="Before rolling-bias removal",
        )
        axis.axhline(0.0, color="#444444", linewidth=0.9, alpha=0.8)
        axis.set_xlim(window)
        axis.set_title(f"{zoom_title}: {window[0]:.2f} to {window[1]:.2f} min")
        axis.set_xlabel("Time (min)")
        axis.set_ylabel("Signed accel (m/s^2)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)

    figure.suptitle(
        "Acceleration review: magnitude trace, signed trace, and peak-event zooms",
        fontsize=17,
        weight="bold",
    )
    output_path = OUTPUT_DIR / f"{path.stem}_acceleration_review.png"
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _signed_signal_assumptions():
    from imu_pipeline.battery_sizing import SignalProcessingAssumptions

    return SignalProcessingAssumptions(
        resample_hz=100.0,
        winsor_percentile=99.9,
        lowpass_cutoff_hz=0.5,
        lowpass_order=4,
        bias_window_s=20.0,
        v_max_m_s=11.0 * 0.44704,
        representative_minutes=60.0,
        session_hours=2.0,
        use_acceleration_magnitude=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.csv")):
        plot_game(path)
    print(f"Wrote acceleration review plots to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
