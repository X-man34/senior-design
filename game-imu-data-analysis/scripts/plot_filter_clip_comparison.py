#!/usr/bin/env python3
"""Plot raw, smoothed, bias-corrected, and clipped acceleration for inspection."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imu_pipeline.battery_sizing import preprocess_game_csv  # noqa: E402
from imu_pipeline.io import load_game_csv  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INPUT_PATH = Path("data/processed/clean_games/Game1CharlesPhone_clean.csv")
OUTPUT_PATH = Path("data/processed/acceleration_review/filter_clip_comparison.png")
G = 9.80665


def centered_rolling_median(signal: np.ndarray, window_samples: int) -> np.ndarray:
    return (
        pd.Series(signal)
        .rolling(window_samples, center=True, min_periods=max(1, window_samples // 5))
        .median()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def lowpass(signal: np.ndarray, sample_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    b, a = butter(order, cutoff_hz / (sample_hz / 2.0), btype="low")
    return filtfilt(b, a, signal)


def build_signed_trace(path: Path) -> pd.DataFrame:
    raw_frame = load_game_csv(path)
    timestamps = pd.to_datetime(raw_frame["loggingTime(txt)"])
    elapsed_s = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=float)

    accel = raw_frame[
        [
            "motionUserAccelerationX(G)",
            "motionUserAccelerationY(G)",
            "motionUserAccelerationZ(G)",
        ]
    ].to_numpy(dtype=float) * G
    gravity = raw_frame[
        [
            "motionGravityX(G)",
            "motionGravityY(G)",
            "motionGravityZ(G)",
        ]
    ].to_numpy(dtype=float)
    gravity_hat = gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
    vertical = np.sum(accel * gravity_hat, axis=1)
    horizontal = accel - vertical[:, None] * gravity_hat

    assumptions = _signed_assumptions()
    processed = preprocess_game_csv(path, assumptions)
    axis = np.array(processed.forward_axis, dtype=float)
    raw_signed = horizontal @ axis

    winsor_limit = float(np.percentile(np.abs(raw_signed), 99.9))
    winsorized = np.clip(raw_signed, -winsor_limit, winsor_limit)
    smoothed = lowpass(winsorized, sample_hz=100.0, cutoff_hz=0.5, order=4)
    bias = centered_rolling_median(smoothed, int(round(20.0 * 100.0)))
    bias_corrected = smoothed - bias
    clipped = np.clip(bias_corrected, -2.85, 2.85)

    return pd.DataFrame(
        {
            "elapsed_min": elapsed_s / 60.0,
            "raw_signed_m_s2": raw_signed,
            "winsorized_m_s2": winsorized,
            "smoothed_m_s2": smoothed,
            "bias_m_s2": bias,
            "bias_corrected_m_s2": bias_corrected,
            "clipped_m_s2": clipped,
        }
    )


def pick_windows(frame: pd.DataFrame) -> list[tuple[float, float]]:
    corrected = frame["bias_corrected_m_s2"].to_numpy()
    pos_time = float(frame.iloc[int(np.argmax(corrected))]["elapsed_min"])
    neg_time = float(frame.iloc[int(np.argmin(corrected))]["elapsed_min"])
    return [
        (max(0.0, pos_time - 0.12), pos_time + 0.12),
        (max(0.0, neg_time - 0.12), neg_time + 0.12),
    ]


def make_plot(frame: pd.DataFrame) -> None:
    windows = pick_windows(frame)
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.85, 1.0, 1.0])

    ax_top = fig.add_subplot(grid[0, :])
    plot_slice = slice(None, None, 10)
    ax_top.plot(
        frame["elapsed_min"].to_numpy()[plot_slice],
        frame["raw_signed_m_s2"].to_numpy()[plot_slice],
        color="#9ecae9",
        linewidth=0.7,
        alpha=0.5,
        label="Raw signed acceleration",
    )
    ax_top.plot(
        frame["elapsed_min"].to_numpy()[plot_slice],
        frame["smoothed_m_s2"].to_numpy()[plot_slice],
        color="#f58518",
        linewidth=1.0,
        alpha=0.9,
        label="After 0.5 Hz low-pass smoothing",
    )
    ax_top.plot(
        frame["elapsed_min"].to_numpy()[plot_slice],
        frame["clipped_m_s2"].to_numpy()[plot_slice],
        color="#54a24b",
        linewidth=1.1,
        alpha=0.9,
        label="After bias removal and ±2.85 clip",
    )
    ax_top.axhline(2.85, color="#e45756", linestyle="--", linewidth=1.0, alpha=0.9, label="Clip threshold")
    ax_top.axhline(-2.85, color="#e45756", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_top.set_title("Where smoothing and clipping happen in the current pipeline")
    ax_top.set_xlabel("Time in cleaned game (min)")
    ax_top.set_ylabel("Signed acceleration (m/s^2)")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="upper right")

    zoom_center = 20.0
    zoom_window = (zoom_center - 1.0, zoom_center + 1.0)
    zoom_mask = frame["elapsed_min"].between(zoom_window[0], zoom_window[1])
    ax_zoom = fig.add_subplot(grid[1, :])
    ax_zoom.plot(
        frame.loc[zoom_mask, "elapsed_min"],
        frame.loc[zoom_mask, "raw_signed_m_s2"],
        color="#9ecae9",
        linewidth=0.9,
        alpha=0.6,
        label="Raw signed acceleration",
    )
    ax_zoom.plot(
        frame.loc[zoom_mask, "elapsed_min"],
        frame.loc[zoom_mask, "smoothed_m_s2"],
        color="#f58518",
        linewidth=1.0,
        alpha=0.95,
        label="After 0.5 Hz low-pass smoothing",
    )
    ax_zoom.plot(
        frame.loc[zoom_mask, "elapsed_min"],
        frame.loc[zoom_mask, "clipped_m_s2"],
        color="#54a24b",
        linewidth=1.2,
        alpha=0.95,
        label="After bias removal and ±2.85 clip",
    )
    ax_zoom.axhline(2.85, color="#e45756", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_zoom.axhline(-2.85, color="#e45756", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_zoom.set_xlim(zoom_window)
    ax_zoom.set_title("2-minute zoomed view of the same pipeline")
    ax_zoom.set_xlabel("Time in cleaned game (min)")
    ax_zoom.set_ylabel("Signed acceleration (m/s^2)")
    ax_zoom.grid(alpha=0.25)
    ax_zoom.legend(loc="upper right")

    ax_mid_left = fig.add_subplot(grid[2, 0])
    ax_mid_left.hist(frame["raw_signed_m_s2"], bins=100, density=True, alpha=0.45, color="#9ecae9", label="Raw")
    ax_mid_left.hist(frame["smoothed_m_s2"], bins=100, density=True, alpha=0.55, color="#f58518", label="Smoothed")
    ax_mid_left.hist(frame["clipped_m_s2"], bins=100, density=True, alpha=0.55, color="#54a24b", label="Clipped")
    ax_mid_left.set_title("Distribution shift through the pipeline")
    ax_mid_left.set_xlabel("Signed acceleration (m/s^2)")
    ax_mid_left.set_ylabel("Density")
    ax_mid_left.grid(alpha=0.2)
    ax_mid_left.legend(loc="upper right")

    ax_mid_right = fig.add_subplot(grid[2, 1])
    ax_mid_right.plot(frame["elapsed_min"], frame["smoothed_m_s2"], color="#f58518", linewidth=0.9, label="Smoothed")
    ax_mid_right.plot(frame["elapsed_min"], frame["bias_m_s2"], color="#72b7b2", linewidth=1.0, label="Rolling bias")
    ax_mid_right.plot(
        frame["elapsed_min"], frame["bias_corrected_m_s2"], color="#e45756", linewidth=0.9, alpha=0.9, label="Bias corrected"
    )
    ax_mid_right.axhline(0.0, color="#444444", linewidth=0.9)
    ax_mid_right.set_title("Bias removal step")
    ax_mid_right.set_xlabel("Time in cleaned game (min)")
    ax_mid_right.set_ylabel("Signed acceleration (m/s^2)")
    ax_mid_right.grid(alpha=0.25)
    ax_mid_right.legend(loc="upper right")

    labels = ["Strongest positive event", "Strongest negative event"]
    for axis, window, label in zip(
        [fig.add_subplot(grid[3, 0]), fig.add_subplot(grid[3, 1])],
        windows,
        labels,
    ):
        mask = frame["elapsed_min"].between(window[0], window[1])
        axis.plot(frame.loc[mask, "elapsed_min"], frame.loc[mask, "raw_signed_m_s2"], color="#9ecae9", alpha=0.6, label="Raw")
        axis.plot(frame.loc[mask, "elapsed_min"], frame.loc[mask, "smoothed_m_s2"], color="#f58518", linewidth=1.0, label="Smoothed")
        axis.plot(
            frame.loc[mask, "elapsed_min"],
            frame.loc[mask, "bias_corrected_m_s2"],
            color="#e45756",
            linewidth=1.2,
            label="Bias corrected",
        )
        axis.plot(frame.loc[mask, "elapsed_min"], frame.loc[mask, "clipped_m_s2"], color="#54a24b", linewidth=1.4, label="Final clipped")
        axis.axhline(2.85, color="#777777", linestyle="--", linewidth=0.9)
        axis.axhline(-2.85, color="#777777", linestyle="--", linewidth=0.9)
        axis.axhline(0.0, color="#444444", linewidth=0.9)
        axis.set_xlim(window)
        axis.set_title(label)
        axis.set_xlabel("Time (min)")
        axis.set_ylabel("Signed acceleration (m/s^2)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)

    fig.suptitle("Acceleration filtering pipeline review", fontsize=17, weight="bold")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _signed_assumptions():
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
        max_realistic_accel_m_s2=2.85,
    )


def main() -> None:
    frame = build_signed_trace(INPUT_PATH)
    make_plot(frame)
    print(f"Wrote filter/clip comparison plot to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
