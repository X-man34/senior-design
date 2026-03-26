#!/usr/bin/env python3
"""Create first-pass inspection summaries and plots for raw game CSV exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIME_COL = "loggingTime(txt)"

USER_ACC_COLS = [
    "motionUserAccelerationX(G)",
    "motionUserAccelerationY(G)",
    "motionUserAccelerationZ(G)",
]
ROT_RATE_COLS = [
    "motionRotationRateX(rad/s)",
    "motionRotationRateY(rad/s)",
    "motionRotationRateZ(rad/s)",
]
ACC_COLS = [
    "accelerometerAccelerationX(G)",
    "accelerometerAccelerationY(G)",
    "accelerometerAccelerationZ(G)",
]


def magnitude(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Compute vector magnitude for the selected columns."""
    return np.sqrt((frame[cols] ** 2).sum(axis=1))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add timestamps and simple motion intensity features."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data[TIME_COL])
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["elapsed_s"] = (data["timestamp"] - data["timestamp"].iloc[0]).dt.total_seconds()
    data["elapsed_min"] = data["elapsed_s"] / 60.0
    data["user_acc_mag_g"] = magnitude(data, USER_ACC_COLS)
    data["rot_rate_mag"] = magnitude(data, ROT_RATE_COLS)
    data["acc_mag_g"] = magnitude(data, ACC_COLS)

    # Smooth enough to reveal phases of play without hiding shorter bursts.
    data["user_acc_roll"] = data["user_acc_mag_g"].rolling(window=500, min_periods=1).mean()
    data["rot_rate_roll"] = data["rot_rate_mag"].rolling(window=500, min_periods=1).mean()
    return data


def summarize_game(data: pd.DataFrame) -> dict:
    """Return a compact statistical summary for discussion and review."""
    dt = data["timestamp"].diff().dt.total_seconds().dropna()
    minute_bin = data["elapsed_min"].astype(int)
    minute_summary = (
        data.groupby(minute_bin)[["user_acc_mag_g", "rot_rate_mag"]]
        .mean()
        .rename(columns={"user_acc_mag_g": "mean_user_acc_g", "rot_rate_mag": "mean_rot_rate"})
    )

    return {
        "rows": int(len(data)),
        "start": data["timestamp"].iloc[0].isoformat(),
        "end": data["timestamp"].iloc[-1].isoformat(),
        "duration_minutes": round(float(data["elapsed_min"].iloc[-1]), 2),
        "median_sample_period_ms": round(float(dt.median() * 1000), 3),
        "mean_sample_period_ms": round(float(dt.mean() * 1000), 3),
        "max_sample_gap_s": round(float(dt.max()), 3),
        "user_acc_mag_g": {
            "mean": round(float(data["user_acc_mag_g"].mean()), 4),
            "median": round(float(data["user_acc_mag_g"].median()), 4),
            "p90": round(float(data["user_acc_mag_g"].quantile(0.9)), 4),
            "p99": round(float(data["user_acc_mag_g"].quantile(0.99)), 4),
        },
        "rot_rate_mag": {
            "mean": round(float(data["rot_rate_mag"].mean()), 4),
            "median": round(float(data["rot_rate_mag"].median()), 4),
            "p90": round(float(data["rot_rate_mag"].quantile(0.9)), 4),
            "p99": round(float(data["rot_rate_mag"].quantile(0.99)), 4),
        },
        "lowest_activity_minutes": [
            {
                "minute_from_start": int(idx),
                "mean_user_acc_g": round(float(row["mean_user_acc_g"]), 4),
                "mean_rot_rate": round(float(row["mean_rot_rate"]), 4),
            }
            for idx, row in minute_summary.nsmallest(8, "mean_user_acc_g").iterrows()
        ],
    }


def minute_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate motion features to one row per elapsed minute."""
    minute_bin = data["elapsed_min"].astype(int)
    summary = data.groupby(minute_bin).agg(
        mean_user_acc_g=("user_acc_mag_g", "mean"),
        median_user_acc_g=("user_acc_mag_g", "median"),
        p90_user_acc_g=("user_acc_mag_g", lambda s: s.quantile(0.9)),
        mean_rot_rate=("rot_rate_mag", "mean"),
        median_rot_rate=("rot_rate_mag", "median"),
        p90_rot_rate=("rot_rate_mag", lambda s: s.quantile(0.9)),
    )
    summary.index.name = "minute_from_start"
    return summary.reset_index()


def plot_game(data: pd.DataFrame, title: str, output_path: Path) -> None:
    """Save an overview figure for one game."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(data["elapsed_min"], data["user_acc_mag_g"], color="#9ecae1", alpha=0.25, lw=0.5)
    axes[0].plot(data["elapsed_min"], data["user_acc_roll"], color="#08519c", lw=1.5)
    axes[0].set_ylabel("User Accel (G)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.2)

    axes[1].plot(data["elapsed_min"], data["rot_rate_mag"], color="#fdae6b", alpha=0.25, lw=0.5)
    axes[1].plot(data["elapsed_min"], data["rot_rate_roll"], color="#a63603", lw=1.5)
    axes[1].set_ylabel("Rot Rate (rad/s)")
    axes[1].grid(alpha=0.2)

    axes[2].plot(data["elapsed_min"], data["acc_mag_g"], color="#31a354", lw=0.8)
    axes[2].axhline(1.0, color="black", ls="--", lw=1, alpha=0.6)
    axes[2].set_ylabel("Accel Mag (G)")
    axes[2].set_xlabel("Elapsed Minutes")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_window(data: pd.DataFrame, title: str, output_path: Path, start_min: float, end_min: float) -> None:
    """Save a zoomed view for a selected time window."""
    window = data.loc[data["elapsed_min"].between(start_min, end_min)].copy()
    if window.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    axes[0].plot(window["elapsed_min"], window["user_acc_mag_g"], color="#9ecae1", alpha=0.3, lw=0.5)
    axes[0].plot(window["elapsed_min"], window["user_acc_roll"], color="#08519c", lw=1.6)
    axes[0].set_ylabel("User Accel (G)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.2)

    axes[1].plot(window["elapsed_min"], window["rot_rate_mag"], color="#fdae6b", alpha=0.3, lw=0.5)
    axes[1].plot(window["elapsed_min"], window["rot_rate_roll"], color="#a63603", lw=1.6)
    axes[1].set_ylabel("Rot Rate (rad/s)")
    axes[1].set_xlabel("Elapsed Minutes")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_decision_plots(data: pd.DataFrame, stem: str, output_dir: Path) -> None:
    """Create focused plots that help choose trim windows."""
    total_min = float(data["elapsed_min"].iloc[-1])
    plot_window(
        data,
        f"{stem} Start Window",
        output_dir / f"{stem}_start_window.png",
        0.0,
        min(8.0, total_min),
    )
    plot_window(
        data,
        f"{stem} End Window",
        output_dir / f"{stem}_end_window.png",
        max(0.0, total_min - 8.0),
        total_min,
    )
    if "Game1" in stem:
        plot_window(
            data,
            f"{stem} Mid-Game Window",
            output_dir / f"{stem}_midgame_window.png",
            10.0,
            min(24.0, total_min),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing raw CSV game files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/game_review",
        help="Directory for plots and JSON summaries.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for csv_path in sorted(input_dir.glob("*.csv")):
        frame = pd.read_csv(
            csv_path,
            usecols=[TIME_COL, *USER_ACC_COLS, *ROT_RATE_COLS, *ACC_COLS],
        )
        game = build_features(frame)
        summaries[csv_path.stem] = summarize_game(game)
        plot_game(game, csv_path.stem, output_dir / f"{csv_path.stem}_overview.png")
        save_decision_plots(game, csv_path.stem, output_dir)
        minute_summary(game).round(4).to_csv(
            output_dir / f"{csv_path.stem}_minute_summary.csv",
            index=False,
        )

    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
