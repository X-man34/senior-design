#!/usr/bin/env python3
"""Analyze the lower-rate HyperIMU recording and compare it to the original cleaned games."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imu_pipeline.battery_sizing import (  # noqa: E402
    MotorOption,
    SignalProcessingAssumptions,
    VehicleAssumptions,
    build_representative_session,
    preprocess_game_csv,
    summarize_motor_requirements,
)
from imu_pipeline.io import load_game_csv, load_hyperimu_csv  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


G = 9.80665
RAW_HYPERIMU_PATH = Path("data/raw/BothGamesCalebPhone.csv")
CHARLES_INPUT_DIR = Path("data/processed/clean_games")
CALEB_OUTPUT_DIR = Path("data/processed/clean_games_caleb")
REPORT_DIR = Path("data/processed/second_dataset_report")
FIGURE_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "second_dataset_report.md"

VOLTAGE_CANDIDATES_V = [24.0, 36.0, 48.0, 60.0, 72.0]

VEHICLE = VehicleAssumptions(
    system_mass_kg=105.0,
    c_rr=0.002,
    air_density_kg_m3=1.225,
    cd_area_m2=0.45,
    grade_rad=0.0,
    aux_power_w=40.0,
    wheel_rotational_inertia_kg_m2_per_wheel=0.2,
)

MOTOR = MotorOption(
    name="450w_bldc_planetary_16to1",
    motor_mass_kg=3.5,
    driven_wheels=2,
    wheel_radius_m=11.75 * 0.0254,
    gear_ratio=16.0,
    gear_efficiency=0.90,
    torque_constant_nm_per_a=1.43 / 11.72,
    continuous_current_a=11.72,
    peak_current_a=4.30 / (1.43 / 11.72),
    motor_efficiency=0.85,
    rated_torque_nm=1.43,
    peak_torque_nm=4.30,
    rated_speed_rpm=3000.0,
)

CHARLES_SIGNAL = SignalProcessingAssumptions(
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

CALEB_SIGNAL = SignalProcessingAssumptions(
    resample_hz=10.0,
    winsor_percentile=99.5,
    lowpass_cutoff_hz=0.35,
    lowpass_order=2,
    bias_window_s=12.0,
    v_max_m_s=11.0 * 0.44704,
    representative_minutes=60.0,
    session_hours=2.0,
    use_acceleration_magnitude=False,
    max_realistic_accel_m_s2=2.85,
)


def hyperimu_to_pipeline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    accel = frame[
        [
            "lsm6dsr_accelerometer.x",
            "lsm6dsr_accelerometer.y",
            "lsm6dsr_accelerometer.z",
        ]
    ].to_numpy(dtype=float)
    linear = frame[
        [
            "linear_acceleration_sensor.x",
            "linear_acceleration_sensor.y",
            "linear_acceleration_sensor.z",
        ]
    ].to_numpy(dtype=float)
    gravity = accel - linear

    converted = pd.DataFrame(
        {
            "loggingTime(txt)": frame["loggingTime(txt)"],
            "motionUserAccelerationX(G)": linear[:, 0] / G,
            "motionUserAccelerationY(G)": linear[:, 1] / G,
            "motionUserAccelerationZ(G)": linear[:, 2] / G,
            "motionGravityX(G)": gravity[:, 0] / G,
            "motionGravityY(G)": gravity[:, 1] / G,
            "motionGravityZ(G)": gravity[:, 2] / G,
        }
    )
    return converted


def extract_caleb_game_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_hyperimu = load_hyperimu_csv(RAW_HYPERIMU_PATH)
    pipeline_frame = hyperimu_to_pipeline_frame(raw_hyperimu)
    CALEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifests = []
    for charles_path in sorted(CHARLES_INPUT_DIR.glob("Game*CharlesPhone_clean.csv")):
        charles_game = load_game_csv(charles_path)
        start_time = pd.to_datetime(charles_game["loggingTime(txt)"].iloc[0])
        end_time = pd.to_datetime(charles_game["loggingTime(txt)"].iloc[-1])
        caleb_slice = pipeline_frame.loc[
            pipeline_frame["loggingTime(txt)"].between(start_time, end_time)
        ].copy()
        game_name = charles_path.name.replace("CharlesPhone", "CalebPhone")
        caleb_slice.to_csv(CALEB_OUTPUT_DIR / game_name, index=False)
        manifests.append(
            {
                "game_name": game_name.replace("_clean.csv", ""),
                "source_rows": len(caleb_slice),
                "start_time": str(caleb_slice["loggingTime(txt)"].iloc[0]) if not caleb_slice.empty else None,
                "end_time": str(caleb_slice["loggingTime(txt)"].iloc[-1]) if not caleb_slice.empty else None,
                "duration_min": (
                    (pd.to_datetime(caleb_slice["loggingTime(txt)"].iloc[-1]) - pd.to_datetime(caleb_slice["loggingTime(txt)"].iloc[0])).total_seconds() / 60.0
                    if len(caleb_slice) > 1
                    else 0.0
                ),
            }
        )

    manifest_frame = pd.DataFrame(manifests)
    manifest_frame.to_csv(REPORT_DIR / "caleb_window_manifest.csv", index=False)
    return raw_hyperimu, manifest_frame


def analyze_dataset(dataset_label: str, input_dir: Path, signal: SignalProcessingAssumptions) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    traces: dict[str, pd.DataFrame] = {}

    for csv_path in sorted(input_dir.glob("Game*.csv")):
        processed = preprocess_game_csv(csv_path, signal, gravity_m_s2=VEHICLE.gravity_m_s2)
        session = build_representative_session(processed, signal)
        trace = processed.frame.copy()
        trace["dataset"] = dataset_label
        trace["game_name"] = processed.game_name
        traces[processed.game_name] = trace

        requirements = summarize_motor_requirements(
            processed_signal=processed,
            session_frame=session,
            vehicle=VEHICLE,
            motor=MOTOR,
            voltage_candidates_v=VOLTAGE_CANDIDATES_V,
            project_peak_torque_per_wheel_nm=39.6,
            project_speed_target_mph=11.0,
        )
        for requirement in requirements:
            row = requirement.to_row()
            row["dataset"] = dataset_label
            rows.append(row)

    return pd.DataFrame(rows), traces


def _plot_trace_comparison(charles_traces: dict[str, pd.DataFrame], caleb_traces: dict[str, pd.DataFrame]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False, constrained_layout=True)
    for axis, game_name in zip(axes, sorted(charles_traces)):
        charles = charles_traces[game_name]
        caleb_name = game_name.replace("CharlesPhone", "CalebPhone")
        caleb = caleb_traces[caleb_name]

        axis.plot(
            charles["time_s"] / 60.0,
            charles["forward_accel_m_s2"],
            color="#4c78a8",
            linewidth=0.9,
            alpha=0.8,
            label="Charles 100 Hz cleaned",
        )
        axis.plot(
            caleb["time_s"] / 60.0,
            caleb["forward_accel_m_s2"],
            color="#f58518",
            linewidth=1.0,
            alpha=0.85,
            label="Caleb 10 Hz clipped",
        )
        axis.set_title(game_name.replace("_clean", ""))
        axis.set_xlabel("Time in game (min)")
        axis.set_ylabel("Signed clipped accel (m/s^2)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    figure.suptitle("Signed clipped acceleration traces: original vs lower-rate dataset", fontsize=17, weight="bold")
    figure.savefig(FIGURE_DIR / "trace_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_distribution_comparison(charles_traces: dict[str, pd.DataFrame], caleb_traces: dict[str, pd.DataFrame]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for axis, game_name in zip(axes, sorted(charles_traces)):
        charles = np.abs(charles_traces[game_name]["forward_accel_m_s2"].to_numpy(dtype=float))
        caleb = np.abs(caleb_traces[game_name.replace("CharlesPhone", "CalebPhone")]["forward_accel_m_s2"].to_numpy(dtype=float))

        axis.hist(charles, bins=70, density=True, alpha=0.65, color="#4c78a8", label="Charles 100 Hz")
        axis.hist(caleb, bins=45, density=True, alpha=0.6, color="#f58518", label="Caleb 10 Hz")
        axis.set_title(game_name.replace("_clean", ""))
        axis.set_xlabel("Absolute clipped accel (m/s^2)")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")

    figure.suptitle("Acceleration distribution comparison after aggressive clipping", fontsize=17, weight="bold")
    figure.savefig(FIGURE_DIR / "distribution_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_voltage_sweep(summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    palette = {"charles_cleaned": "#4c78a8", "caleb_low_rate": "#f58518"}
    for axis, game_name in zip(axes, sorted(summary["game_name"].unique())):
        game_summary = summary.loc[summary["game_name"] == game_name]
        for dataset, group in game_summary.groupby("dataset"):
            axis.plot(
                group["voltage_v"],
                group["peak_pack_current_a"],
                marker="o",
                linewidth=2.0,
                color=palette[dataset],
                label=dataset,
            )
        axis.set_title(game_name)
        axis.set_xlabel("Pack voltage (V)")
        axis.set_ylabel("Peak pack current (A)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    figure.suptitle("Voltage sweep comparison for the 450 W planetary motor assumption", fontsize=17, weight="bold")
    figure.savefig(FIGURE_DIR / "voltage_sweep_current.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_peak_bars(summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    peak_48v = summary.loc[summary["voltage_v"] == 48.0].copy()
    peak_48v["label"] = peak_48v["game_name"] + "\n" + peak_48v["dataset"]

    axes[0].bar(peak_48v["label"], peak_48v["peak_wheel_torque_per_motor_nm"], color=["#4c78a8", "#f58518"] * 2)
    axes[0].axhline(39.6, color="#444444", linestyle="--", linewidth=1.3, label="Project torque target")
    axes[0].set_title("Peak per-wheel torque at 48 V")
    axes[0].set_ylabel("Torque per wheel (Nm)")
    axes[0].tick_params(axis="x", labelrotation=0)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(peak_48v["label"], peak_48v["peak_electrical_power_w"], color=["#4c78a8", "#f58518"] * 2)
    axes[1].set_title("Peak electrical power at 48 V")
    axes[1].set_ylabel("Power (W)")
    axes[1].tick_params(axis="x", labelrotation=0)
    axes[1].grid(alpha=0.25, axis="y")

    figure.suptitle("Peak load comparison at the 48 V design point", fontsize=17, weight="bold")
    figure.savefig(FIGURE_DIR / "peak_load_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def build_report(summary: pd.DataFrame, manifest: pd.DataFrame) -> None:
    report_summary = summary.loc[summary["voltage_v"] == 48.0].copy()
    report_summary = report_summary[
        [
            "dataset",
            "game_name",
            "session_energy_wh",
            "peak_electrical_power_w",
            "peak_pack_current_a",
            "peak_wheel_torque_per_motor_nm",
            "required_peak_motor_torque_nm",
            "required_peak_motor_current_a",
            "clipped_positive_samples",
            "clipped_negative_samples",
        ]
    ]

    def table_markdown(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines = ["| " + " | ".join(headers) + " |", separator]
        for _, row in frame.iterrows():
            values = [f"{value:.2f}" if isinstance(value, float) else str(value) for value in row.tolist()]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    report_text = f"""# Lower-Rate Second Dataset Analysis Report

## Goal

This report reruns the clipped motor and voltage analysis on the new lower-rate HyperIMU recording
([BothGamesCalebPhone.csv](/home/chafri/Documents/seniorDesign/data/raw/BothGamesCalebPhone.csv))
and compares it against the existing cleaned Charles gameplay files.

## Shared Assumptions

- Total system mass: `105 kg`
- Wheel radius: `11.75 in`
- Two-wheel drive with one motor per wheel
- Gear ratio: `16:1`
- Rolling resistance coefficient: `0.002`
- Wheel rotational inertia: `0.2 kg·m^2` per driven wheel
- No slope
- Aggressive clipping: signed acceleration limited to `+/- 2.85 m/s^2`
- Motor assumption: 450 W planetary gear BLDC family used in the earlier study

## How The Second Dataset Was Prepared

The HyperIMU file did not contain explicit timestamps per sample, so it was reconstructed from its
metadata header (`100 ms` sampling interval) and then split into game windows using the same absolute
time windows as the Charles cleaned game files.

### Lower-rate window manifest

{table_markdown(manifest)}

## Key 48 V Comparison

{table_markdown(report_summary)}

## Figures

### Signed clipped acceleration traces

![Trace comparison](figures/trace_comparison.png)

### Acceleration distribution comparison

![Distribution comparison](figures/distribution_comparison.png)

### Peak pack current across candidate voltages

![Voltage sweep](figures/voltage_sweep_current.png)

### Peak torque and power at 48 V

![Peak load comparison](figures/peak_load_comparison.png)

## Interpretation

- The lower-rate dataset is useful as a second opinion because it records the same games with a very different sampling profile.
- Because it is much lower rate than the Charles phone, it will naturally smooth or miss short spikes that the higher-rate data can see.
- That means the Caleb file is not automatically “more correct,” but it is valuable for checking whether conclusions depend entirely on very sharp high-frequency events.
- The aggressive clipping rule prevents either dataset from dominating the analysis with unrealistic collision or body-motion spikes.
- The most important shared outputs for design remain peak wheel torque, peak motor torque/current, peak electrical power, and peak pack current at each candidate voltage.

## Output Files

- Summary CSV: [second_dataset_summary.csv](/home/chafri/Documents/seniorDesign/data/processed/second_dataset_report/second_dataset_summary.csv)
- Summary JSON: [second_dataset_summary.json](/home/chafri/Documents/seniorDesign/data/processed/second_dataset_report/second_dataset_summary.json)
- Report figures: [figures](/home/chafri/Documents/seniorDesign/data/processed/second_dataset_report/figures)
"""
    REPORT_PATH.write_text(report_text, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    _, manifest = extract_caleb_game_windows()
    charles_summary, charles_traces = analyze_dataset("charles_cleaned", CHARLES_INPUT_DIR, CHARLES_SIGNAL)
    caleb_summary, caleb_traces = analyze_dataset("caleb_low_rate", CALEB_OUTPUT_DIR, CALEB_SIGNAL)

    summary = pd.concat([charles_summary, caleb_summary], ignore_index=True).sort_values(
        ["game_name", "dataset", "voltage_v"]
    )
    summary.to_csv(REPORT_DIR / "second_dataset_summary.csv", index=False)
    summary.to_json(REPORT_DIR / "second_dataset_summary.json", orient="records", indent=2)
    (REPORT_DIR / "analysis_assumptions.json").write_text(
        json.dumps(
            {
                "vehicle": asdict(VEHICLE),
                "motor": asdict(MOTOR),
                "charles_signal": asdict(CHARLES_SIGNAL),
                "caleb_signal": asdict(CALEB_SIGNAL),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_trace_comparison(charles_traces, caleb_traces)
    _plot_distribution_comparison(charles_traces, caleb_traces)
    _plot_voltage_sweep(summary)
    _plot_peak_bars(summary)
    build_report(summary, manifest)

    print(summary.to_string(index=False))
    print(f"\nWrote comparison report to {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
