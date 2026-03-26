#!/usr/bin/env python3
"""Estimate required motor, power, and pack current across candidate voltages."""

from __future__ import annotations

from pathlib import Path
import sys

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


INPUT_DIR = Path("data/processed/clean_games")
OUTPUT_DIR = Path("data/processed/motor_requirements")

VOLTAGE_CANDIDATES_V = [24.0, 36.0, 48.0, 60.0, 72.0]

VEHICLE = VehicleAssumptions(
    system_mass_kg=105.0,
    pack_voltage_v=48.0,
    c_rr=0.002,
    air_density_kg_m3=1.225,
    cd_area_m2=0.45,
    grade_rad=0.0,
    aux_power_w=40.0,
    wheel_rotational_inertia_kg_m2_per_wheel=0.2,
)

SIGNAL = SignalProcessingAssumptions(
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

SELECTED_MOTOR = MotorOption(
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in sorted(INPUT_DIR.glob("*.csv")):
        processed = preprocess_game_csv(csv_path, SIGNAL, gravity_m_s2=VEHICLE.gravity_m_s2)
        session = build_representative_session(processed, SIGNAL)
        rows.extend(
            result.to_row()
            for result in summarize_motor_requirements(
                processed_signal=processed,
                session_frame=session,
                vehicle=VEHICLE,
                motor=SELECTED_MOTOR,
                voltage_candidates_v=VOLTAGE_CANDIDATES_V,
                project_peak_torque_per_wheel_nm=39.6,
                project_speed_target_mph=11.0,
            )
        )

    summary = pd.DataFrame(rows).sort_values(["game_name", "voltage_v"])
    summary.to_csv(OUTPUT_DIR / "motor_requirement_summary.csv", index=False)
    summary.to_json(OUTPUT_DIR / "motor_requirement_summary.json", orient="records", indent=2)

    print(summary.to_string(index=False))
    print(f"\nWrote voltage sweep summary to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
