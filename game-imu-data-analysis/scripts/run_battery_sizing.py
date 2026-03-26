#!/usr/bin/env python3
"""Run battery sizing from processed gameplay IMU traces."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imu_pipeline.battery_sizing import (
    BatteryOption,
    MotorOption,
    SignalProcessingAssumptions,
    VehicleAssumptions,
    print_console_summary,
    run_battery_sizing_pipeline,
)


INPUT_DIR = Path("data/processed/clean_games")
OUTPUT_DIR = Path("data/processed/battery_sizing")

# These are practical first-pass assumptions for a powered sports chair concept.
VEHICLE = VehicleAssumptions(
    rider_mass_kg=80.0,
    chair_mass_without_battery_kg=35.0,
    pack_voltage_v=48.0,
    c_rr=0.015,
    air_density_kg_m3=1.225,
    cd_area_m2=0.45,
    grade_rad=0.0,
    aux_power_w=40.0,
    equiv_rotational_inertia_kg_m2=0.0,
    initial_battery_mass_guess_kg=5.0,
    convergence_tol_kg=0.05,
    max_iterations=20,
)

SIGNAL = SignalProcessingAssumptions(
    resample_hz=100.0,
    winsor_percentile=99.9,
    lowpass_cutoff_hz=0.5,
    lowpass_order=4,
    bias_window_s=20.0,
    v_max_m_s=6.0,
    representative_minutes=60.0,
    session_hours=2.0,
    forward_axis_override=None,
)

MOTORS = [
    MotorOption(
        name="baseline_48v",
        motor_mass_kg=3.0,
        driven_wheels=2,
        wheel_radius_m=0.305,
        gear_ratio=12.0,
        gear_efficiency=0.92,
        torque_constant_nm_per_a=0.11,
        continuous_current_a=40.0,
        peak_current_a=80.0,
        drivetrain_efficiency=0.80,
    ),
    MotorOption(
        name="high_torque_48v",
        motor_mass_kg=3.5,
        driven_wheels=2,
        wheel_radius_m=0.305,
        gear_ratio=14.0,
        gear_efficiency=0.92,
        torque_constant_nm_per_a=0.13,
        continuous_current_a=60.0,
        peak_current_a=120.0,
        drivetrain_efficiency=0.78,
    ),
]

BATTERIES = [
    BatteryOption(
        name="nmc_high_rate",
        specific_energy_wh_per_kg=160.0,
        usable_fraction=0.90,
        continuous_c=3.0,
        peak_c=5.0,
    ),
    BatteryOption(
        name="lifepo4_high_rate",
        specific_energy_wh_per_kg=110.0,
        usable_fraction=0.85,
        continuous_c=2.0,
        peak_c=3.0,
    ),
    BatteryOption(
        name="sla_baseline",
        specific_energy_wh_per_kg=35.0,
        usable_fraction=0.60,
        continuous_c=0.5,
        peak_c=1.0,
    ),
]


def main() -> None:
    results = run_battery_sizing_pipeline(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        vehicle=VEHICLE,
        signal=SIGNAL,
        motors=MOTORS,
        batteries=BATTERIES,
        write_timeseries=True,
        write_plots=True,
    )
    print_console_summary(results)
    print(f"\nWrote summary outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
