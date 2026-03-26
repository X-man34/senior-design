from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imu_pipeline.battery_sizing import (
    BatteryOption,
    MotorOption,
    ProcessedGameSignal,
    SignalProcessingAssumptions,
    VehicleAssumptions,
    build_representative_session,
    compute_longitudinal_dynamics,
    integrate_energy_wh,
    integrate_speed,
    iterate_battery_mass,
    preprocess_game_csv,
    run_battery_sizing_pipeline,
)


def make_processed_signal(
    frame: pd.DataFrame,
    sample_period_s: float = 0.1,
    game_name: str = "SyntheticGame",
) -> ProcessedGameSignal:
    return ProcessedGameSignal(
        game_name=game_name,
        sample_hz=1.0 / sample_period_s,
        sample_period_s=sample_period_s,
        start_time=pd.Timestamp("2026-03-25T00:00:00"),
        forward_axis=(1.0, 0.0, 0.0),
        winsor_limit_m_s2=5.0,
        clipped_positive_samples=0,
        clipped_negative_samples=0,
        frame=frame,
    )


def test_constant_acceleration_energy_matches_expected_kinetic_energy() -> None:
    dt_s = 0.01
    duration_s = 10.0
    samples = int(duration_s / dt_s)
    accel = pd.Series([1.0] * samples, dtype=float)
    speed = integrate_speed(accel.to_numpy(), dt_s, v_max_m_s=100.0)

    vehicle = VehicleAssumptions(
        rider_mass_kg=60.0,
        chair_mass_without_battery_kg=20.0,
        c_rr=0.0,
        air_density_kg_m3=0.0,
        cd_area_m2=0.0,
        grade_rad=0.0,
        aux_power_w=0.0,
        gravity_m_s2=9.80665,
    )
    motor = MotorOption(
        name="ideal",
        motor_mass_kg=0.0,
        driven_wheels=2,
        wheel_radius_m=0.3,
        gear_ratio=1.0,
        gear_efficiency=1.0,
        torque_constant_nm_per_a=1.0,
        continuous_current_a=1_000.0,
        peak_current_a=1_000.0,
        drivetrain_efficiency=1.0,
    )

    dynamics = compute_longitudinal_dynamics(accel.to_numpy(), speed, vehicle, motor, battery_mass_kg=0.0)
    energy_wh = integrate_energy_wh(dynamics["battery_power_w"], dt_s)
    expected_wh = 0.5 * (vehicle.rider_mass_kg + vehicle.chair_mass_without_battery_kg) * (duration_s ** 2) / 3600.0

    assert energy_wh == pytest.approx(expected_wh, rel=0.02)


def test_constant_speed_energy_matches_hand_calculation() -> None:
    dt_s = 0.5
    samples = 120
    accel = pd.Series([0.0] * samples, dtype=float)
    speed = pd.Series([2.0] * samples, dtype=float)

    vehicle = VehicleAssumptions(
        rider_mass_kg=70.0,
        chair_mass_without_battery_kg=30.0,
        c_rr=0.02,
        air_density_kg_m3=0.0,
        cd_area_m2=0.0,
        grade_rad=0.0,
        aux_power_w=0.0,
    )
    motor = MotorOption(
        name="ideal",
        motor_mass_kg=0.0,
        driven_wheels=2,
        wheel_radius_m=0.3,
        gear_ratio=1.0,
        gear_efficiency=1.0,
        torque_constant_nm_per_a=1.0,
        continuous_current_a=1_000.0,
        peak_current_a=1_000.0,
        drivetrain_efficiency=1.0,
    )

    dynamics = compute_longitudinal_dynamics(accel.to_numpy(), speed.to_numpy(), vehicle, motor, battery_mass_kg=0.0)
    energy_wh = integrate_energy_wh(dynamics["battery_power_w"], dt_s)
    expected_force_n = vehicle.c_rr * (vehicle.rider_mass_kg + vehicle.chair_mass_without_battery_kg) * vehicle.gravity_m_s2
    expected_energy_wh = expected_force_n * 2.0 * (samples * dt_s) / 3600.0

    assert energy_wh == pytest.approx(expected_energy_wh)


def test_negative_wheel_power_uses_aux_only_without_regen() -> None:
    accel = pd.Series([-1.0, -1.0, -1.0], dtype=float)
    speed = pd.Series([2.0, 2.0, 2.0], dtype=float)

    vehicle = VehicleAssumptions(
        rider_mass_kg=70.0,
        chair_mass_without_battery_kg=20.0,
        c_rr=0.0,
        air_density_kg_m3=0.0,
        cd_area_m2=0.0,
        grade_rad=0.0,
        aux_power_w=25.0,
    )
    motor = MotorOption(
        name="ideal",
        motor_mass_kg=0.0,
        driven_wheels=2,
        wheel_radius_m=0.3,
        gear_ratio=1.0,
        gear_efficiency=1.0,
        torque_constant_nm_per_a=1.0,
        continuous_current_a=1_000.0,
        peak_current_a=1_000.0,
        drivetrain_efficiency=1.0,
    )

    dynamics = compute_longitudinal_dynamics(accel.to_numpy(), speed.to_numpy(), vehicle, motor, battery_mass_kg=0.0)

    assert dynamics["wheel_power_w"].max() < 0.0
    assert dynamics["battery_power_w"].tolist() == [25.0, 25.0, 25.0]


def test_battery_mass_iteration_converges_and_history_increases() -> None:
    sample_period_s = 0.1
    frame = pd.DataFrame(
        {
            "loggingTime(txt)": pd.date_range("2026-03-25", periods=300, freq="100ms"),
            "time_s": [index * sample_period_s for index in range(300)],
            "raw_forward_accel_m_s2": [0.0] * 300,
            "winsorized_forward_accel_m_s2": [0.0] * 300,
            "filtered_forward_accel_m_s2": [0.0] * 300,
            "bias_accel_m_s2": [0.0] * 300,
            "forward_accel_m_s2": [0.0] * 300,
        }
    )
    processed = make_processed_signal(frame, sample_period_s=sample_period_s)
    signal = SignalProcessingAssumptions(representative_minutes=0.5, session_hours=1.0 / 60.0, v_max_m_s=6.0)
    session = build_representative_session(processed, signal)
    session["surrogate_speed_m_s"] = 2.0

    vehicle = VehicleAssumptions(
        rider_mass_kg=70.0,
        chair_mass_without_battery_kg=25.0,
        initial_battery_mass_guess_kg=0.001,
        convergence_tol_kg=0.001,
        max_iterations=20,
        c_rr=0.02,
        air_density_kg_m3=0.0,
        cd_area_m2=0.0,
        aux_power_w=0.0,
    )
    motor = MotorOption(
        name="test_motor",
        motor_mass_kg=2.0,
        driven_wheels=2,
        wheel_radius_m=0.3,
        gear_ratio=1.0,
        gear_efficiency=1.0,
        torque_constant_nm_per_a=1.0,
        continuous_current_a=1_000.0,
        peak_current_a=1_000.0,
        drivetrain_efficiency=1.0,
    )
    battery = BatteryOption(
        name="test_battery",
        specific_energy_wh_per_kg=100.0,
        usable_fraction=1.0,
        continuous_c=5.0,
        peak_c=10.0,
    )

    result, _ = iterate_battery_mass(session, processed, signal, vehicle, motor, battery)

    assert result.converged is True
    assert result.battery_iteration_count >= 1
    assert list(result.battery_mass_history_kg) == sorted(result.battery_mass_history_kg)


def test_representative_profile_builder_matches_requested_duration() -> None:
    sample_period_s = 0.5
    frame = pd.DataFrame(
        {
            "loggingTime(txt)": pd.date_range("2026-03-25", periods=6, freq="500ms"),
            "time_s": [index * sample_period_s for index in range(6)],
            "raw_forward_accel_m_s2": [0.1, 0.2, 0.1, -0.1, 0.0, 0.05],
            "winsorized_forward_accel_m_s2": [0.1, 0.2, 0.1, -0.1, 0.0, 0.05],
            "filtered_forward_accel_m_s2": [0.1, 0.2, 0.1, -0.1, 0.0, 0.05],
            "bias_accel_m_s2": [0.0] * 6,
            "forward_accel_m_s2": [0.1, 0.2, 0.1, -0.1, 0.0, 0.05],
        }
    )
    processed = make_processed_signal(frame, sample_period_s=sample_period_s)
    signal = SignalProcessingAssumptions(representative_minutes=1.0, session_hours=0.05, v_max_m_s=6.0)

    session = build_representative_session(processed, signal)

    expected_samples = int(round((signal.session_hours * 3600.0) / sample_period_s))
    assert len(session) == expected_samples
    assert session["time_s"].iloc[-1] == pytest.approx((expected_samples - 1) * sample_period_s)


def test_real_cleaned_games_pipeline_runs_without_nans(tmp_path: Path) -> None:
    vehicle = VehicleAssumptions()
    signal = SignalProcessingAssumptions()
    motors = [
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
        )
    ]
    batteries = [
        BatteryOption(
            name="nmc_high_rate",
            specific_energy_wh_per_kg=160.0,
            usable_fraction=0.90,
            continuous_c=3.0,
            peak_c=5.0,
        )
    ]

    results = run_battery_sizing_pipeline(
        input_dir=Path("data/processed/clean_games"),
        output_dir=tmp_path,
        vehicle=vehicle,
        signal=signal,
        motors=motors,
        batteries=batteries,
        write_timeseries=False,
        write_plots=False,
    )

    summary = pd.read_csv(tmp_path / "scenario_summary.csv")
    assert {result.game_name for result in results} == {"Game1CharlesPhone", "Game2CharlesPhone"}
    assert set(summary["game_name"]) == {"Game1CharlesPhone", "Game2CharlesPhone"}
    assert not summary.isna().any().any()
    for path in sorted(Path("data/processed/clean_games").glob("*.csv")):
        processed = preprocess_game_csv(path, signal, gravity_m_s2=vehicle.gravity_m_s2)
        assert np.isfinite(processed.forward_axis).all()
