"""Battery sizing analysis derived from processed gameplay IMU traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from imu_pipeline.io import load_game_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USER_ACCEL_COLUMNS = [
    "motionUserAccelerationX(G)",
    "motionUserAccelerationY(G)",
    "motionUserAccelerationZ(G)",
]

GRAVITY_COLUMNS = [
    "motionGravityX(G)",
    "motionGravityY(G)",
    "motionGravityZ(G)",
]


@dataclass(frozen=True)
class SignalProcessingAssumptions:
    """Signal conditioning and profile construction choices."""

    resample_hz: float = 100.0
    winsor_percentile: float = 99.9
    lowpass_cutoff_hz: float = 0.5
    lowpass_order: int = 4
    bias_window_s: float = 20.0
    v_max_m_s: float = 6.0
    representative_minutes: float = 60.0
    session_hours: float = 2.0
    forward_axis_override: tuple[float, float, float] | None = None
    use_acceleration_magnitude: bool = False
    max_realistic_accel_m_s2: float | None = None


@dataclass(frozen=True)
class VehicleAssumptions:
    """Vehicle- and environment-level assumptions for forward dynamics."""

    rider_mass_kg: float = 80.0
    chair_mass_without_battery_kg: float = 35.0
    system_mass_kg: float | None = None
    pack_voltage_v: float = 48.0
    c_rr: float = 0.015
    air_density_kg_m3: float = 1.225
    cd_area_m2: float = 0.45
    grade_rad: float = 0.0
    aux_power_w: float = 40.0
    equiv_rotational_inertia_kg_m2: float = 0.0
    wheel_rotational_inertia_kg_m2_per_wheel: float = 0.0
    gravity_m_s2: float = 9.80665
    initial_battery_mass_guess_kg: float = 5.0
    convergence_tol_kg: float = 0.05
    max_iterations: int = 20


@dataclass(frozen=True)
class MotorOption:
    """Motor and drivetrain scenario for torque/current mapping."""

    name: str
    motor_mass_kg: float
    driven_wheels: int
    wheel_radius_m: float
    gear_ratio: float
    gear_efficiency: float
    torque_constant_nm_per_a: float
    continuous_current_a: float
    peak_current_a: float
    drivetrain_efficiency: float | None = None
    motor_efficiency: float = 1.0
    rated_torque_nm: float | None = None
    peak_torque_nm: float | None = None
    rated_speed_rpm: float | None = None

    def overall_drive_efficiency(self) -> float:
        """Return the battery-to-wheel efficiency used in the simplified model."""

        if self.drivetrain_efficiency is not None:
            return self.drivetrain_efficiency
        return self.motor_efficiency * self.gear_efficiency


@dataclass(frozen=True)
class BatteryOption:
    """Battery pack assumption for energy mass and C-rate checks."""

    name: str
    specific_energy_wh_per_kg: float
    usable_fraction: float
    continuous_c: float
    peak_c: float


@dataclass(frozen=True)
class ProcessedGameSignal:
    """Conditioned gameplay trace before profile repetition."""

    game_name: str
    sample_hz: float
    sample_period_s: float
    start_time: pd.Timestamp
    forward_axis: tuple[float, float, float]
    winsor_limit_m_s2: float
    clipped_positive_samples: int
    clipped_negative_samples: int
    frame: pd.DataFrame


@dataclass(frozen=True)
class BatterySizingResult:
    """Summary metrics for one game x motor x battery scenario."""

    game_name: str
    motor_name: str
    battery_name: str
    representative_profile_minutes: float
    session_duration_hours: float
    usable_energy_wh: float
    nominal_energy_wh: float
    nominal_capacity_ah: float
    battery_mass_kg: float
    total_vehicle_mass_kg: float
    average_battery_power_w: float
    peak_battery_power_w: float
    peak_traction_force_n: float
    peak_wheel_torque_nm: float
    peak_motor_torque_nm: float
    peak_motor_current_a: float
    peak_battery_current_a: float
    peak_battery_c_rate: float
    motor_continuous_current_violation: bool
    motor_peak_current_violation: bool
    battery_continuous_c_violation: bool
    battery_peak_c_violation: bool
    battery_iteration_count: int
    converged: bool
    forward_axis_x: float
    forward_axis_y: float
    forward_axis_z: float
    winsor_limit_m_s2: float
    battery_mass_history_kg: tuple[float, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_summary_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["battery_mass_history_kg"] = json.dumps(self.battery_mass_history_kg)
        row["notes"] = " | ".join(self.notes)
        return row


@dataclass(frozen=True)
class MotorRequirementResult:
    """Load-focused summary without assuming a battery chemistry."""

    game_name: str
    voltage_v: float
    motor_name: str
    session_energy_wh: float
    average_electrical_power_w: float
    peak_electrical_power_w: float
    peak_pack_current_a: float
    peak_acceleration_m_s2: float
    peak_speed_m_s: float
    peak_speed_mph: float
    peak_traction_force_n: float
    peak_wheel_torque_total_nm: float
    peak_wheel_torque_per_motor_nm: float
    required_peak_motor_torque_nm: float
    required_peak_motor_current_a: float
    required_continuous_motor_current_a: float
    clipped_positive_samples: int
    clipped_negative_samples: int
    rated_output_speed_m_s: float
    rated_output_speed_mph: float
    meets_11mph_speed_target: bool
    meets_project_peak_torque_target: bool
    meets_motor_peak_torque: bool | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["notes"] = " | ".join(self.notes)
        return row


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps source onto target."""

    source_unit = _normalize(np.asarray(source, dtype=float))
    target_unit = _normalize(np.asarray(target, dtype=float))
    cross = np.cross(source_unit, target_unit)
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        # 180-degree rotation around any axis orthogonal to source.
        reference = np.array([1.0, 0.0, 0.0])
        if abs(source_unit[0]) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(source_unit, reference))
        return rotation_matrix_from_axis_angle(axis, math.pi)

    axis = cross / cross_norm
    angle = math.atan2(cross_norm, dot)
    return rotation_matrix_from_axis_angle(axis, angle)


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Build a rotation matrix from axis-angle form."""

    unit_axis = _normalize(np.asarray(axis, dtype=float))
    x, y, z = unit_axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def align_vectors_to_average_gravity(vectors: np.ndarray, gravity_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate vectors so the average gravity direction aligns with +Z."""

    gravity_mean = np.mean(np.asarray(gravity_vectors, dtype=float), axis=0)
    rotation = rotation_matrix_from_vectors(gravity_mean, np.array([0.0, 0.0, 1.0]))
    transformed = np.asarray(vectors, dtype=float) @ rotation.T
    return transformed, rotation


def _uniform_resample(frame: pd.DataFrame, resample_hz: float) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame["loggingTime(txt)"])
    elapsed_s = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    dt = 1.0 / resample_hz
    uniform_time = np.arange(0.0, elapsed_s[-1] + (dt / 2.0), dt, dtype=float)

    numeric_columns = [
        column for column in frame.columns if column != "loggingTime(txt)" and frame[column].dtype != "O"
    ]
    resampled: dict[str, Any] = {
        "loggingTime(txt)": timestamps.iloc[0] + pd.to_timedelta(uniform_time, unit="s"),
        "time_s": uniform_time,
    }

    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            raise ValueError(f"Column {column} does not have enough numeric values to resample.")
        resampled[column] = np.interp(uniform_time, elapsed_s[valid], values[valid])

    return pd.DataFrame(resampled)


def _lowpass(signal: np.ndarray, sample_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    if len(signal) < 8 or cutoff_hz <= 0.0:
        return signal.copy()
    normalized_cutoff = cutoff_hz / (sample_hz / 2.0)
    if normalized_cutoff >= 1.0:
        return signal.copy()
    b, a = butter(order, normalized_cutoff, btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal.copy()
    return filtfilt(b, a, signal)


def _winsorize(signal: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    finite_signal = signal[np.isfinite(signal)]
    if finite_signal.size == 0:
        raise ValueError("Signal is empty after filtering non-finite values.")
    limit = float(np.percentile(np.abs(finite_signal), percentile))
    if limit == 0.0:
        return signal.copy(), limit
    return np.clip(signal, -limit, limit), limit


def _centered_rolling_median(signal: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return np.zeros_like(signal)
    min_periods = max(1, window_samples // 5)
    return (
        pd.Series(signal)
        .rolling(window_samples, center=True, min_periods=min_periods)
        .median()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def integrate_speed(accel_m_s2: np.ndarray, dt_s: float | np.ndarray, v_max_m_s: float) -> np.ndarray:
    """Integrate signed acceleration to a surrogate speed bounded to [0, v_max]."""

    if np.isscalar(dt_s):
        dt = np.full(len(accel_m_s2), float(dt_s), dtype=float)
    else:
        dt = np.asarray(dt_s, dtype=float)

    speed = np.zeros(len(accel_m_s2), dtype=float)
    for index in range(1, len(accel_m_s2)):
        candidate = speed[index - 1] + (accel_m_s2[index] * dt[index])
        speed[index] = min(v_max_m_s, max(0.0, candidate))
    return speed


def _repeat_frame_to_duration(frame: pd.DataFrame, duration_s: float, dt_s: float) -> pd.DataFrame:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")

    target_samples = int(round(duration_s / dt_s))
    if target_samples <= 0:
        raise ValueError("Target duration yields zero samples.")

    take_index = np.arange(target_samples, dtype=int) % len(frame)
    repeated = frame.iloc[take_index].reset_index(drop=True).copy()
    repeated["repeat_cycle"] = np.arange(target_samples, dtype=int) // len(frame)
    repeated["source_row_index"] = take_index
    repeated["time_s"] = np.arange(target_samples, dtype=float) * dt_s
    repeated["profile_elapsed_min"] = repeated["time_s"] / 60.0
    if "loggingTime(txt)" in repeated.columns:
        start_time = pd.to_datetime(frame["loggingTime(txt)"].iloc[0])
        repeated["loggingTime(txt)"] = start_time + pd.to_timedelta(repeated["time_s"], unit="s")
    return repeated


def build_representative_session(
    processed_signal: ProcessedGameSignal,
    assumptions: SignalProcessingAssumptions,
) -> pd.DataFrame:
    """Build a repeated 60-minute profile and duplicate it to a 2-hour session."""

    representative_s = assumptions.representative_minutes * 60.0
    session_s = assumptions.session_hours * 3600.0

    hourly_frame = _repeat_frame_to_duration(processed_signal.frame, representative_s, processed_signal.sample_period_s)
    session_frame = _repeat_frame_to_duration(hourly_frame, session_s, processed_signal.sample_period_s)
    session_frame["surrogate_speed_m_s"] = integrate_speed(
        session_frame["forward_accel_m_s2"].to_numpy(dtype=float),
        processed_signal.sample_period_s,
        assumptions.v_max_m_s,
    )
    return session_frame


def _estimate_forward_axis(
    horizontal_accel_m_s2: np.ndarray,
    sample_hz: float,
    assumptions: SignalProcessingAssumptions,
) -> np.ndarray:
    if assumptions.forward_axis_override is not None:
        return _normalize(np.asarray(assumptions.forward_axis_override, dtype=float))

    smoothed = np.column_stack(
        [
            _lowpass(horizontal_accel_m_s2[:, column], sample_hz, assumptions.lowpass_cutoff_hz, assumptions.lowpass_order)
            for column in range(horizontal_accel_m_s2.shape[1])
        ]
    )
    covariance = np.cov(smoothed.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    return _normalize(axis)


def preprocess_game_csv(
    path: str | Path,
    assumptions: SignalProcessingAssumptions,
    gravity_m_s2: float = 9.80665,
) -> ProcessedGameSignal:
    """Load, resample, and condition one processed gameplay CSV."""

    raw_frame = load_game_csv(path)
    resampled = _uniform_resample(raw_frame, assumptions.resample_hz)

    accel_m_s2 = resampled[USER_ACCEL_COLUMNS].to_numpy(dtype=float) * gravity_m_s2
    gravity_vectors = resampled[GRAVITY_COLUMNS].to_numpy(dtype=float)
    gravity_unit = gravity_vectors / np.linalg.norm(gravity_vectors, axis=1, keepdims=True)
    vertical_component = np.sum(accel_m_s2 * gravity_unit, axis=1)
    horizontal_accel = accel_m_s2 - (vertical_component[:, None] * gravity_unit)

    if assumptions.use_acceleration_magnitude:
        axis = np.zeros(3, dtype=float)
        raw_forward = np.linalg.norm(horizontal_accel, axis=1)
    else:
        axis = _estimate_forward_axis(horizontal_accel, assumptions.resample_hz, assumptions)
        raw_forward = horizontal_accel @ axis
    winsorized_forward, winsor_limit = _winsorize(raw_forward, assumptions.winsor_percentile)
    filtered_forward = _lowpass(
        winsorized_forward,
        assumptions.resample_hz,
        assumptions.lowpass_cutoff_hz,
        assumptions.lowpass_order,
    )
    bias = _centered_rolling_median(
        filtered_forward,
        int(round(assumptions.bias_window_s * assumptions.resample_hz)),
    )
    forward_accel = filtered_forward - bias
    clipped_positive_samples = 0
    clipped_negative_samples = 0

    if assumptions.max_realistic_accel_m_s2 is not None:
        accel_limit = float(assumptions.max_realistic_accel_m_s2)
        clipped_positive_samples = int(np.sum(forward_accel > accel_limit))
        clipped_negative_samples = int(np.sum(forward_accel < -accel_limit))
        forward_accel = np.clip(forward_accel, -accel_limit, accel_limit)

    positive_distance = float(integrate_speed(forward_accel, 1.0 / assumptions.resample_hz, assumptions.v_max_m_s).sum())
    negative_distance = float(integrate_speed(-forward_accel, 1.0 / assumptions.resample_hz, assumptions.v_max_m_s).sum())
    if not assumptions.use_acceleration_magnitude and negative_distance > positive_distance:
        axis = -axis
        raw_forward = -raw_forward
        winsorized_forward = -winsorized_forward
        filtered_forward = -filtered_forward
        bias = -bias
        forward_accel = -forward_accel
        clipped_positive_samples, clipped_negative_samples = (
            clipped_negative_samples,
            clipped_positive_samples,
        )

    processed_frame = pd.DataFrame(
        {
            "loggingTime(txt)": resampled["loggingTime(txt)"],
            "time_s": resampled["time_s"],
            "raw_forward_accel_m_s2": raw_forward,
            "winsorized_forward_accel_m_s2": winsorized_forward,
            "filtered_forward_accel_m_s2": filtered_forward,
            "bias_accel_m_s2": bias,
            "forward_accel_m_s2": forward_accel,
        }
    )

    game_name = Path(path).stem.replace("_clean", "")
    return ProcessedGameSignal(
        game_name=game_name,
        sample_hz=assumptions.resample_hz,
        sample_period_s=1.0 / assumptions.resample_hz,
        start_time=pd.to_datetime(processed_frame["loggingTime(txt)"].iloc[0]),
        forward_axis=(float(axis[0]), float(axis[1]), float(axis[2])),
        winsor_limit_m_s2=winsor_limit,
        clipped_positive_samples=clipped_positive_samples,
        clipped_negative_samples=clipped_negative_samples,
        frame=processed_frame,
    )


def compute_longitudinal_dynamics(
    accel_m_s2: np.ndarray,
    speed_m_s: np.ndarray,
    vehicle: VehicleAssumptions,
    motor: MotorOption,
    battery_mass_kg: float,
) -> dict[str, np.ndarray | float]:
    """Map target motion to force, torque, current, and battery power."""

    motor_mass_total = motor.motor_mass_kg * motor.driven_wheels
    if vehicle.system_mass_kg is not None:
        total_mass = vehicle.system_mass_kg
    else:
        total_mass = vehicle.rider_mass_kg + vehicle.chair_mass_without_battery_kg + motor_mass_total + battery_mass_kg
    effective_mass = total_mass + (vehicle.equiv_rotational_inertia_kg_m2 / (motor.wheel_radius_m ** 2))

    rolling_force_n = vehicle.c_rr * total_mass * vehicle.gravity_m_s2 * math.cos(vehicle.grade_rad)
    aero_force_n = 0.5 * vehicle.air_density_kg_m3 * vehicle.cd_area_m2 * np.square(speed_m_s)
    grade_force_n = total_mass * vehicle.gravity_m_s2 * math.sin(vehicle.grade_rad)
    traction_force_n = (effective_mass * accel_m_s2) + rolling_force_n + aero_force_n + grade_force_n

    wheel_inertia_torque_total_nm = motor.driven_wheels * vehicle.wheel_rotational_inertia_kg_m2_per_wheel * (
        accel_m_s2 / motor.wheel_radius_m
    )
    wheel_torque_total_nm = (motor.wheel_radius_m * traction_force_n) + wheel_inertia_torque_total_nm
    wheel_torque_per_motor_nm = wheel_torque_total_nm / motor.driven_wheels
    wheel_power_w = traction_force_n * speed_m_s

    battery_power_w = np.where(
        wheel_power_w > 0.0,
        (wheel_power_w / motor.overall_drive_efficiency()) + vehicle.aux_power_w,
        vehicle.aux_power_w,
    )

    motor_torque_nm = wheel_torque_per_motor_nm / (motor.gear_ratio * motor.gear_efficiency)
    motor_speed_rad_s = motor.gear_ratio * speed_m_s / motor.wheel_radius_m
    motor_current_a = motor_torque_nm / motor.torque_constant_nm_per_a
    battery_current_a = battery_power_w / vehicle.pack_voltage_v

    return {
        "total_mass_kg": float(total_mass),
        "effective_mass_kg": float(effective_mass),
        "rolling_force_n": np.full_like(accel_m_s2, rolling_force_n, dtype=float),
        "aero_force_n": aero_force_n,
        "grade_force_n": np.full_like(accel_m_s2, grade_force_n, dtype=float),
        "traction_force_n": traction_force_n,
        "wheel_torque_total_nm": wheel_torque_total_nm,
        "wheel_torque_per_motor_nm": wheel_torque_per_motor_nm,
        "wheel_inertia_torque_total_nm": wheel_inertia_torque_total_nm,
        "wheel_power_w": wheel_power_w,
        "battery_power_w": battery_power_w,
        "motor_torque_nm": motor_torque_nm,
        "motor_speed_rad_s": motor_speed_rad_s,
        "motor_current_a": motor_current_a,
        "battery_current_a": battery_current_a,
    }


def integrate_energy_wh(power_w: np.ndarray, dt_s: float | np.ndarray) -> float:
    """Integrate power to energy in Wh."""

    if np.isscalar(dt_s):
        return float(np.sum(power_w, dtype=float) * float(dt_s) / 3600.0)
    dt = np.asarray(dt_s, dtype=float)
    return float(np.sum(power_w * dt, dtype=float) / 3600.0)


def iterate_battery_mass(
    session_frame: pd.DataFrame,
    processed_signal: ProcessedGameSignal,
    signal: SignalProcessingAssumptions,
    vehicle: VehicleAssumptions,
    motor: MotorOption,
    battery: BatteryOption,
) -> tuple[BatterySizingResult, pd.DataFrame]:
    """Iterate battery mass until the energy and mass assumptions converge."""

    dt_s = processed_signal.sample_period_s
    accel = session_frame["forward_accel_m_s2"].to_numpy(dtype=float)
    speed = session_frame["surrogate_speed_m_s"].to_numpy(dtype=float)

    battery_mass = vehicle.initial_battery_mass_guess_kg
    history = [battery_mass]
    converged = False
    iterations = 0

    for iteration in range(1, vehicle.max_iterations + 1):
        dynamics = compute_longitudinal_dynamics(accel, speed, vehicle, motor, battery_mass)
        usable_energy_wh = integrate_energy_wh(dynamics["battery_power_w"], dt_s)
        nominal_energy_wh = usable_energy_wh / battery.usable_fraction
        updated_mass = nominal_energy_wh / battery.specific_energy_wh_per_kg
        history.append(updated_mass)
        iterations = iteration
        if abs(updated_mass - battery_mass) <= vehicle.convergence_tol_kg:
            battery_mass = updated_mass
            converged = True
            break
        battery_mass = updated_mass

    final_dynamics = compute_longitudinal_dynamics(accel, speed, vehicle, motor, battery_mass)
    usable_energy_wh = integrate_energy_wh(final_dynamics["battery_power_w"], dt_s)
    nominal_energy_wh = usable_energy_wh / battery.usable_fraction
    nominal_capacity_ah = nominal_energy_wh / vehicle.pack_voltage_v
    session_hours = vehicle_session_minutes(session_frame, dt_s) / 60.0
    battery_c_rate = (
        final_dynamics["battery_current_a"] / nominal_capacity_ah if nominal_capacity_ah > 0.0 else np.zeros_like(accel)
    )
    peak_battery_c_rate = float(np.max(battery_c_rate))

    result = BatterySizingResult(
        game_name=processed_signal.game_name,
        motor_name=motor.name,
        battery_name=battery.name,
        representative_profile_minutes=signal.representative_minutes,
        session_duration_hours=signal.session_hours,
        usable_energy_wh=usable_energy_wh,
        nominal_energy_wh=nominal_energy_wh,
        nominal_capacity_ah=nominal_capacity_ah,
        battery_mass_kg=battery_mass,
        total_vehicle_mass_kg=float(final_dynamics["total_mass_kg"]),
        average_battery_power_w=usable_energy_wh / session_hours,
        peak_battery_power_w=float(np.max(final_dynamics["battery_power_w"])),
        peak_traction_force_n=float(np.max(np.abs(final_dynamics["traction_force_n"]))),
        peak_wheel_torque_nm=float(np.max(np.abs(final_dynamics["wheel_torque_total_nm"]))),
        peak_motor_torque_nm=float(np.max(np.abs(final_dynamics["motor_torque_nm"]))),
        peak_motor_current_a=float(np.max(np.abs(final_dynamics["motor_current_a"]))),
        peak_battery_current_a=float(np.max(final_dynamics["battery_current_a"])),
        peak_battery_c_rate=peak_battery_c_rate,
        motor_continuous_current_violation=bool(np.any(np.abs(final_dynamics["motor_current_a"]) > motor.continuous_current_a)),
        motor_peak_current_violation=bool(np.any(np.abs(final_dynamics["motor_current_a"]) > motor.peak_current_a)),
        battery_continuous_c_violation=bool(np.any(battery_c_rate > battery.continuous_c)),
        battery_peak_c_violation=bool(np.any(battery_c_rate > battery.peak_c)),
        battery_iteration_count=iterations,
        converged=converged,
        forward_axis_x=processed_signal.forward_axis[0],
        forward_axis_y=processed_signal.forward_axis[1],
        forward_axis_z=processed_signal.forward_axis[2],
        winsor_limit_m_s2=processed_signal.winsor_limit_m_s2,
        battery_mass_history_kg=tuple(round(value, 6) for value in history),
        notes=(
            "Surrogate speed comes from filtered under-leg IMU acceleration with clipping and no zero-velocity resets.",
            "Battery power uses no-regeneration logic; negative wheel power falls back to auxiliary load only.",
        ),
    )

    scenario_frame = session_frame.copy()
    for column, values in final_dynamics.items():
        if isinstance(values, np.ndarray):
            scenario_frame[column] = values
    return result, scenario_frame


def vehicle_session_minutes(session_frame: pd.DataFrame, dt_s: float) -> float:
    """Return the total represented session duration in minutes."""

    return (len(session_frame) * dt_s) / 60.0


def summarize_results(results: list[BatterySizingResult]) -> pd.DataFrame:
    """Convert scenario results to a flat summary dataframe."""

    return pd.DataFrame([result.to_summary_row() for result in results]).sort_values(
        ["game_name", "motor_name", "battery_name"]
    )


def _scenario_slug(game_name: str, motor_name: str, battery_name: str) -> str:
    safe = f"{game_name}__{motor_name}__{battery_name}"
    return safe.replace(" ", "_")


def _write_summary_outputs(
    output_dir: Path,
    vehicle: VehicleAssumptions,
    signal: SignalProcessingAssumptions,
    motors: list[MotorOption],
    batteries: list[BatteryOption],
    results: list[BatterySizingResult],
) -> None:
    summary = summarize_results(results)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)

    payload = {
        "assumptions": {
            "vehicle": asdict(vehicle),
            "signal_processing": asdict(signal),
            "motors": [asdict(motor) for motor in motors],
            "batteries": [asdict(battery) for battery in batteries],
            "uncertainty_notes": [
                "The forward axis is inferred from horizontal principal motion because the phone yaw orientation is unknown.",
                "The derived speed is a bounded surrogate for battery sizing, not ground-truth wheelchair velocity.",
                "The default scenario assumes flat court, no regenerative braking, and constant auxiliary load.",
            ],
        },
        "results": [asdict(result) for result in results],
    }
    (output_dir / "scenario_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _create_game_plots(
    output_dir: Path,
    representative_profiles: dict[str, pd.DataFrame],
    battery_power_traces: dict[str, list[tuple[str, np.ndarray]]],
    representative_minutes: float,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for game_name, profile in representative_profiles.items():
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

        representative_mask = profile["profile_elapsed_min"].to_numpy(dtype=float) < representative_minutes
        plot_slice = slice(None, None, 10)
        accel_minutes = profile.loc[representative_mask, "profile_elapsed_min"].to_numpy(dtype=float)[plot_slice]
        axes[0].plot(
            accel_minutes,
            profile.loc[representative_mask, "raw_forward_accel_m_s2"].to_numpy(dtype=float)[plot_slice],
            label="Raw forward accel",
            alpha=0.45,
        )
        axes[0].plot(
            accel_minutes,
            profile.loc[representative_mask, "forward_accel_m_s2"].to_numpy(dtype=float)[plot_slice],
            label="Filtered forward accel",
            linewidth=1.5,
        )
        axes[0].set_title(f"{game_name}: representative {representative_minutes:.0f}-minute acceleration profile")
        axes[0].set_ylabel("Acceleration (m/s^2)")
        axes[0].set_xlabel("Profile time (min)")
        axes[0].legend()

        session_minutes = profile["profile_elapsed_min"].to_numpy(dtype=float)
        axes[1].plot(session_minutes, profile["surrogate_speed_m_s"].to_numpy(dtype=float), color="tab:green")
        axes[1].set_title(f"{game_name}: surrogate speed over 2-hour session")
        axes[1].set_ylabel("Speed (m/s)")
        axes[1].set_xlabel("Session time (min)")

        for label, battery_power in battery_power_traces[game_name]:
            axes[2].plot(session_minutes, battery_power, linewidth=1.0, label=label)
        axes[2].set_title(f"{game_name}: battery power traces by scenario")
        axes[2].set_ylabel("Battery power (W)")
        axes[2].set_xlabel("Session time (min)")
        axes[2].legend(fontsize=8, ncol=2, loc="upper right")

        fig.tight_layout()
        fig.savefig(plots_dir / f"{game_name}_battery_sizing.png", dpi=150)
        plt.close(fig)


def print_console_summary(results: list[BatterySizingResult]) -> None:
    """Print a concise console report grouped by game."""

    grouped: dict[str, list[BatterySizingResult]] = defaultdict(list)
    for result in results:
        grouped[result.game_name].append(result)

    for game_name in sorted(grouped):
        rows = sorted(grouped[game_name], key=lambda item: (item.motor_name, item.battery_name))
        print(f"\n{game_name}")
        print("  Peak force / torque / current constraints")
        for row in rows:
            print(
                "   "
                f"{row.motor_name} + {row.battery_name}: "
                f"force={row.peak_traction_force_n:.1f} N, "
                f"wheel_torque={row.peak_wheel_torque_nm:.1f} Nm, "
                f"motor_current={row.peak_motor_current_a:.1f} A, "
                f"motor_cont_violation={row.motor_continuous_current_violation}, "
                f"motor_peak_violation={row.motor_peak_current_violation}"
            )
        print("  Total energy / Wh / Ah constraints")
        for row in rows:
            print(
                "   "
                f"{row.motor_name} + {row.battery_name}: "
                f"usable={row.usable_energy_wh:.1f} Wh, "
                f"nominal={row.nominal_energy_wh:.1f} Wh, "
                f"capacity={row.nominal_capacity_ah:.1f} Ah @ 48 V, "
                f"battery_mass={row.battery_mass_kg:.2f} kg, "
                f"peak_batt_power={row.peak_battery_power_w:.1f} W, "
                f"peak_c={row.peak_battery_c_rate:.2f}C"
            )
        print("  Assumption-driven uncertainty notes")
        print("   Forward direction is inferred from principal horizontal motion because phone yaw is unknown.")
        print("   Speed is a bounded surrogate derived from IMU acceleration, not measured wheel speed.")
        print("   No regeneration, flat court, and constant auxiliary load are assumed in this v1 model.")


def summarize_motor_requirements(
    processed_signal: ProcessedGameSignal,
    session_frame: pd.DataFrame,
    vehicle: VehicleAssumptions,
    motor: MotorOption,
    voltage_candidates_v: list[float],
    project_peak_torque_per_wheel_nm: float = 39.6,
    project_speed_target_mph: float = 11.0,
) -> list[MotorRequirementResult]:
    """Summarize load and current requirements across candidate pack voltages."""

    dt_s = processed_signal.sample_period_s
    accel = session_frame["forward_accel_m_s2"].to_numpy(dtype=float)
    speed = session_frame["surrogate_speed_m_s"].to_numpy(dtype=float)
    dynamics = compute_longitudinal_dynamics(accel, speed, vehicle, motor, battery_mass_kg=0.0)

    session_hours = vehicle_session_minutes(session_frame, dt_s) / 60.0
    session_energy_wh = integrate_energy_wh(dynamics["battery_power_w"], dt_s)
    avg_electrical_power_w = session_energy_wh / session_hours

    rated_output_speed_m_s = 0.0
    if motor.rated_speed_rpm is not None:
        rated_output_speed_m_s = (
            (motor.rated_speed_rpm / motor.gear_ratio)
            * (2.0 * math.pi * motor.wheel_radius_m)
            / 60.0
        )
    rated_output_speed_mph = rated_output_speed_m_s * 2.2369362920544
    meets_motor_peak_torque = (
        float(np.max(np.abs(dynamics["motor_torque_nm"]))) <= motor.peak_torque_nm
        if motor.peak_torque_nm is not None
        else None
    )

    rows: list[MotorRequirementResult] = []
    for voltage_v in voltage_candidates_v:
        peak_pack_current_a = float(np.max(dynamics["battery_power_w"]) / voltage_v)
        rows.append(
            MotorRequirementResult(
                game_name=processed_signal.game_name,
                voltage_v=voltage_v,
                motor_name=motor.name,
                session_energy_wh=session_energy_wh,
                average_electrical_power_w=avg_electrical_power_w,
                peak_electrical_power_w=float(np.max(dynamics["battery_power_w"])),
                peak_pack_current_a=peak_pack_current_a,
                peak_acceleration_m_s2=float(np.max(accel)),
                peak_speed_m_s=float(np.max(speed)),
                peak_speed_mph=float(np.max(speed) * 2.2369362920544),
                peak_traction_force_n=float(np.max(np.abs(dynamics["traction_force_n"]))),
                peak_wheel_torque_total_nm=float(np.max(np.abs(dynamics["wheel_torque_total_nm"]))),
                peak_wheel_torque_per_motor_nm=float(np.max(np.abs(dynamics["wheel_torque_per_motor_nm"]))),
                required_peak_motor_torque_nm=float(np.max(np.abs(dynamics["motor_torque_nm"]))),
                required_peak_motor_current_a=float(np.max(np.abs(dynamics["motor_current_a"]))),
                required_continuous_motor_current_a=float(np.percentile(np.abs(dynamics["motor_current_a"]), 95)),
                clipped_positive_samples=processed_signal.clipped_positive_samples,
                clipped_negative_samples=processed_signal.clipped_negative_samples,
                rated_output_speed_m_s=rated_output_speed_m_s,
                rated_output_speed_mph=rated_output_speed_mph,
                meets_11mph_speed_target=rated_output_speed_mph >= project_speed_target_mph,
                meets_project_peak_torque_target=float(np.max(np.abs(dynamics["wheel_torque_per_motor_nm"])))
                >= project_peak_torque_per_wheel_nm,
                meets_motor_peak_torque=meets_motor_peak_torque,
                notes=(
                    "Voltage sweep changes pack current, not required electrical power.",
                    "Acceleration demand is derived from a signed filtered IMU signal with unrealistic peaks clipped.",
                ),
            )
        )
    return rows


def run_battery_sizing_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    vehicle: VehicleAssumptions,
    signal: SignalProcessingAssumptions,
    motors: list[MotorOption],
    batteries: list[BatteryOption],
    write_timeseries: bool = True,
    write_plots: bool = True,
) -> list[BatterySizingResult]:
    """Run the end-to-end battery sizing workflow across all cleaned games."""

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if write_timeseries:
        (output_root / "timeseries").mkdir(parents=True, exist_ok=True)

    results: list[BatterySizingResult] = []
    representative_profiles: dict[str, pd.DataFrame] = {}
    battery_power_traces: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)

    for csv_path in sorted(input_root.glob("*.csv")):
        processed = preprocess_game_csv(csv_path, signal, gravity_m_s2=vehicle.gravity_m_s2)
        session_frame = build_representative_session(processed, signal)
        representative_profiles[processed.game_name] = session_frame

        for motor in motors:
            for battery in batteries:
                result, scenario_frame = iterate_battery_mass(
                    session_frame,
                    processed,
                    signal,
                    vehicle,
                    motor,
                    battery,
                )
                results.append(result)

                label = f"{motor.name} + {battery.name}"
                battery_power_traces[processed.game_name].append(
                    (label, scenario_frame["battery_power_w"].to_numpy(dtype=float))
                )

                if write_timeseries:
                    slug = _scenario_slug(processed.game_name, motor.name, battery.name)
                    scenario_frame.to_parquet(output_root / "timeseries" / f"{slug}.parquet", index=False)

    _write_summary_outputs(output_root, vehicle, signal, motors, batteries, results)
    if write_plots:
        _create_game_plots(output_root, representative_profiles, battery_power_traces, signal.representative_minutes)
    return results
