from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CRUISING_WHEEL_RPM = 250.0
DEFAULT_GEAR_REDUCTION = 23.0
DEFAULT_PEAK_ACCELERATION = 2.85
DEFAULT_MOTOR_FREE_SPEED_RPM = 6000.0


MOTOR_COLUMNS = {
    "RPM": {"rpm"},
    "Efficiency %": {"efficiency", "efficiencypercent", "efficiencypct"},
    "Power (W)": {"power", "powerw", "powerwatts"},
    "Torque (Nm)": {"torque", "torquenm", "torquenmeter", "torquenewtonmeters"},
    "Current (A)": {"current", "currenta", "currentamp", "currentamps", "currentampere"},
}

SYSTEM_COLUMNS = {
    "Desired acceleration": {"desiredacceleration", "acceleration", "targetacceleration"},
    "Required Torque Nm": {
        "requiredtorque",
        "requiredtorquenm",
        "requiredwheeltorque",
        "requiredwheeltorquenm",
    },
}


@dataclass
class LinearFit:
    slope: float
    intercept: float
    r_squared: float

    def predict(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.slope * np.asarray(x) + self.intercept


@dataclass
class PowerFit:
    rpm_squared_coefficient: float
    rpm_coefficient: float
    r_squared: float

    def predict(self, rpm: np.ndarray | float) -> np.ndarray | float:
        rpm_values = np.asarray(rpm)
        return self.rpm_squared_coefficient * rpm_values**2 + self.rpm_coefficient * rpm_values


def normalize_header(header: Any) -> str:
    text = str(header).strip().lower().replace("%", " percent ")
    return re.sub(r"[^a-z0-9]+", "", text)


def compute_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual_sum = np.sum((actual - predicted) ** 2)
    total_sum = np.sum((actual - np.mean(actual)) ** 2)
    if math.isclose(total_sum, 0.0):
        return 1.0 if math.isclose(residual_sum, 0.0) else 0.0
    return 1.0 - residual_sum / total_sum


def standardize_frame(frame: pd.DataFrame, aliases: dict[str, set[str]]) -> pd.DataFrame | None:
    rename_map: dict[str, str] = {}

    for column in frame.columns:
        normalized = normalize_header(column)
        for canonical, accepted in aliases.items():
            if normalized in accepted and canonical not in rename_map.values():
                rename_map[column] = canonical
                break

    standardized = frame.rename(columns=rename_map)
    if any(column not in standardized.columns for column in aliases):
        return None

    numeric = standardized[list(aliases)].apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.empty:
        return None

    return numeric.reset_index(drop=True)


def load_table(path: Path, aliases: dict[str, set[str]], sheet_name: str | None = None) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    frames: dict[str, pd.DataFrame]

    if suffix == ".csv":
        frames = {path.name: pd.read_csv(path)}
    elif suffix in {".xls", ".xlsx", ".xlsm"}:
        if sheet_name:
            frames = {sheet_name: pd.read_excel(path, sheet_name=sheet_name)}
        else:
            frames = pd.read_excel(path, sheet_name=None)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    for source_name, frame in frames.items():
        standardized = standardize_frame(frame, aliases)
        if standardized is not None:
            return standardized, source_name

    required = ", ".join(aliases)
    raise ValueError(f"Could not find columns [{required}] in {path}")


def fit_linear(x: pd.Series, y: pd.Series) -> LinearFit:
    slope, intercept = np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), 1)
    predicted = slope * x.to_numpy(dtype=float) + intercept
    return LinearFit(float(slope), float(intercept), compute_r_squared(y.to_numpy(dtype=float), predicted))


def fit_power_curve(rpm: pd.Series, power: pd.Series) -> PowerFit:
    rpm_values = rpm.to_numpy(dtype=float)
    power_values = power.to_numpy(dtype=float)
    design_matrix = np.column_stack((rpm_values**2, rpm_values))
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, power_values, rcond=None)
    predicted = design_matrix @ coefficients
    return PowerFit(
        rpm_squared_coefficient=float(coefficients[0]),
        rpm_coefficient=float(coefficients[1]),
        r_squared=compute_r_squared(power_values, predicted),
    )


def build_summary_rows(data: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(build_summary_rows(value, flat_key))
        else:
            rows.append({"metric": flat_key, "value": value})
    return rows


def add_fitted_columns(
    motor_df: pd.DataFrame,
    system_df: pd.DataFrame,
    power_fit: PowerFit,
    torque_fit: LinearFit,
    current_fit: LinearFit,
    required_torque_fit: LinearFit,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    motor_output = motor_df.copy()
    motor_output["RPM^2"] = motor_output["RPM"] ** 2
    motor_output["Power Fit (W)"] = power_fit.predict(motor_output["RPM"].to_numpy(dtype=float))
    motor_output["Torque Fit (Nm)"] = torque_fit.predict(motor_output["RPM"].to_numpy(dtype=float))
    motor_output["Current Fit (A)"] = current_fit.predict(motor_output["RPM"].to_numpy(dtype=float))

    system_output = system_df.copy()
    system_output["Required Torque Fit (Nm)"] = required_torque_fit.predict(
        system_output["Desired acceleration"].to_numpy(dtype=float)
    )
    return motor_output, system_output


def plot_motor_performance(
    motor_df: pd.DataFrame,
    power_fit: PowerFit,
    torque_fit: LinearFit,
    current_fit: LinearFit,
    output_path: Path,
) -> None:
    rpm = motor_df["RPM"].to_numpy(dtype=float)
    rpm_span = np.linspace(rpm.min(), rpm.max(), 500)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.flatten()

    axes[0].plot(rpm, motor_df["Efficiency %"], color="#2a6f97", linewidth=2)
    axes[0].set_title("Efficiency")
    axes[0].set_xlabel("RPM")
    axes[0].set_ylabel("Efficiency (%)")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(rpm, motor_df["Power (W)"], color="#d1495b", s=24, label="Raw data")
    axes[1].plot(rpm_span, power_fit.predict(rpm_span), color="#1d3557", linewidth=2, label="Fit")
    axes[1].set_title("Power")
    axes[1].set_xlabel("RPM")
    axes[1].set_ylabel("Power (W)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    axes[2].scatter(rpm, motor_df["Torque (Nm)"], color="#2b9348", s=24, label="Raw data")
    axes[2].plot(rpm_span, torque_fit.predict(rpm_span), color="#264653", linewidth=2, label="Fit")
    axes[2].set_title("Torque")
    axes[2].set_xlabel("RPM")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(frameon=False)

    axes[3].scatter(rpm, motor_df["Current (A)"], color="#f4a261", s=24, label="Raw data")
    axes[3].plot(rpm_span, current_fit.predict(rpm_span), color="#6d597a", linewidth=2, label="Fit")
    axes[3].set_title("Current")
    axes[3].set_xlabel("RPM")
    axes[3].set_ylabel("Current (A)")
    axes[3].grid(alpha=0.3)
    axes[3].legend(frameon=False)

    figure.suptitle("Motor Performance Curves", fontsize=16)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_system_requirement(
    system_df: pd.DataFrame,
    required_torque_fit: LinearFit,
    peak_acceleration: float,
    peak_wheel_torque: float,
    output_path: Path,
) -> None:
    acceleration = system_df["Desired acceleration"].to_numpy(dtype=float)
    span = np.linspace(acceleration.min(), acceleration.max(), 300)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.scatter(
        acceleration,
        system_df["Required Torque Nm"],
        color="#457b9d",
        s=28,
        label="Input data",
    )
    axis.plot(span, required_torque_fit.predict(span), color="#e76f51", linewidth=2, label="Fit")
    axis.scatter(
        [peak_acceleration],
        [peak_wheel_torque],
        color="#1d3557",
        s=70,
        marker="D",
        label="Peak target",
        zorder=5,
    )
    axis.set_title("Wheel Torque Requirement")
    axis.set_xlabel("Desired acceleration")
    axis.set_ylabel("Required Torque (Nm)")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_wheel_operating_points(
    rpm: pd.Series,
    torque_fit: LinearFit,
    gear_reduction: float,
    cruising_wheel_rpm: float,
    cruising_wheel_torque: float,
    peak_wheel_rpm: float,
    peak_wheel_torque: float,
    output_path: Path,
) -> None:
    rpm_span = np.linspace(rpm.min(), rpm.max(), 500)
    wheel_rpm_span = rpm_span / gear_reduction
    wheel_torque_span = torque_fit.predict(rpm_span) * gear_reduction

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(wheel_rpm_span, wheel_torque_span, color="#264653", linewidth=2, label="Available wheel torque")
    axis.scatter(
        [cruising_wheel_rpm],
        [cruising_wheel_torque],
        color="#2a9d8f",
        s=70,
        marker="o",
        label="Cruising point",
        zorder=5,
    )
    axis.scatter(
        [peak_wheel_rpm],
        [peak_wheel_torque],
        color="#e76f51",
        s=80,
        marker="D",
        label="Peak point",
        zorder=5,
    )
    axis.set_title("Wheel-Side Operating Points")
    axis.set_xlabel("Wheel RPM")
    axis.set_ylabel("Wheel Torque (Nm)")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_gear_ratio_sweep(
    wheel_rpm: float,
    motor_free_speed_rpm: float,
    torque_fit: LinearFit,
    current_fit: LinearFit,
    power_fit: PowerFit,
    selected_gear_reduction: float,
    title: str,
    output_path: Path,
) -> None:
    max_gear_ratio = motor_free_speed_rpm / wheel_rpm
    gear_ratios = np.linspace(1.0, max_gear_ratio, 500)
    motor_rpm = gear_ratios * wheel_rpm
    wheel_torque = torque_fit.predict(motor_rpm) * gear_ratios
    current = current_fit.predict(motor_rpm)
    power = power_fit.predict(motor_rpm)

    selected_motor_rpm = selected_gear_reduction * wheel_rpm
    selected_wheel_torque = float(torque_fit.predict(selected_motor_rpm) * selected_gear_reduction)
    selected_current = float(current_fit.predict(selected_motor_rpm))
    selected_power = float(power_fit.predict(selected_motor_rpm))

    torque_color = "#2a9d8f"
    current_color = "#e76f51"
    power_color = "#1d3557"
    selection_color = "#264653"

    figure, torque_axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    current_axis = torque_axis.twinx()
    power_axis = torque_axis.twinx()
    power_axis.spines["right"].set_position(("axes", 1.12))
    power_axis.patch.set_visible(False)

    torque_line = torque_axis.plot(
        gear_ratios, wheel_torque, color=torque_color, linewidth=2.5, label="Wheel torque (Nm)"
    )[0]
    current_line = current_axis.plot(
        gear_ratios, current, color=current_color, linewidth=2.5, label="Current (A)"
    )[0]
    power_line = power_axis.plot(
        gear_ratios, power, color=power_color, linewidth=2.5, label="Power (W)"
    )[0]

    for axis in (torque_axis, current_axis, power_axis):
        axis.axvline(selected_gear_reduction, color=selection_color, linestyle="--", linewidth=1.25)

    torque_axis.scatter([selected_gear_reduction], [selected_wheel_torque], color=torque_color, s=60, zorder=5)
    current_axis.scatter([selected_gear_reduction], [selected_current], color=current_color, s=60, zorder=5)
    power_axis.scatter([selected_gear_reduction], [selected_power], color=power_color, s=60, zorder=5)

    torque_axis.set_title(title)
    torque_axis.set_xlabel("Gear Ratio")
    torque_axis.set_ylabel("Wheel Torque (Nm)", color=torque_color)
    current_axis.set_ylabel("Current (A)", color=current_color)
    power_axis.set_ylabel("Power (W)", color=power_color)
    torque_axis.tick_params(axis="y", colors=torque_color)
    current_axis.tick_params(axis="y", colors=current_color)
    power_axis.tick_params(axis="y", colors=power_color)
    torque_axis.grid(alpha=0.3)

    lines = [torque_line, current_line, power_line]
    torque_axis.legend(lines, [line.get_label() for line in lines], loc="upper center", ncol=3, frameon=False)

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def create_parser(default_input: Path | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replicate the motor-selection spreadsheet in Python.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--motor-input",
        type=Path,
        default=default_input,
        help="CSV or Excel file containing the motor performance data.",
    )
    parser.add_argument(
        "--system-input",
        type=Path,
        help="CSV or Excel file containing desired acceleration versus required torque.",
    )
    parser.add_argument("--motor-sheet", help="Optional Excel sheet name for the motor data.")
    parser.add_argument("--system-sheet", help="Optional Excel sheet name for the system data.")
    parser.add_argument(
        "--cruising-wheel-rpm",
        type=float,
        default=DEFAULT_CRUISING_WHEEL_RPM,
        help="Cruising wheel RPM used to evaluate the fitted curves.",
    )
    parser.add_argument(
        "--gear-reduction",
        type=float,
        default=DEFAULT_GEAR_REDUCTION,
        help="Gear reduction ratio between motor RPM and wheel RPM.",
    )
    parser.add_argument(
        "--peak-acceleration",
        type=float,
        default=DEFAULT_PEAK_ACCELERATION,
        help="Desired peak acceleration used to evaluate the required wheel torque.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(default_input.parent / "output") if default_input else Path("output"),
        help="Directory where result files and charts will be written.",
    )
    parser.add_argument(
        "--motor-free-speed-rpm",
        type=float,
        default=DEFAULT_MOTOR_FREE_SPEED_RPM,
        help="Motor free speed used to cap the gear-ratio sweep.",
    )
    return parser


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    sample_workbook = script_dir / "Kraken X60_raw.xlsx"
    parser = create_parser(sample_workbook if sample_workbook.exists() else None)
    args = parser.parse_args()

    if args.motor_input is None:
        parser.error("--motor-input is required when no default workbook is present.")

    system_input = args.system_input or args.motor_input
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    motor_df, motor_source = load_table(args.motor_input, MOTOR_COLUMNS, args.motor_sheet)
    system_df, system_source = load_table(system_input, SYSTEM_COLUMNS, args.system_sheet)

    motor_df = motor_df.sort_values("RPM").reset_index(drop=True)
    system_df = system_df.sort_values("Desired acceleration").reset_index(drop=True)

    power_fit = fit_power_curve(motor_df["RPM"], motor_df["Power (W)"])
    torque_fit = fit_linear(motor_df["RPM"], motor_df["Torque (Nm)"])
    current_fit = fit_linear(motor_df["RPM"], motor_df["Current (A)"])
    required_torque_fit = fit_linear(system_df["Desired acceleration"], system_df["Required Torque Nm"])

    cruising_motor_rpm = args.cruising_wheel_rpm * args.gear_reduction
    cruising_wheel_torque = float(torque_fit.predict(cruising_motor_rpm) * args.gear_reduction)
    cruising_power = float(power_fit.predict(cruising_motor_rpm))
    cruising_current = float(current_fit.predict(cruising_motor_rpm))

    peak_wheel_torque = float(required_torque_fit.predict(args.peak_acceleration))
    peak_motor_torque = peak_wheel_torque / args.gear_reduction
    if math.isclose(torque_fit.slope, 0.0):
        peak_torque_rpm = math.nan
    else:
        peak_torque_rpm = float((peak_motor_torque - torque_fit.intercept) / torque_fit.slope)
    peak_wheel_rpm = peak_torque_rpm / args.gear_reduction if not math.isnan(peak_torque_rpm) else math.nan
    peak_power = float(power_fit.predict(peak_torque_rpm)) if not math.isnan(peak_torque_rpm) else math.nan
    peak_current = float(current_fit.predict(peak_torque_rpm)) if not math.isnan(peak_torque_rpm) else math.nan
    if math.isnan(peak_wheel_rpm) or peak_wheel_rpm <= 0:
        raise ValueError("Peak wheel RPM is invalid, so the peak gear-ratio sweep cannot be generated.")

    motor_output, system_output = add_fitted_columns(
        motor_df, system_df, power_fit, torque_fit, current_fit, required_torque_fit
    )

    summary = {
        "inputs": {
            "motor_input": str(args.motor_input.resolve()),
            "motor_source": motor_source,
            "system_input": str(system_input.resolve()),
            "system_source": system_source,
            "cruising_wheel_rpm": args.cruising_wheel_rpm,
            "gear_reduction": args.gear_reduction,
            "peak_acceleration": args.peak_acceleration,
            "motor_free_speed_rpm": args.motor_free_speed_rpm,
        },
        "fits": {
            "power_curve": asdict(power_fit),
            "motor_torque_curve": asdict(torque_fit),
            "motor_current_curve": asdict(current_fit),
            "required_wheel_torque_curve": asdict(required_torque_fit),
        },
        "operating_points": {
            "cruising_motor_rpm": cruising_motor_rpm,
            "cruising_wheel_torque_nm": cruising_wheel_torque,
            "cruising_power_w": cruising_power,
            "cruising_current_a": cruising_current,
            "peak_wheel_torque_nm": peak_wheel_torque,
            "peak_motor_torque_nm": peak_motor_torque,
            "peak_torque_rpm_point": peak_torque_rpm,
            "peak_wheel_rpm_point": peak_wheel_rpm,
            "peak_power_w": peak_power,
            "peak_current_a": peak_current,
        },
        "data_ranges": {
            "motor_rpm_min": float(motor_df["RPM"].min()),
            "motor_rpm_max": float(motor_df["RPM"].max()),
            "gear_ratio_sweep_min": 1.0,
            "cruising_gear_ratio_sweep_max": float(args.motor_free_speed_rpm / args.cruising_wheel_rpm),
            "peak_gear_ratio_sweep_max": float(args.motor_free_speed_rpm / peak_wheel_rpm),
            "peak_rpm_within_motor_data": bool(
                not math.isnan(peak_torque_rpm)
                and motor_df["RPM"].min() <= peak_torque_rpm <= motor_df["RPM"].max()
            ),
            "cruising_rpm_within_motor_data": bool(
                motor_df["RPM"].min() <= cruising_motor_rpm <= motor_df["RPM"].max()
            ),
        },
    }

    motor_output.to_csv(output_dir / "motor_analysis.csv", index=False)
    system_output.to_csv(output_dir / "system_analysis.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="ascii") as handle:
        json.dump(summary, handle, indent=2)
    pd.DataFrame(build_summary_rows(summary)).to_csv(output_dir / "summary.csv", index=False)

    plot_motor_performance(
        motor_output,
        power_fit,
        torque_fit,
        current_fit,
        output_dir / "motor_performance.png",
    )
    plot_system_requirement(
        system_output,
        required_torque_fit,
        args.peak_acceleration,
        peak_wheel_torque,
        output_dir / "system_requirement.png",
    )
    plot_wheel_operating_points(
        motor_df["RPM"],
        torque_fit,
        args.gear_reduction,
        args.cruising_wheel_rpm,
        cruising_wheel_torque,
        peak_wheel_rpm,
        peak_wheel_torque,
        output_dir / "wheel_operating_points.png",
    )
    legacy_sweep_plot = output_dir / "gear_ratio_sweep.png"
    if legacy_sweep_plot.exists():
        legacy_sweep_plot.unlink()

    plot_gear_ratio_sweep(
        args.cruising_wheel_rpm,
        args.motor_free_speed_rpm,
        torque_fit,
        current_fit,
        power_fit,
        args.gear_reduction,
        "Cruising Response vs Gear Ratio",
        output_dir / "gear_ratio_sweep_cruising.png",
    )
    plot_gear_ratio_sweep(
        peak_wheel_rpm,
        args.motor_free_speed_rpm,
        torque_fit,
        current_fit,
        power_fit,
        args.gear_reduction,
        "Peak Response vs Gear Ratio",
        output_dir / "gear_ratio_sweep_peak.png",
    )

    print(f"Analysis complete. Results written to {output_dir.resolve()}")
    print(
        "Cruising point:",
        f"motor_rpm={cruising_motor_rpm:.3f},",
        f"wheel_torque_nm={cruising_wheel_torque:.6f},",
        f"power_w={cruising_power:.6f},",
        f"current_a={cruising_current:.6f}",
    )
    print(
        "Peak point:",
        f"wheel_torque_nm={peak_wheel_torque:.6f},",
        f"motor_torque_nm={peak_motor_torque:.6f},",
        f"rpm={peak_torque_rpm:.6f},",
        f"power_w={peak_power:.6f},",
        f"current_a={peak_current:.6f}",
    )


if __name__ == "__main__":
    main()
