# Motor Selection Analysis

This folder now includes a Python workflow that reproduces the calculations from `Kraken X60_raw.xlsx` without relying on spreadsheet formulas.

## Files

- `analyze_motor.py`: loads the motor data and wheelchair system data, performs the fitted calculations, and writes result files plus charts.
- `Kraken X60_raw.xlsx`: the original workbook used to reverse-engineer the calculations.
- `requirements.txt`: Python packages needed by the script.

## Expected Input Columns

Motor input:

- `RPM`
- `Efficiency %`
- `Power (W)`
- `Torque (Nm)`
- `Current (A)`

Wheelchair system input:

- `Desired acceleration`
- `Required Torque Nm`

The script can read either CSV or Excel input. It scans the file for the required column headers, so the two datasets can live in separate files or in the same workbook.

## Usage

```powershell
python motor-selection\analyze_motor.py
```

The command above uses `Kraken X60_raw.xlsx` as both the motor and system source if no other files are supplied.

```powershell
python motor-selection\analyze_motor.py `
  --motor-input path\to\motor_data.xlsx `
  --system-input path\to\wheelchair_requirements.csv `
  --cruising-wheel-rpm 250 `
  --gear-reduction 23 `
  --peak-acceleration 2.85 `
  --motor-free-speed-rpm 6000 `
  --output-dir motor-selection\output
```

## Outputs

The script writes:

- `motor_analysis.csv`: motor input with `RPM^2` and fitted columns added.
- `system_analysis.csv`: wheelchair input with fitted required torque added.
- `summary.json`: fitted coefficients and operating-point results.
- `summary.csv`: the same summary in flat key/value form.
- `motor_performance.png`: raw motor curves plus fitted torque, power, and current curves.
- `system_requirement.png`: required wheel torque versus desired acceleration with linear fit.
- `wheel_operating_points.png`: available wheel torque versus wheel RPM with cruising and peak markers.
- `gear_ratio_sweep_cruising.png`: cruising-point wheel torque, motor current, and power overlaid on one chart versus gear ratio from `1:1` to the free-speed limit.
- `gear_ratio_sweep_peak.png`: peak-point wheel torque, motor current, and power overlaid on one chart versus gear ratio from `1:1` to the free-speed limit.
