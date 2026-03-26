# seniorDesign

Starter repository for processing large volumes of IMU data collected from two phones.

## Goals

- Ingest raw IMU exports from each phone
- Normalize timestamps, units, and sensor labels
- Align streams across devices
- Build repeatable preprocessing and analysis steps
- Keep raw data out of git while versioning code and metadata
- Estimate wheelchair battery requirements from processed gameplay IMU traces

## Project Layout

```text
.
├── data/
│   ├── raw/         # original exports from each phone
│   ├── interim/     # cleaned but not fully modeled data
│   └── processed/   # analysis-ready outputs
├── notebooks/       # exploration and validation
├── scripts/         # runnable entrypoints
├── src/imu_pipeline/
│   ├── io.py        # loading and path helpers
│   └── schema.py    # shared column names and expectations
├── tests/
├── pyproject.toml
└── README.md
```

## Suggested Raw Data Convention

Keep one folder per collection session, and separate each phone inside it:

```text
data/raw/
  2026-03-25-walk-test/
    phone_a/
    phone_b/
```

That makes it easier to pair recordings and trace provenance later.

## Quick Start

```bash
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/inspect_dataset.py --help
pytest
```

## Battery Sizing Workflow

Run the end-to-end battery sizing analysis from the cleaned gameplay CSVs:

```bash
source .venv/bin/activate
python scripts/run_battery_sizing.py
```

This writes:

- `data/processed/battery_sizing/scenario_summary.csv`
- `data/processed/battery_sizing/scenario_summary.json`
- `data/processed/battery_sizing/timeseries/*.parquet`
- `data/processed/battery_sizing/plots/*.png`

The model uses the processed IMU trace to build:

```text
filtered acceleration -> surrogate speed -> traction force -> wheel torque
-> motor torque/current -> battery power -> integrated battery energy
```

The default v1 assumptions are intentionally simple and editable near the top of
`scripts/run_battery_sizing.py`.

## Next Good Steps

1. Decide the exact export format from each phone app.
2. Add one or two representative sample files outside git.
3. Implement dataset-specific parsers in `src/imu_pipeline/io.py`.
4. Add tests for timestamp parsing and axis naming.
